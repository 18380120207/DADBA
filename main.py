#main.py
"""
| 阶段  | 功能                     | 主要函数                                                 |
| --- | ---------------------- | ---------------------------------------------------- |
| 1 | 解析参数、加载配置文件            | argparse + yaml                                      |
| 2 | 创建数据加载器（Helper 类）      | `ImageHelper` / `LoanHelper`                         |
| 3 | 创建全局模型与本地模型            | `helper.create_model()`                              |
| 4 | 启动联邦学习训练循环             | `for epoch in range(...)`                            |
| 5 | 每轮：选择客户端 → 本地训练 → 聚合模型 | `train.train()` + `helper.average_shrink_models()` 等 |
| 6 | 评估模型：正常准确率 + 后门攻击成功率   | `test.Mytest()` + `test.Mytest_poison()`             |
| 7 | 触发器分布式可视化与记录           | Visdom + csv_record                                  |
| 8 | 保存模型和结果                | `helper.save_model()`                                |
main.py
│
├── 参数加载 (YAML)
├── 数据与模型初始化 (Helper)
│
├── for each global round:
│     ├── 选择客户端 (部分为攻击者)
│     ├── 本地训练 (train.train)
│     ├── 模型聚合 (FedAvg/GeoMed/FoolsGold)
│     ├── 测试模型 (clean & poisoned)
│     ├── 可视化/记录结果
│     └── 保存模型
│
└── 保存可视化结果 (Visdom)
1.如何处理的数据？每个客户端分配数据，每次选十个客户端的数据进行训练，然后聚合到全局模型
2.触发器如何植入投毒？投毒比例如何？
3.如何训练和测试到ACC和ASR
"""

import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import train
import test
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms
from loan_helper import LoanHelper
from image_helper import ImageHelper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config
import copy

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")

# 在程序开头定义一个变量来控制是否启用 Visdom
ENABLE_VISDOM = True

if ENABLE_VISDOM:
    import visdom
    vis = visdom.Visdom(port=8098)
else:
    vis = None  # 如果不启用 Visdom，设置为 None
#启动 Visdom 服务器以便后续可视化（训练曲线、聚合权重、触发器可视化）。
# vis = visdom.Visdom(port=8098, use_incoming_socket=False)
criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

#测试第 index 个触发器（即第 i 个攻击客户端的局部 trigger）对当前全局模型的攻击效果。helper: 封装联邦学习实验的辅助类（包含模型、参数、数据、记录对象）。
def trigger_test_byindex(helper, index, vis, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_trigger(helper=helper, model=helper.target_model, #返回：分别是当前触发器下的测试损失、准确率、正确预测数、总样本数
                                   adver_trigger_index=index)       #调用 test.Mytest_poison_trigger()：在带触发器的测试样本上评估模型性能（输出 ASR、loss、acc 等）。
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total]) #把这一轮的触发器测试结果写入全局 CSV 记录中。
    if vis and helper.params['vis_trigger_split_test']:
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_index_" + str(index) + "_trigger")

#按客户端名测试触发器效果，与上一个函数类似，但这里通过 客户端名字（agent_name_key） 来测试触发器效果，而不是索引。
def trigger_test_byname(helper, agent_name_key, vis, epoch):        #agent_name_key 是字符串（如 "attacker_1", "attacker_2"）；
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        test.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)   #Mytest_poison_agent_trigger() 按客户端名测试。
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])
    if vis and helper.params['vis_trigger_split_test']:     #可视化这位攻击客户端的 trigger 在每个 epoch 的表现。
        helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                   eid=helper.params['environment_name'],
                                                   name="global_in_" + str(agent_name_key) + "_trigger")
#在联邦聚合（FedAvg）阶段，对每个客户端的聚合权重分布进行可视化。
#每一轮联邦更新时，不同客户端的权重参数可能因数据差异或攻击导致变化，该函数可以观察到：哪些客户端的权重被大幅调整；攻击客户端的权重是否异常。
# names: 客户端名称列表，例如 ['client_0', 'client_1', 'attacker_2']，weights: 每个客户端的聚合权重（数值），epoch: 当前训练轮次，vis: Visdom 实例，adversarial_name_keys: 攻击者客户端名列表
def vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):   #遍历所有客户端；
        _name= names[i]
        _weight=weights[i]
        _is_poison=False
        if _name in adversarial_name_keys:  #判断该客户端是否为攻击者；
            _is_poison=True
        helper.target_model.weight_vis(vis=vis,epoch=epoch,weight=_weight, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)   #调用模型内部可视化方法：

#与上一个类似，但这里可视化的是 客户端的 α 系数（可能是学习率缩放或聚合权重因子）。其中的 α_i 表示客户端 i 的贡献度权重。
# 观察攻击者的 α 是否与正常客户端差异显著；检查是否出现了聚合偏置；用于检测或验证 DBA 对聚合过程的影响
def vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys):
    print(names)
    print(adversarial_name_keys)
    for i in range(0,len(names)):
        _name= names[i]
        _alpha=alphas[i]
        _is_poison=False
        if _name in adversarial_name_keys:
            _is_poison=True
        helper.target_model.alpha_vis(vis=vis,epoch=epoch,alpha=_alpha, eid=helper.params['environment_name'],
                                       name=_name,is_poisoned=_is_poison)

if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    #从命令行读取 YAML 配置文件路径；解析参数（包括：数据集类型、模型、客户端数量、攻击者比例等）。
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r', encoding='utf-8') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    #根据数据类型创建 Helper 实例
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_GTSRB:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'gtsrb'))
        helper.load_data()
    else:
        helper = None
    #Helper 的职责：加载数据并分发给各客户端；初始化模型；管理聚合方式；管理攻击客户端信息
    logger.info(f'load data done')
    helper.create_model()       #创建全局模型；
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    best_loss = float('inf')
    # 在 Visdom 界面显示当前实验参数
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)        #初始化参数聚合存储；

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    submit_update_dict = None
    num_no_progress = 0

    #联邦训练主循环,每次循环 = 一次 联邦通信轮（global round）
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()
        #接下来会随机或固定地选择一些客户端参与训练：
        agent_name_keys = helper.participants_list  #agent_name_keys：被选中的客户端；
        adversarial_name_keys = []                  #adversarial_name_keys：其中哪些是攻击者（带有局部触发器）。
        if helper.params['is_random_namelist']:
            if helper.params['is_random_adversary']:  # random choose , maybe don't have advasarial
                agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
                for _name_keys in agent_name_keys:
                    if _name_keys in helper.params['adversary_list']:
                        adversarial_name_keys.append(_name_keys)
            else:  # must have advasarial if this epoch is in their poison epoch
                ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
                for idx in range(0, len(helper.params['adversary_list'])):
                    for ongoing_epoch in ongoing_epochs:
                        if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                            if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                                adversarial_name_keys.append(helper.params['adversary_list'][idx])      #将要在这轮进行投毒恶意客户端加入到这一轮的训练列表中

                nonattacker=[]
                for adv in helper.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        nonattacker.append(copy.deepcopy(adv))
                benign_num = helper.params['no_models'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(helper.benign_namelist+nonattacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
        else:
            if helper.params['is_random_adversary']==False:
                adversarial_name_keys=copy.deepcopy(helper.params['adversary_list'])
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        # 各客户端本地训练：每个客户端用自己的本地数据训练一个模型；若客户端在 adversarial_name_keys 中 → 在训练中注入后门 trigger；训练完成后，将每个客户端的模型参数返回（epochs_submit_update_dict）
        epochs_submit_update_dict, num_samples_dict = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        logger.info(f'time spent on training: {time.time() - t}')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        is_updated = True
        #参数聚合：联邦聚合算法核心
        if helper.params['aggregation_methods'] == config.AGGR_MEAN:        #普通 FedAvg 聚合：经典基线
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])
            num_oracle_calls = 1
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:   #几何中值防御：对抗鲁棒聚合
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)
            vis_agg_weight(helper, names, weights, epoch, vis, adversarial_name_keys)
            vis_fg_alpha(helper, names, alphas, epoch, vis, adversarial_name_keys)

        elif helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:     #FoolsGold 防御：针对共谋攻击的防御，
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)
            vis_agg_weight(helper,names,weights,epoch,vis,adversarial_name_keys)    #并在可视化中也展示这些聚合权重变化。
            vis_fg_alpha(helper,names,alphas,epoch,vis,adversarial_name_keys )
            num_oracle_calls = 1

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1
        #模型评估（正常 + 后门）：正常测试 在干净测试集上评估模型准确率（Clean Accuracy）。

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if helper.params['is_poison']:
            #后门攻击测试：在含触发器输入的测试集上评估后门攻击成功率（Attack Success Rate）。
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

            """
            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            if helper.params['vis_trigger_split_test']:
                helper.target_model.trigger_agent_test_vis(vis=vis, epoch=epoch, acc=epoch_acc_p, loss=None,
                                                           eid=helper.params['environment_name'],
                                                           name="global_combine")
                #验证分布式后门组合效果。
            if len(helper.params['adversary_list']) == 1:  # centralized attack
                if helper.params['centralized_test_trigger'] == True:  # centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, vis, epoch)
            else:  # distributed attack
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, vis, epoch)
            """
        #保存当前模型权重、训练结果、测试结果，方便后续分析。
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)



    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    # 将所有 Visdom 图表保存，以便后期分析。
    vis.save([helper.params['environment_name']])
