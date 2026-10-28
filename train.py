#train.py
import datetime
import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import loan_train
import image_train
import config
import random


def train(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    """
    训练函数，根据任务类型调用不同的训练模块，执行恶意攻击或正常训练。

    输入:
    - helper: 提供训练过程所需的辅助信息，如数据、模型等。
    - start_epoch: 开始训练的epoch编号。
    - local_model: 当前客户端的本地模型。
    - target_model: 目标模型，通常是全局模型。
    - is_poison: 是否进行恶意攻击（即是否使用毒化数据进行训练）。
    - agent_name_keys: 参与训练的代理（客户端）名称列表。

    输出:
    - epochs_submit_update_dict: 存储每个客户端的训练结果（模型更新）。
    - num_samples_dict: 每个客户端使用的数据样本数。
    """
    epochs_submit_update_dict = {}  # 用于存储每个客户端的模型更新结果
    num_samples_dict = {}  # 用于存储每个客户端的样本数

    # 根据任务类型执行不同的训练流程
    if helper.params['type'] == config.TYPE_LOAN:
        # 如果是贷款数据任务，调用贷款任务的训练函数
        epochs_submit_update_dict, num_samples_dict = loan_train.LoanTrain(helper, start_epoch, local_model,
                                                                           target_model, is_poison, agent_name_keys)

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_GTSRB \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        # 如果是图像分类任务（如CIFAR, MNIST, 或TinyImageNet），调用图像训练函数
        epochs_submit_update_dict, num_samples_dict = image_train.ImageTrain(helper, start_epoch, local_model,
                                                                             target_model, is_poison, agent_name_keys)

    # 返回训练结果：模型更新和样本数
    return epochs_submit_update_dict, num_samples_dict
