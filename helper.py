#helper.py
from shutil import copyfile

import math
import torch

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record

# 初始化 Helper 类的实例，设置当前时间、模型、数据以及配置等。
# 设定日志记录机制，所有训练过程的日志都会保存到 saved_models/ 文件夹中。
class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/DADBA_gtsrb_Ralpha_AS/model_{self.name}_{current_time}'
        try:
            os.makedirs(self.folder_path, exist_ok=True)
            # os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])     #self.fg: 初始化 FoolsGold 类，用于存储与优化相关的内存（具体细节待后述）。

    # 保存当前模型的检查点，方便之后的恢复。
    # 如果is_best为True，则将模型保存为model_best.pth.tar，表示保存的是性能最好的模型。
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):   #state: 当前模型的状态，包括权重、偏置等。filename: 保存模型的文件名，默认为 checkpoint.pth.tar。
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    #计算模型所有参数的 L2 范数，该范数用于衡量模型的整体大小。
    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    #计算当前模型与目标模型之间的 L2 距离。目标模型可以是全局模型或者某个特定的参考模型。
    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):     #这个函数将模型中的所有参数展平成一个长向量，通常用于后续的计算，如 计算模型之间的距离 或 计算损失函数。
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss
    #计算当前模型与目标模型之间的 余弦相似度。该度量反映了两者在参数空间中的相似性。若相似度较高，说明两者的更新方向相近，若相似度较低，则说明两者差异较大。
    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list) / len(cs_list)).data[0])
        return 1e3 * sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):  #累积计算两个模型或更新之间的 余弦相似度，通常用于比较不同训练轮次之间的相似度变化。last_acc: 上一轮的模型参数。new_acc: 当前轮的模型参数。

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)

    @staticmethod
    def dp_noise(param, sigma): #给参数加 高斯噪声，用于 差分隐私 (Differential Privacy)。
        #后面在 average_shrink_models 和 geometric_median_update 里，如果开启了:self.params['diff_privacy'] == True，就会在每一层的更新 update_per_layer 上加上这段噪声。
        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    #聚合前，把各客户端的更新收集整理，
    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
        weight_accumulator：全局的“累计梯度/更新”的字典，键是参数名，值是累加的梯度。
        epochs_submit_update_dict：每个客户端上传的本地更新结果
        state_keys：参与本轮训练的客户端名称列表，例如 ['client_1', 'attacker_2', ...]
        num_samples_dict：各客户端本轮训练使用的数据量，例如：
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        #如果使用 Foolsgold 聚合方式，不做简单求和，而是把每个客户端的更新（梯度）保留下来，交给后面的 foolsgold_update 自己处理。
        #也就是说Foolsgold 要用所有客户端的梯度来做“相似性惩罚”，所以这里不提前累加。
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates
        #普通 FedAvg / RFA 使用的路径：
        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]  #ocal_model_update_list：这个客户端在一个聚合间隔里的多个本地 epoch 更新列表，比如 [ep1_update, ep2_update, ...]。
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():   #初始化 update[name] = 0，准备累加多轮本地 epoch 的更新。
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])    #全局累计器加上本客户端本 epoch 的更新（用于 全局聚合）。
                        update[name].add_(local_model_update_dict[name])                #update 这个客户端自己的总更新（用于 几何中值算法 等）。
                        detached_data= data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data=detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name]=detached_data # from gpu to cpu   把 GPU tensor 转成 CPU 上的 Python list，方便之后存盘或做其他操作（比如 CSV 记录）。

                updates[state_keys[i]]=(num_samples,update)
            #weight_accumulator：为 FedAvg 或 RFA 提供全局累加的更新。updates：为 RFA（几何中值）、FoolsGold 等高级聚合算法提供各客户端独立的更新。
            return weight_accumulator,updates

    def init_weight_accumulator(self, target_model):
        """
        根据当前全局模型 target_model 的参数结构，建立同样形状的零张量字典，用来累加本轮各客户端上传的更新。
        每轮聚合完成后会重置该结构。
        """
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        标准 FedAvg 更新（带学习率和可选差分隐私）
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            #weight_accumulator[name]：某一层所有参与客户端更新求和后的结果。乘上eta 是全局学习率/no_models 是本轮参与的客户端数。
            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            # 确保 data 和 update_per_layer 的数据类型一致
            data = data.float()  # 强制转换为 float 类型
            update_per_layer = update_per_layer.float()  # 强制转换为 float 类型
            data.add_(update_per_layer) #data.add_(update_per_layer)：对全局模型参数做原位更新（in-place）
        return True

    def foolsgold_update(self,target_model,updates):
        """
        FoolsGold 鲁棒聚合算法
        updates 由之前 accumulate_weight 在 FoolsGold 模式下构造：updates[client_name] = (num_samples, gradients_dict)
        """
        client_grads = []   #长度 = 客户端数量，每个元素是一个“参数名 → 梯度张量”的 dict。
        alphas = []         #每个客户端的样本数。
        names = []          #客户端名字
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)
        #接下来这一段是为了估计整体的毒化比例：
        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)     #统计本轮参与训练的客户端中，攻击者占总样本数的比例 adver_ratio。
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']  #估算每个 batch 中的毒化样本占比 poison_fraction（更多是日志信息）。
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        #然后是真正的 FoolsGold 聚合：
        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        #核心点：根据客户端梯度之间的 余弦相似度，对那些“高度相似”（可能是共谋攻击）的客户端降低权重。
        # 返回agg_grads：FoolsGold加权后的聚合梯度（列表，和模型参数顺序一致）。wv：每个客户端的权重（FoolsGold 给出的可信度）。alpha：可能是额外的调节系数。
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        for i, (name, params) in enumerate(target_model.named_parameters()):    #然后使用 SGD 的方式，把 agg_grads 应用到 target_model 上：
            agg_grads[i]=agg_grads[i] * self.params["eta"]                      #给每个 params.grad 赋值；
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()    #optimizer.step() 更新模型。
        wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)        #最后记录权重 wv 和 alpha 到 CSV，便于分析“哪些客户端被降权”。
        return True, names, wv, alpha

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
        """
        这是 Robust Federated Aggregation（RFA）里经典的 几何中值（geometric median） 聚合方法，使用 Weiszfeld 算法 迭代求解。
        Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        #① 收集点和权重
        points = [] #points：每个客户端的更新（一个 dict，名字→梯度张量）。
        alphas = [] #alphas：每个客户端对应的样本数。
        names = []
        for name, data in updates.items():
            points.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        #同样先估计毒化比例（和 FoolsGold 基本一样）：
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')
        #然后把 alphas 归一化成权重：
        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(config.device)
        #② 初始化几何中值：一开始先用 加权平均 当作初始点；
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging， geometric_median_objective 是几何中值的目标函数：
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv=None
        #③ Weiszfeld 迭代
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
        #4.计算 median 的范数并决定是否更新
        alphas = [Helper.l2dist(median, p) for p in points]     #alphas 此时重新被用为“median 与各客户端的距离”。

        update_norm = 0         #update_norm 是几何中值更新向量的 L2 范数。
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm= math.sqrt(update_norm)
        #如果 max_update_norm 为空或者当前 update_norm 不超过阈值：
        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.params["eta"])      #将几何中值结果 median 作为最终聚合更新；
                if self.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:   #若更新过大则拒绝更新，以防被恶意大更新“炸掉模型”。
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)        #最后记录权重，并返回：

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(),alphas

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays
        计算两个字典 p1 和 p2 对应的 L2 距离。
        p1 和 p2 是包含了模型参数（如权重）的字典，键是参数的名称，值是对应的张量。
        主要用在 几何中值更新 和 加权平均 中来衡量当前模型与目标模型之间的差异。
        """
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """计算 几何中值 聚合的目标函数，具体来说是 加权 L2 距离 之和 ，Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """计算 加权平均，返回一组加权平均后的更新结果。加权是基于每个客户端的样本数 weights 来进行的。
        Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()

        for name, data in points[0].items():
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        """
        保存当前模型的 state_dict（模型的参数）和 训练状态（如 epoch，学习率）。
        如果开启了 save_model，则会将模型保存到指定路径。
        """
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self,epochs_submit_update_dict,global_epochs_submit_dict, epoch,state_keys):
        """
        用来 更新全局的 epoch 提交字典，将每个客户端本轮的更新记录到全局字典中。
        epochs_submit_update_dict：每个客户端在本轮的多次训练（epoch）后的更新，结构类似：
        """

        epoch_len= len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]]= local_model_update_dict

            global_epochs_submit_dict[epoch+j]= per_epoch_dict

        return global_epochs_submit_dict


    #保存全局 epoch 提交字典
    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion, data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        """
        估算 Fisher 信息矩阵，该矩阵用于衡量模型参数对于样本的敏感性。
        通过计算每个参数的 对数似然 函数的梯度来进行估算。
        Fisher 信息矩阵 在深度学习中用于优化、正则化、元学习等场景，衡量不同参数在不同数据上的重要性。
        过程1. 数据遍历：通过 data_loader 迭代数据，计算每个 batch 的 log-likelihood。
        2. 计算梯度：对 log-likelihood 求导，得到每个参数的梯度。
        3. 计算 Fisher 信息：返回每个参数的 梯度平方，这是 Fisher 信息的近似。
        """
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        """
        该方法用于 合并模型参数的均值和 Fisher 信息。
        将模型的每个参数的 均值 和 Fisher 信息 记录到模型中。
        这些值将用于后续的 EWC 损失计算，确保在训练过程中不会遗忘之前学习到的重要知识。
        在 EWC 训练方法中，我们需要保存每个参数的均值和 Fisher 信息，以便后续计算 EWC 损失。该方法是一个 预处理步骤。
        """
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        """
        该方法用于 计算 EWC 损失，它是一种正则化技术，用来防止模型在不断训练新任务时忘记旧任务的知识。
        遍历模型的每个参数，从模型中获取该参数的 均值 和 Fisher 信息。
        计算损失：对于每个参数，计算其当前值与均值之间的差异，再乘以对应的 Fisher 信息，并求和。
        最终返回损失值：计算所有参数的损失之和，并乘以系数 λ / 2。
        EWC 损失用于在 联邦学习 或 增量学习 中，通过正则化来防止模型遗忘先前的任务。
        """
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        """
        聚合客户端的梯度
        FoolsGold 聚合算法 聚合来自多个客户端的梯度时，对每个客户端的贡献进行加权。
        加权是基于每个客户端的梯度相似度来计算的，恶意客户端的影响会被抑制。
        在 联邦学习 中，当数据存在恶意客户端时，使用 FoolsGold 聚合算法可以 降低恶意客户端对模型的负面影响，通过加权更可靠的客户端梯度来更新全局模型。
        """
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        计算梯度相似度并计算权重
        计算客户端梯度的相似度，并基于相似度为每个客户端分配权重（wv）。
        通过 余弦相似度 衡量客户端更新的相似性，类似的客户端更新会得到更高的权重，而与其他客户端的更新差异较大的客户端则会得到较低的权重。
        :param grads:
        :return: compute similatiry and return weightings
        过程：
        1.计算每个客户端之间的 余弦相似度。
        2.根据相似度调整权重，使得最大相似度的客户端权重为 1，其他客户端的权重进行归一化。
        3.使用 logit 函数 来平滑权重值，确保权重不会过大
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha