import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime


# 定义一个简单的神经网络基类
class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        # 初始化网络时设置名称和创建时间
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name = name

    # 可视化训练过程中的准确率
    def train_vis(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        # 绘制训练准确率图
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='train_acc_{0}'.format(self.created_time), env=eid,
                 update='append' if vis.win_exists('train_acc_{0}'.format(self.created_time), env=eid) else None,
                 opts=dict(showlegend=True, title='Train Accuracy_{0}'.format(self.created_time),
                           width=700, height=400))
        if loss is not None:
            # 绘制训练损失图
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                     win='train_loss_{0}'.format(self.created_time),
                     update='append' if vis.win_exists('train_loss_{0}'.format(self.created_time), env=eid) else None,
                     opts=dict(showlegend=True, title='Train Loss_{0}'.format(self.created_time), width=700, height=400))
        return

    # 可视化每个批次的损失
    def train_batch_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='train_batch_loss', is_poisoned=False):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        # 绘制每个批次的训练损失图
        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                 env=eid,
                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                 opts=dict(showlegend=True, width=700, height=400, title='Train Batch loss_{0}'.format(self.created_time)))

    # 可视化每个批次与全局模型的距离
    def track_distance_batch_vis(self, vis, epoch, data_len, batch, distance_to_global_model, eid, name=None, is_poisoned=False):
        x = (epoch-1)*data_len + batch + 1

        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        # 绘制每个批次与全局模型的距离图
        vis.line(Y=np.array([distance_to_global_model]), X=np.array([x]),
                 win=f"global_dist_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"global_dist_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Distance to Global {self.created_time}",
                           width=700, height=400))

    # 可视化聚合权重
    def weight_vis(self, vis, epoch, weight, eid, name, is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        # 绘制聚合权重变化图
        vis.line(Y=np.array([weight]), X=np.array([epoch]),
                 win=f"Aggregation_Weight_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"Aggregation_Weight_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Aggregation Weight {self.created_time}",
                           width=700, height=400))

    # 可视化 alpha 参数（例如触发器的影响程度）
    def alpha_vis(self, vis, epoch, alpha, eid, name, is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        # 绘制 alpha 参数变化图
        vis.line(Y=np.array([alpha]), X=np.array([epoch]),
                 win=f"FG_Alpha_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"FG_Alpha_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"FG Alpha {self.created_time}",
                           width=700, height=400))

    # 可视化触发器的测试结果（准确率与损失）
    def trigger_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key, trigger_name, trigger_value):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_triggerweight_vis_acc_{self.created_time}",
                 env=eid,
                 name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                 update='append' if vis.win_exists(f"poison_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                     update='append' if vis.win_exists(f"poison_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Trigger Test Loss_{self.created_time}",
                               width=700, height=400))

    # 可视化触发器的代理测试结果
    def trigger_agent_test_vis(self, vis, epoch, acc, loss, eid, name):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_state_trigger_acc_{self.created_time}",
                 env=eid,
                 name=f'{name}',
                 update='append' if vis.win_exists(f"poison_state_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor State Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_state_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{name}',
                     update='append' if vis.win_exists(f"poison_state_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor State Trigger Test Loss_{self.created_time}",
                               width=700, height=400))

    # 可视化毒化测试结果
    def poison_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        # 绘制毒化测试准确率图
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"poison_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Task Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            # 绘制毒化测试损失图
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_loss_acc_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"poison_loss_acc_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Task Test Loss_{self.created_time}",
                               width=700, height=400))

    # 可视化附加测试结果
    def additional_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        # 绘制附加测试准确率图
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"additional_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"additional_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Additional Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            # 绘制附加测试损失图
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"additional_test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"additional_test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Additional Test Loss_{self.created_time}",
                               width=700, height=400))

    # 可视化主任务测试结果
    def test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        # 绘制主任务测试准确率图
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Main Task Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            # 绘制主任务测试损失图
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Main Task Test Loss_{self.created_time}",
                               width=700, height=400))


    def save_stats(self, epoch, loss, acc):
        # 保存统计信息（epoch、损失、准确率）
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):
        # 将模型的参数拷贝到当前模型中
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())



# 定义用于MNIST数据集的简单神经网络
class SimpleMnist(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(SimpleMnist, self).__init__(name, created_time)
        # 定义两层卷积层和全连接层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        # 前向传播过程，依次进行卷积、池化、全连接等操作
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # 展平
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # 返回log概率
