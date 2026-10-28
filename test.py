#test.py
"""
1.Mytest: 用于测试模型的函数，计算并输出模型在标准数据上的损失、准确率等信息。根据不同任务类型（贷款任务或图像分类任务），使用不同的数据加载器。
2.Mytest_poison: 用于测试带有恶意触发器（中毒数据）的模型。修改了输入数据以添加触发器并进行测试，计算并输出测试损失、准确率以及中毒数据的数量。
3.Mytest_poison_trigger: 该函数专门用于测试特定恶意触发器的效果，支持通过触发器索引来选择特定的触发器。
4.Mytest_poison_agent_trigger: 用于测试来自特定恶意代理（客户端）的中毒数据，支持通过 agent_name_key 来选择特定代理的触发器数据进行测试。
"""
import torch
import torch.nn as nn
import config

import main

def Mytest(helper, epoch,model, is_poison=False, visualize=True, agent_name_key=""):
    """
    用于测试模型的函数，计算并输出测试损失和准确率。
    参数:
    - helper: 提供训练和测试过程所需的辅助信息。
    - epoch: 当前训练的epoch，用于记录日志。
    - model: 需要测试的模型。
    - is_poison: 是否进行恶意攻击的测试。
    - visualize: 是否进行可视化。
    - agent_name_key: 代理（客户端）的标识，用于标识具体客户端。

    返回:
    - total_l: 总测试损失
    - acc: 测试准确率
    - correct: 正确预测的数量
    - dataset_size: 测试数据集的大小
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct = 0
    dataset_size = 0

    # 针对贷款任务进行测试
    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):  # 遍历每个state_helper（客户端）
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()  # 获取测试数据加载器
            for batch_id, batch in enumerate(data_iterator):
                data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)  # 获取批次数据
                dataset_size += len(data)
                output = model(data)  # 使用模型进行预测
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()  # 累加损失
                pred = output.data.max(1)[1]  # 获取最大概率的类别索引
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()  # 累加正确预测的数量

    # 针对图像分类任务进行测试
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_GTSRB:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 计算准确率和平均损失
    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size != 0 else 0
    total_l = total_loss / dataset_size if dataset_size != 0 else 0

    # 记录测试结果
    main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
    if visualize:  # 可视化损失和准确率
        model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
                       eid=helper.params['environment_name'],
                       agent_name_key=str(agent_name_key))
    model.train()  # 恢复训练模式
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch, model, is_poison=False, visualize=True, agent_name_key=""):
    """
    用于测试带有恶意触发器数据的模型，计算并输出损失和准确率。

    参数:
    - helper: 提供训练和测试过程所需的辅助信息。
    - epoch: 当前训练的epoch，用于记录日志。
    - model: 需要测试的模型。
    - is_poison: 是否进行恶意攻击的测试。
    - visualize: 是否进行可视化。
    - agent_name_key: 代理（客户端）的标识，用于标识具体客户端。

    返回:
    - total_l: 总测试损失
    - acc: 测试准确率
    - correct: 正确预测的数量
    - poison_data_count: 中毒数据的数量
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    # 针对贷款任务进行测试
    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        # 收集所有中毒触发器的名称和值
        for j in range(0, helper.params['trigger_num']):
            for name in helper.params[str(j) + '_poison_trigger_names']:
                trigger_names.append(name)
            for value in helper.params[str(j) + '_poison_trigger_values']:
                trigger_values.append(value)

        for i in range(0, len(helper.allStateHelperList)):  # 遍历每个state_helper（客户端）
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                # 对每个batch进行触发器中毒处理
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value  # 添加触发器
                    poison_data_count += 1

                # 获取测试数据并计算损失
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 针对图像分类任务进行测试
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_GTSRB:
        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 计算准确率和平均损失
    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count != 0 else 0
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    # 记录测试结果
    main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))
    if visualize:  # 可视化损失和准确率
        model.poison_test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None, eid=helper.params['environment_name'], agent_name_key=str(agent_name_key))

    model.train()  # 恢复训练模式
    return total_l, acc, correct, poison_data_count

 
def Mytest_poison_trigger(helper, model, adver_trigger_index):
    """
    用于测试特定恶意触发器的数据，计算损失和准确率。

    参数:
    - helper: 提供训练和测试过程所需的辅助信息。
    - model: 需要测试的模型。
    - adver_trigger_index: 恶意触发器的索引。

    返回:
    - total_l: 总测试损失
    - acc: 测试准确率
    - correct: 正确预测的数量
    - poison_data_count: 中毒数据的数量
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    # 针对贷款任务进行测试
    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        if adver_trigger_index == -1:  # 如果是全局触发器
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:  # 如果是特定的触发器
            trigger_names = helper.params[str(adver_trigger_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adver_trigger_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):  # 遍历每个state_helper（客户端）
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                # 对每个batch进行触发器中毒处理
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1

                # 获取测试数据并计算损失
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 针对图像分类任务进行测试
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_GTSRB:
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 计算准确率和平均损失
    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count != 0 else 0
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    model.train()  # 恢复训练模式
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    """
    用于测试特定恶意代理的数据，计算损失和准确率。

    参数:
    - helper: 提供训练和测试过程所需的辅助信息。
    - model: 需要测试的模型。
    - agent_name_key: 代理的名称，用于标识特定客户端。

    返回:
    - total_l: 总测试损失
    - acc: 测试准确率
    - correct: 正确预测的数量
    - poison_data_count: 中毒数据的数量
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    # 针对贷款任务进行测试
    if helper.params['type'] == config.TYPE_LOAN:
        adv_index = -1
        # 查找当前代理在adversary列表中的索引
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if agent_name_key == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        trigger_names = helper.params[str(adv_index) + '_poison_trigger_names']
        trigger_values = helper.params[str(adv_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):  # 遍历每个state_helper（客户端）
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                # 对每个batch进行触发器中毒处理
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1

                # 获取测试数据并计算损失
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 针对图像分类任务进行测试
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_GTSRB:
        data_iterator = helper.test_data_poison
        adv_index = -1
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    # 计算准确率和平均损失
    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count != 0 else 0
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    model.train()  # 恢复训练模式
    return total_l, acc, correct, poison_data_count
