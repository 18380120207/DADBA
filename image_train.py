#image_train.py
"""
1. 主要训练流程：函数 ImageTrain 是训练和更新模型的主要函数，基于不同的参与者（恶意或非恶意）来执行本地训练。如果是恶意参与者，则进行中毒训练，并定期执行对目标模型的更新。
2. 恶意参与者的处理：当模型是恶意参与者时，训练过程不仅会对分类损失进行优化，还会向模型中注入毒化模式，以实现攻击。每个恶意参与者会在指定的周期内执行毒化训练，并且会有特定的学习率调整策略。
3. 常规训练：对于非恶意参与者，模型会按正常的训练过程进行，不进行任何毒化操作，并且模型参数会根据训练过程逐步更新。
4. 模型更新和梯度聚合：每个epoch结束后，会通过指定的聚合方法（如 AGGR_FOOLSGOLD）来聚合所有客户端的模型更新，并在全局模型中进行更新。
5. 距离计算与可视化：每个批次训练结束后，模型会计算与全局模型的距离，若配置了可视化功能，将结果展示出来。
6. 恶意模型的测试：在恶意训练周期结束后，进行对毒化模型的测试，包括标准测试和毒化测试，以确保恶意行为成功。
"""
import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config


def ImageTrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    """
    训练每个参与者的本地模型，并根据是否为恶意参与者执行中毒训练。

    输入:
    - helper: 训练过程中需要的辅助工具类（如数据加载、模型管理等）
    - start_epoch: 开始训练的起始epoch
    - local_model: 本地模型
    - target_model: 目标模型
    - is_poison: 是否为恶意攻击模式
    - agent_name_keys: 参与者的名称（包括恶意和非恶意参与者）

    输出:
    - epochs_submit_update_dict: 存储每个参与者的模型更新信息
    - num_samples_dict: 每个参与者的数据样本数量
    """
    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries = 0

    # 统计恶意参与者的数量
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries += 1

    # 遍历每个参与者的模型ID
    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()  # 存储上一轮的模型参数
        client_grad = []  # 用于记录每个客户端的梯度

        # 初始化last_local_model，存储目标模型的当前状态
        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        # 同步学习率和模型参数
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        adversarial_index = -1
        # localmodel_poison_epochs = helper.params['poison_epochs']

        # 如果是恶意参与者，根据配置确定其中毒策略
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(0, len(helper.params['adversary_list'])):
                if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                    adversarial_index = temp_index
                    localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    main.logger.info(f'poison local model {agent_name_key} index {adversarial_index}')
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # 如果只有一个恶意参与者，使用全局触发器

        # 遍历每个训练epoch
        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):
            target_params_variables = dict()

            # 保存目标模型的参数状态
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            # 恶意参与者执行中毒训练
            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                main.logger.info('poison_now')

                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr, momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num, 0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) * internal_epoch_num

                # 内部poison epoch循环
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []

                    # 遍历批次数据进行训练
                    for batch_id, batch in enumerate(data_iterator):
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index, evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        # 损失函数：结合分类损失和距离损失
                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        # 获取梯度
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # 获取最大log概率的索引
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        # 计算并记录模型与全局模型的距离
                        if helper.params["batch_track_distance"]:
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len, batch=batch_id,
                                                           distance_to_global_model=distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key), is_poisoned=True)

                    # 调整学习率
                    if step_lr:
                        scheduler.step()
                        main.logger.info(f'Current lr: {scheduler.get_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch, agent_name_key,
                                                                                      internal_epoch,
                                                                                      total_l, correct, dataset_size,
                                                                                     acc, poison_data_count))
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=True,
                                        name=str(agent_name_key))

                    num_samples_dict[agent_name_key] = dataset_size

                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # 内部epoch结束后，进行全局模型和本地模型的距离记录
                main.logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main.logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                if not helper.params['baseline']:
                    main.logger.info(f'will scale.')
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   visualize=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          visualize=False,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main.logger.info(f"Scaling by  {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value = last_local_model[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main.logger.info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))

                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(helper.train_data[agent_name_key][1])
                        model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                       data_len=temp_data_len,
                                                       batch=temp_data_len-1,
                                                       distance_to_global_model=distance,
                                                       eid=helper.params['environment_name'],
                                                       name=str(agent_name_key), is_poisoned=True)

                distance = helper.model_dist_norm(model, target_params_variables)
                main.logger.info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                # 非恶意参与者正常训练
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    for batch_id, batch in enumerate(data_iterator):
                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch, evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # 获取梯度
                        if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # 获取最大log概率的索引
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        # 可视化训练损失
                        if helper.params["vis_train_batch_loss"]:
                            cur_loss = loss.data
                            temp_data_len = len(data_iterator)
                            model.train_batch_vis(vis=main.vis,
                                                  epoch=temp_local_epoch,
                                                  data_len=temp_data_len,
                                                  batch=batch_id,
                                                  loss=cur_loss,
                                                  eid=helper.params['environment_name'],
                                                  name=str(agent_name_key), win='train_batch_loss', is_poisoned=False)

                        if helper.params["batch_track_distance"]:
                            # 计算本地模型与全局模型的距离
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                            batch=batch_id,distance_to_global_model=distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key), is_poisoned=False)

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main.logger.info(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=False,
                                        name=str(agent_name_key))
                    num_samples_dict[agent_name_key] = dataset_size

                    if helper.params["batch_track_distance"]:
                        main.logger.info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # 在内部epoch结束后进行测试
                # epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                #                                                                model=model, is_poison=False, visualize=True,
                #                                                                agent_name_key=agent_name_key)
                # csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            # 在恶意训练后，进行毒化触发器的测试
            # if is_poison:
            #     if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
            #         epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
            #                                                                               epoch=epoch,
            #                                                                               model=model,
            #                                                                               is_poison=True,
            #                                                                               visualize=True,
            #                                                                               agent_name_key=agent_name_key)
            #         csv_record.posiontest_result.append(
            #             [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
            #
            #     # 测试本地触发器
            #     if agent_name_key in helper.params['adversary_list']:
            #         if helper.params['vis_trigger_split_test']:
            #             model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
            #                                          eid=helper.params['environment_name'],
            #                                          name=str(agent_name_key) + "_combine")
            #
            #         epoch_loss, epoch_acc, epoch_corret, epoch_total = \
            #             test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
            #         csv_record.poisontriggertest_result.append(
            #             [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
            #              epoch_acc, epoch_corret, epoch_total])
            #         if helper.params['vis_trigger_split_test']:
            #             model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
            #                                          eid=helper.params['environment_name'],
            #                                          name=str(agent_name_key) + "_trigger")

            # 更新模型权重
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            # 根据不同的聚合方法处理更新
            if helper.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        # 存储当前参与者的模型更新
        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list

    return epochs_submit_update_dict, num_samples_dict
