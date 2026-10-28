#image_helper.py
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import random
import logging
from torchvision import datasets, transforms
import numpy as np


from helper import Helper
from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
from util import apply_da_rgb, apply_da_single_channel_safe


logger = logging.getLogger("logger")
import config
from config import device
import copy
import cv2
import yaml
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import datetime
import json


class ImageHelper(Helper):
    """
    继承自 Helper 类，负责处理图像数据集的加载、处理和模型的构建等任务。
    这个类提供了许多功能，比如训练数据的生成，图像数据的毒化，Dirichlet 分布的数据分配等。
    """

    def create_model(self):
        """
        创建本地模型和目标模型，根据不同的数据集（CIFAR, MNIST, Tiny ImageNet）来选择不同的网络结构。
        如果指定了恢复模型的名称，则加载预训练模型的参数。
        """
        local_model = None
        target_model = None
        if self.params['type'] == config.TYPE_CIFAR:
            # CIFAR数据集使用ResNet18
            local_model = ResNet18(name='Local', created_time=self.params['current_time'])
            target_model = ResNet18(name='Target', created_time=self.params['current_time'])
        elif self.params['type'] == config.TYPE_MNIST:
            # MNIST数据集使用MnistNet
            local_model = MnistNet(name='Local', created_time=self.params['current_time'])
            target_model = MnistNet(name='Target', created_time=self.params['current_time'])
        elif self.params['type'] == config.TYPE_TINYIMAGENET:
            # Tiny ImageNet使用ResNet18
            local_model = resnet18(name='Local', created_time=self.params['current_time'])
            target_model = resnet18(name='Target', created_time=self.params['current_time'])
        elif self.params['type'] == config.TYPE_GTSRB:
            # GTSRB 使用轻量化 ResNet18，类别数为 43
            local_model = ResNet18(name='Local', created_time=self.params['current_time'], num_classes=43)
            target_model = ResNet18(name='Target', created_time=self.params['current_time'], num_classes=43)

        local_model = local_model.to(device)  # 将模型移到指定的设备（CPU/GPU）
        target_model = target_model.to(device)

        if self.params['resumed_model']:
            if torch.cuda.is_available():
                # 如果GPU可用，加载GPU上的模型参数
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                # 否则加载CPU上的模型参数
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}", map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])  # 加载模型参数
            self.start_epoch = loaded_params['epoch'] + 1  # 从保存的模型恢复训练进度
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])  # 获取学习率
            logger.info(
                f"Loaded parameters from saved model: LR is {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1  # 如果没有恢复模型，从第1个epoch开始训练

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        """
        构建训练数据集中每个类别的索引字典。
        返回一个字典，键为类别标签，值为该类别下所有样本的索引。
        """
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # 对CIFAR-10数据集进行遍历
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    # def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
    #     """
    #     使用Dirichlet分布从CIFAR训练集中为每个参与者采样数据。

    #     输入：
    #         - no_participants：参与者的数量
    #         - alpha：Dirichlet分布的参数
    #     输出：
    #         - per_participant_list：每个参与者的数据索引列表
    #     """
    #     cifar_classes = self.classes_dict
    #     class_size = len(cifar_classes[0])  # 每个类别的样本数（CIFAR-10为5000）
    #     per_participant_list = defaultdict(list)  # 用来存储每个参与者的数据索引
    #     no_classes = len(cifar_classes.keys())  # 类别数量（CIFAR-10为10）

    #     image_nums = []
    #     for n in range(no_classes):
    #         image_num = []
    #         random.shuffle(cifar_classes[n])  # 打乱每个类别的样本
    #         sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))  # 从Dirichlet分布中采样
    #         for user in range(no_participants):
    #             no_imgs = int(round(sampled_probabilities[user]))  # 每个参与者分配的图像数量
    #             sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]  # 根据采样数选择数据
    #             image_num.append(len(sampled_list))
    #             per_participant_list[user].extend(sampled_list)
    #             cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]  # 更新类别中的数据
    #         image_nums.append(image_num)
    #     return per_participant_list
    
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
        使用Dirichlet分布从训练集为每个参与者采样数据。

        输入：
            - no_participants：参与者的数量
            - alpha：Dirichlet分布的参数
        输出：
            - per_participant_list：每个参与者的数据索引列表
        """
        cifar_classes = self.classes_dict
        per_participant_list = defaultdict(list)  # 用来存储每个参与者的数据索引
        no_classes = len(cifar_classes.keys())

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])  # 打乱每个类别的样本
            class_size = len(cifar_classes[n])  # 使用当前类别的实际大小，兼容不均衡数据集
            sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        return per_participant_list

    def draw_dirichlet_plot(self, no_classes, no_participants, image_nums, alpha):
        """
        绘制Dirichlet分布下每个类别数据在各个参与者之间的分布情况图。

        输入：
            - no_classes：类别数量
            - no_participants：参与者数量
            - image_nums：每个类别下每个参与者的数据数量
            - alpha：Dirichlet分布的alpha参数
        """
        fig = plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()

        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))

        for k in range(no_participants):
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            left += s[k]

        plt.legend(ncol=20, loc='lower left', bbox_to_anchor=(0, 1), fontsize=4)
        plt.xlabel("Number of Images", fontsize=16)
        fig.tight_layout(pad=0.1)
        fig.savefig(self.folder_path + '/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        """
        获取包含中毒数据的测试数据集，并返回两个测试数据加载器：一个是正常标签的测试集，另一个是包含中毒标签的测试集。
        """
        logger.info('get poison test loader')

        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)

        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(range_no_id)), \
            torch.utils.data.DataLoader(self.test_dataset,
                                        batch_size=self.params['batch_size'],
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(poison_label_inds))

    def load_data(self):
        """
        加载训练数据和测试数据，根据指定的类型（CIFAR、MNIST、Tiny ImageNet）加载不同的数据集，并为每个参与者分配数据。
        """
        logger.info('Loading data')
        dataPath = './data'

        # 加载CIFAR-10数据集
        if self.params['type'] == config.TYPE_CIFAR:
            transform_train = transforms.Compose([transforms.ToTensor()])
            transform_test = transforms.Compose([transforms.ToTensor()])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

        # 加载MNIST数据集
        elif self.params['type'] == config.TYPE_MNIST:
            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                                                transform=transforms.Compose([transforms.ToTensor()]))
            self.test_dataset = datasets.MNIST('./data', train=False,
                                               transform=transforms.Compose([transforms.ToTensor()]))

        # 加载Tiny ImageNet数据集
        elif self.params['type'] == config.TYPE_TINYIMAGENET:
            _data_transforms = {
                'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
                'val': transforms.Compose([transforms.ToTensor()]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'), _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'), _data_transforms['val'])
            logger.info('reading data done')

        elif self.params['type'] == config.TYPE_GTSRB:
            # GTSRB images are resized to 32x32 to reuse the CIFAR-style backbone
            gtsrb_mean = (0.3403, 0.3121, 0.3214)
            gtsrb_std = (0.2724, 0.2608, 0.2669)

            transform_train = transforms.Compose([
            transforms.Resize((36, 36)),
            transforms.RandomCrop((32, 32)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(gtsrb_mean, gtsrb_std),
            ])
            transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(gtsrb_mean, gtsrb_std),
            ])

            self.train_dataset = datasets.GTSRB(dataPath, split='train', download=True, transform=transform_train)
            self.test_dataset = datasets.GTSRB(dataPath, split='test', download=True, transform=transform_test)
            logger.info('reading GTSRB data done')
            
        self.classes_dict = self.build_classes_dict()  # 创建类别索引字典
        logger.info('build_classes_dict done')

        if self.params['sampling_dirichlet']:
            # 如果使用Dirichlet分布进行数据采样
            indices_per_participant = self.sample_dirichlet_train_data(self.params['number_of_total_participants'],
                                                                       alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in indices_per_participant.items()]
        else:
            # 如果不使用Dirichlet分布，数据均匀分配给参与者
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos)) for pos in
                             range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()

        self.adversarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        self.benign_namelist = list(set(self.participants_list) - set(self.adversarial_namelist))

        # ================= DA-GA-DBA =================
        # load best DA combo (global trigger definition)
        if 'best_combo_path' in self.params:
            self.load_best_combo()
            logger.info(f"[DA-GA-DBA] Loaded best_combo from {self.params['best_combo_path']}")

    def get_train(self, indices):
        """
        根据Dirichlet采样的结果，获取训练数据加载器。

        输入：
            - indices：参与者所持有的训练数据的索引列表
        输出：
            - train_loader：训练数据加载器
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                                   pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        均匀分配数据集给每个参与者，获取训练数据加载器。

        输入：
            - all_range：所有数据的索引列表
            - model_no：当前参与者的编号
        输出：
            - train_loader：训练数据加载器
        """
        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(sub_indices))
        return train_loader

    def get_test(self):
        """
        获取测试数据加载器。
        """
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        """
        获取一个训练批次的数据，并将数据和目标标签送入GPU。

        输入：
            - train_data：训练数据
            - bptt：批次数据
            - evaluation：是否为评估模式（默认为False）
        输出：
            - data：处理后的输入数据
            - target：目标标签
        """
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def load_best_combo(self):
        if not hasattr(self, "best_combo"):
            with open(self.params['best_combo_path'], 'r') as f:
                self.best_combo = json.load(f)

    def get_poison_batch(self, bptt, adversarial_index=-1, evaluation=False):
        """
        在训练或评估时，将部分数据进行毒化。

        输入：
            - bptt：批次数据
            - adversarial_index：中毒模式的索引
            - evaluation：是否为评估模式（默认为False）
        输出：
            - new_images：毒化后的图像
            - new_targets：毒化后的标签
            - poison_count：毒化数据的数量
        """
        images, targets = bptt

        poison_count = 0
        # new_images = images
        # new_targets = targets
        new_images = images.clone()
        new_targets = targets.clone()

        use_da_fcba14 = self.params.get('use_da_fcba14', False)
        add_pattern_fn = self.add_da_pattern_fcba14 if use_da_fcba14 else self.add_da_pattern
        if (not use_da_fcba14) and adversarial_index not in [-1, 0, 1, 2]:
            raise ValueError(
                f"adversarial_index {adversarial_index} is out of range for add_da_pattern; "
                "enable use_da_fcba14 or pass -1/0/1/2."
            )

        for index in range(0, len(images)):
            if evaluation:  # 如果是评估模式，所有数据都毒化
                adversarial_index = -1
                new_targets[index] = self.params['poison_label_swap']
                # new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                new_images[index] = add_pattern_fn(images[index], adversarial_index)
                poison_count += 1
            else:  # 如果是训练模式，部分数据毒化
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    # new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                    new_images[index] = add_pattern_fn(images[index], adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    def add_da_pattern(self, ori_image, adversarial_index):
        """
        Apply DA-based trigger (single-channel or global) to a tensor image.
        ori_image: torch.Tensor, shape [3, H, W], range [0,1]
        adversarial_index:
          - 0/1/2: apply DA to R/G/B channel respectively
          - -1: apply global trigger (R+G+B)
        """
        img = ori_image.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # HWC
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        combo = self.best_combo
        assert adversarial_index in [-1, 0, 1, 2], \
            f"Invalid adversarial_index: {adversarial_index}"

        if adversarial_index == -1:
            # global trigger（评估 ASR 用）
            out = apply_da_rgb(img, combo)
        else:
            # 单通道 trigger（DBA-style local trigger）
            out = img.copy()
            ch = adversarial_index  # 0:R,1:G,2:B
            method, params = combo[ch]
            out[:, :, ch] = apply_da_single_channel_safe(img[:, :, ch], method, params)

        out = np.clip(out, 0, 255).astype(np.float32) / 255.0
        out = torch.from_numpy(out).permute(2, 0, 1).type_as(ori_image)
        return out

    def add_da_pattern_fcba14(self, ori_image, adversarial_index):
        """
        使用 DA 组合在 14 个恶意客户端上分布式投毒。
        - adversarial_index: -1 表示全局触发，0-13 按通道循环 [R,G,B,...]
        通道顺序可通过参数 fcba14_channel_cycle 配置，默认使用 [0,1,2] 重复的 14 位序列。
        """
        if adversarial_index == -1:
            return self.add_da_pattern(ori_image, adversarial_index)

        channel_cycle = self.params.get(
            'fcba14_channel_cycle',
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
        )
        if len(channel_cycle) == 0:
            raise ValueError('fcba14_channel_cycle is empty; please provide a valid channel list.')

        ch = channel_cycle[adversarial_index % len(channel_cycle)]
        if ch not in [0, 1, 2]:
            raise ValueError(f'Invalid channel {ch} in fcba14_channel_cycle; expect 0/1/2 for R/G/B.')

        if not hasattr(self, "best_combo"):
            self.load_best_combo()
        combo = self.best_combo

        img = ori_image.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        out = img.copy()
        method, params = combo[ch]
        out[:, :, ch] = apply_da_single_channel_safe(img[:, :, ch], method, params)

        out = np.clip(out, 0, 255).astype(np.float32) / 255.0
        out = torch.from_numpy(out).permute(2, 0, 1).type_as(ori_image)
        return out

    # def add_pixel_pattern(self, ori_image, adversarial_index):
    #     """
    #     在图像中添加中毒模式（例如，触发器）以进行数据中毒。
    #
    #     输入：
    #         - ori_image：原始图像
    #         - adversarial_index：中毒模式的索引
    #     输出：
    #         - image：中毒后的图像
    #     """
    #     image = copy.deepcopy(ori_image)
    #     poison_patterns = []
    #     if adversarial_index == -1:
    #         for i in range(0, self.params['trigger_num']):
    #             poison_patterns = poison_patterns + self.params[str(i) + '_poison_pattern']
    #     else:
    #         poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
    #
    #     if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET:
    #         for i in range(0, len(poison_patterns)):        #什么意思？如何植入触发器的
    #             pos = poison_patterns[i]                    #pos = poison_patterns[0] = 0_poison_pattern = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    #             image[0][pos[0]][pos[1]] = 1                #image[0][pos[0]][pos[1]]
    #             image[1][pos[0]][pos[1]] = 1
    #             image[2][pos[0]][pos[1]] = 1
    #
    #     elif self.params['type'] == config.TYPE_MNIST:
    #         for i in range(0, len(poison_patterns)):
    #             pos = poison_patterns[i]
    #             image[0][pos[0]][pos[1]] = 1
    #
    #     return image


if __name__ == '__main__':
    # 设置随机种子，保证结果可重复
    np.random.seed(1)
    # 加载配置参数
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    # 初始化ImageHelper类
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                         name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars = list(range(100))
    count_all = 0
    # 显示数据在所有参与者中的分布
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count = 0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets = batch
            for t in targets:
                cifar_class_count[t.item()] += 1
            count += len(targets)
        count_all += count
        print(par, cifar_class_count, count, max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all * 1.0 / 100)
