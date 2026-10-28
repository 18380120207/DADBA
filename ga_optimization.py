#ga_optimization.py
import random

import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pathlib import Path

# 从 DA-GA-DBA 项目中导入需要的功能
from utils.utils import UniversalDataset
from models.resnet_cifar import get_model  # 从模型定义中获取 ResNet 结构
from helper import client_update, fedavg_aggregate  # DA-GA-DBA 中的客户端更新与联邦聚合方法
from util import apply_da_rgb, list_da_methods, DA_REGISTRY, random_combo_no_repeat, combo_to_json  # 从 util 中导入数据增强操作


# ---------------------------
# GA Methods (Mutation, Crossover, Selection)
# ---------------------------

def enforce_unique_genes(ind, da_keys):
    """确保每个基因在整个组合中唯一"""
    genes = list(ind)
    used = set()
    for i, (name, params) in enumerate(genes):
        if name is None or name == 'noop': continue
        if name in used:
            choices = [k for k in da_keys if k not in used]
            if not choices:
                choices = da_keys[:]
            new_k = random.choice(choices)
            genes[i] = (new_k, dict(DA_REGISTRY[new_k][1]))
        used.add(genes[i][0])
    return tuple(genes)


def mutate_no_repeat(ind, da_keys, pm=0.1):
    """执行变异操作，确保不会重复"""
    genes = list(ind)
    for i in range(len(genes)):
        if random.random() < pm:
            existing = {g[0] for idx, g in enumerate(genes) if idx != i and g[0] not in (None, 'noop')}
            choices = [k for k in da_keys if k not in existing]
            if not choices:
                choices = da_keys[:]
            new_k = random.choice(choices)
            genes[i] = (new_k, dict(DA_REGISTRY[new_k][1]))
    return enforce_unique_genes(tuple(genes), da_keys)


def single_point_crossover(a, b, da_keys=None):
    """单点交叉操作"""
    point = random.choice([1, 2])
    c1 = a[:point] + b[point:]
    c2 = b[:point] + a[point:]
    if da_keys is not None:
        c1 = enforce_unique_genes(c1, da_keys)
        c2 = enforce_unique_genes(c2, da_keys)
    return (c1, c2)


def tournament_selection(pop, fitnesses, k=3):
    """锦标赛选择操作"""
    ids = random.sample(range(len(pop)), k)
    best = max(ids, key=lambda i: fitnesses[i])
    return deepcopy(pop[best])


# ---------------------------
# Federated Learning Simulation (Fitness Evaluation)
# ---------------------------

def evaluate_combo_fitness(combo, root, device,
                           train_txt=None, val_txt=None,
                           surrogate_arch="resnet18", num_classes=10,
                           sample_train_k=500, sample_val_k=200,
                           batch_size=64, poison_per_batch=5,
                           surrogate_epochs=3, fl_rounds=1,
                           num_malicious_clients=3,
                           alpha=0.8, beta=0.2, gamma=1.0):
    """
    通过FL模拟和优化评估DA组合的适应度，包含攻击成功率(ASR)和准确率(ACC)。
    不包含IQA评估。
    """
    LOCAL_EPOCHS = surrogate_epochs
    TARGET_LABEL = 0  # 目标标签固定为0（用于后门攻击）

    if num_malicious_clients != 3:
        raise ValueError("Federated GA evaluation requires exactly 3 malicious clients (R, G, B).")

    # 转换：只包含Resize和ToTensor
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # 加载数据集
    train_ds = UniversalDataset(root, txt_path=train_txt, split="train", transform=transform)
    val_ds = UniversalDataset(root, txt_path=val_txt, split="test", transform=transform)

    # 采样训练集子集
    if len(train_ds) > sample_train_k:
        train_ds = Subset(train_ds, random.sample(range(len(train_ds)), sample_train_k))
    if len(val_ds) > sample_val_k:
        val_ds = Subset(val_ds, random.sample(range(len(val_ds)), sample_val_k))

    # 数据集划分给客户端
    client_indices = []
    client_data_size = len(train_ds) // num_malicious_clients
    indices = list(range(len(train_ds)))
    random.shuffle(indices)
    for i in range(num_malicious_clients):
        start = i * client_data_size
        end = (i + 1) * client_data_size if i < num_malicious_clients - 1 else len(train_ds)
        client_indices.append(indices[start:end])

    # 创建客户端数据集
    client_datasets = [Subset(train_ds, indices) for indices in client_indices]

    # 初始化全局模型
    global_model = get_model(surrogate_arch, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 联邦学习轮次
    for fl_round in range(fl_rounds):
        client_weights_list = []
        client_sizes_list = []

        for i in range(num_malicious_clients):
            client_model = get_model(surrogate_arch, num_classes=num_classes).to(device)
            client_model.load_state_dict(global_model.state_dict())

            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

            # 创建客户端数据加载器
            client_loader = DataLoader(client_datasets[i], batch_size=batch_size, shuffle=True)

            # 更新客户端并进行投毒
            weights, size = client_update(client_model, client_loader, criterion, optimizer, device,
                                          LOCAL_EPOCHS, combo, i, poison_per_batch, TARGET_LABEL)
            client_weights_list.append(weights)
            client_sizes_list.append(size)

        # 聚合 (FedAvg)
        global_weights = fedavg_aggregate(global_model, client_weights_list, client_sizes_list)
        global_model.load_state_dict(global_weights)

    # 全局模型评估
    global_model.eval()
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # 计算准确率
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = global_model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    acc = correct / total if total > 0 else 0.0

    # 计算攻击成功率 (ASR)
    origs, pois = [], []
    for img_pil_tensor, _ in val_ds:
        arr = (img_pil_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pois.append(apply_da_rgb(arr, combo))
        origs.append(arr)

    t_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    pois_t = torch.stack([t_transform(Image.fromarray(p)) for p in pois]).to(device)
    with torch.no_grad():
        preds = global_model(pois_t).argmax(1)
        asr = (preds == TARGET_LABEL).sum().item() / len(preds) if len(preds) > 0 else 0.0

    # 最终适应度评估（最大化ASR和ACC）
    fitness = alpha * asr + beta * acc
    record = {
        "asr": asr,
        "acc": acc,
        "fitness_raw": fitness
    }

    return float(fitness), record


# ---------------------------
# GA Main Loop (Search)
# ---------------------------

def ga_search_main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    da_keys = list_da_methods()  # 获取数据增强方法
    pop = [random_combo_no_repeat(random.randint(0, 2 ** 31 - 1), length=3) for _ in range(args.pop_size)]

    best_ind, best_fitness, best_record = None, -1, None
    fitnesses, records = [], []

    # 初始评估
    for i, ind in enumerate(pop):
        fit, rec = evaluate_combo_fitness(ind, args.data_root, device,
                                          train_txt=args.train_txt, val_txt=args.val_txt,
                                          surrogate_arch=args.surrogate_arch, num_classes=args.num_classes,
                                          sample_train_k=args.sample_train_k, sample_val_k=args.sample_val_k,
                                          batch_size=args.batch_size, poison_per_batch=args.poison_per_batch,
                                          surrogate_epochs=args.local_epochs, fl_rounds=args.fl_rounds,
                                          num_malicious_clients=3, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        fitnesses.append(fit)
        records.append(rec)
        if fit > best_fitness:
            best_fitness, best_ind, best_record = fit, deepcopy(ind), rec
        print(f"[Init] ind {i} fitness={fit:.4f} -> {rec}")

    # GA 迭代
    for gen in range(args.generations):
        print(f"=== GA generation {gen + 1}/{args.generations} ===")
        new_pop = []
        while len(new_pop) < args.pop_size:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            if random.random() < args.pc:
                c1, c2 = single_point_crossover(p1, p2, da_keys=da_keys)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            c1 = mutate_no_repeat(c1, da_keys, pm=args.pm)
            c2 = mutate_no_repeat(c2, da_keys, pm=args.pm)
            new_pop += [c1, c2][:args.pop_size - len(new_pop)]

        new_fitnesses, new_records = [], []
        for i, ind in enumerate(new_pop):
            fit, rec = evaluate_combo_fitness(ind, args.data_root, device,
                                              train_txt=args.train_txt, val_txt=args.val_txt,
                                              surrogate_arch=args.surrogate_arch, num_classes=args.num_classes,
                                              sample_train_k=args.sample_train_k, sample_val_k=args.sample_val_k,
                                              batch_size=args.batch_size, poison_per_batch=args.poison_per_batch,
                                              surrogate_epochs=args.local_epochs, fl_rounds=args.fl_rounds,
                                              num_malicious_clients=3, alpha=args.alpha, beta=args.beta,
                                              gamma=args.gamma)
            new_fitnesses.append(fit)
            new_records.append(rec)
            if fit > best_fitness:
                best_fitness, best_ind, best_record = fit, deepcopy(ind), rec
            print(f"[Gen {gen + 1}] ind {i} fitness={fit:.4f} -> {rec}")

        pop, fitnesses, records = new_pop, new_fitnesses, new_records
        print(f"Generation {gen + 1} best fitness: {best_fitness:.4f} -> {best_record}")

    # 保存最优组合
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "best_combo.json").write_text(combo_to_json(best_ind), encoding="utf-8")
    print(f"[DONE] GA finished. Best combo saved at {args.out_dir}")
    print(f"Best fitness={best_fitness:.4f}, record={best_record}")
    return best_ind, best_fitness, best_record
