#fed_ga_search_as.py
import torch
import numpy as np
import copy
import random
import json
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
# 从项目核心模块导入
import config
from image_helper import ImageHelper
from util import (
    DA_REGISTRY,
    apply_da_rgb,
    combo_to_json,
    random_combo_no_repeat,
    ssim_batch, apply_da_single_channel_safe
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- IQA 相关导入 ---
try:
    import pyiqa
    GMSD_METRIC = pyiqa.create_metric('gmsd').to("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    GMSD_METRIC = None

try:
    import lpips
    LPIPS_MODEL = lpips.LPIPS(net='vgg').to("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    LPIPS_MODEL = None

from skimage.metrics import structural_similarity as sk_ssim


# ---------------------------
# 1. 遗传算法演化算子
# ---------------------------

def mutate_params(params):
    """参数微调：对增强方法的参数字典进行随机扰动"""
    new_params = copy.deepcopy(params)
    for k, v in new_params.items():
        if isinstance(v, float):
            new_params[k] = v * random.uniform(0.8, 1.2)
        elif isinstance(v, int) and not isinstance(v, bool):
            new_params[k] = max(1, v + random.choice([-1, 1]))
    return new_params


def uniform_crossover(ind1, ind2):
    """均匀交叉"""
    c1, c2 = list(ind1), list(ind2)
    for i in range(3):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return tuple(c1), tuple(c2)


def mutate_smart(ind, da_keys, pm=0.2, p_fine_tune=0.4):
    """智能变异：结合方法更换与参数微调"""
    genes = list(ind)
    for i in range(len(genes)):
        if random.random() < pm:
            if random.random() < p_fine_tune:
                method_name, params = genes[i]
                genes[i] = (method_name, mutate_params(params))
            else:
                new_k = random.choice(da_keys)
                genes[i] = (new_k, dict(DA_REGISTRY[new_k][1]))

    # 唯一性检查
    used_methods = set()
    final_genes = []
    for name, p in genes:
        if name in used_methods:
            new_k = random.choice([k for k in da_keys if k not in used_methods])
            final_genes.append((new_k, dict(DA_REGISTRY[new_k][1])))
        else:
            final_genes.append((name, p))
        used_methods.add(final_genes[-1][0])
    return tuple(final_genes)


def tournament_selection(pop, fitnesses, k=3):
    """竞赛选择"""
    candidates_idx = random.sample(range(len(pop)), k)
    best_idx = max(candidates_idx, key=lambda i: fitnesses[i])
    return copy.deepcopy(pop[best_idx])

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


# ---------- Image Quality Helpers ----------

def compute_psnr(clean, adv, eps=1e-8):
    c = clean.astype(np.float32) / 255.0
    a = adv.astype(np.float32) / 255.0
    mse = np.mean((c - a) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + eps)))

def compute_gmsd(clean, adv, device="cuda"):
    if GMSD_METRIC is not None:
        c_t = torch.from_numpy(clean.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        a_t = torch.from_numpy(adv.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        return GMSD_METRIC(c_t, a_t).item()
    return 0.5  # Fallback

def compute_lpips(clean, adv, device="cuda"):
    if LPIPS_MODEL is not None:
        def to_tensor(x):
            return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 127.5 - 1.0

        with torch.no_grad():
            return LPIPS_MODEL(to_tensor(clean), to_tensor(adv)).item()
    return 0.0

# ---------------------------
# 2. 核心搜索类 (增加本地轮次逻辑)
# ---------------------------
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import random
import json
from pathlib import Path
from tqdm import tqdm
import lpips

# 假设这些工具函数已在外部定义
# from utils import DA_REGISTRY, apply_da_single_channel_safe, apply_da_rgb, ...

class FederatedGASearcher:
    def __init__(self, helper):
        self.helper = helper
        self.da_keys = list(DA_REGISTRY.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 获取配置参数
        self.pop_size = helper.params.get('ga_pop_size', 20)
        self.n_gen = helper.params.get('ga_gen', 15)
        self.pm = helper.params.get('ga_pm', 0.2)
        self.local_epochs = helper.params.get('ga_local_epochs', 3)
        self.fl_rounds = helper.params.get('ga_fl_rounds', 10)

        # 权重分配 (推荐 alpha 设高以提升 ASR)
        self.alpha = helper.params.get('ga_alpha', 10.0)
        self.beta = helper.params.get('ga_beta', 0.1)
        self.gamma = helper.params.get('ga_gamma', 0.5)

        # IQA 子权重与参考值
        self.w_psnr = helper.params.get('w_psnr', 0.25)
        self.w_ssim = helper.params.get('w_ssim', 0.25)
        self.w_gmsd = helper.params.get('w_gmsd', 0.25)
        self.w_lpips = helper.params.get('w_lpips', 0.25)
        self.psnr_min, self.psnr_max = 10.0, 40.0
        self.gmsd_max = 0.5

        # --- 单次初始化 IQA 模型，避免显存堆积 ---
        # print("正在初始化 LPIPS 指标模型...")
        # self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        # self.lpips_vgg.eval()

    def evaluate_combo_fitness(self, individual):
        """
        模拟 Attack A-M 并加入显存管理机制
        修复：解决 local_model 引用错误，并确保返回格式一致
        """
        global_model = None
        # 将 local_model 初始化移入循环内或确保其初始状态可安全检测
        current_local_model = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1. 初始化全局模型
            global_model = copy.deepcopy(self.helper.target_model).to(self.device)

            # 2. 数据分片逻辑 (10个客户端)
            all_train_data = list(self.helper.train_loader)
            total_batches = len(all_train_data)
            num_clients = 10
            shard_size = total_batches // num_clients
            client_data_shards = [all_train_data[i * shard_size: (i + 1) * shard_size] for i in range(num_clients)]

            # 3. 超参数
            poison_lr = 0.05
            benign_lr = 0.1
            internal_poison_epochs = 4
            eta = 0.1
            no_models = 10
            num_malicious = 3

            # --- 联邦训练逻辑 ---
            for fl_round in range(self.fl_rounds):
                weight_accumulator = {name: torch.zeros_like(param).float()
                                      for name, param in global_model.state_dict().items()}

                for client_idx in range(num_clients):
                    # 安全释放显存
                    if current_local_model is not None:
                        del current_local_model
                        current_local_model = None

                    current_local_model = copy.deepcopy(global_model).to(self.device)
                    current_local_model.train()

                    is_poison = (client_idx < num_malicious)
                    current_lr = poison_lr if is_poison else benign_lr
                    current_epochs = internal_poison_epochs if is_poison else self.local_epochs

                    optimizer = torch.optim.SGD(current_local_model.parameters(), lr=current_lr, momentum=0.9)

                    for epoch in range(current_epochs):
                        for batch_id, (data, target) in enumerate(client_data_shards[client_idx]):
                            if batch_id > 15: break

                            data_np = data.cpu().numpy().transpose(0, 2, 3, 1)
                            new_targets = target.clone().to(self.device)

                            if is_poison:
                                num_poison = 5
                                # 每个恶意节点使用 individual 中对应的 [method, params]
                                method, params = individual[client_idx]
                                for idx in range(min(len(data), num_poison)):
                                    img_u8 = (np.clip(data_np[idx], 0, 1) * 255).astype(np.uint8)
                                    # 针对不同恶意节点处理不同通道
                                    res_ch = apply_da_single_channel_safe(img_u8[:, :, client_idx], method, params)
                                    img_u8[:, :, client_idx] = res_ch
                                    data_np[idx] = img_u8.astype(np.float32) / 255.0
                                    new_targets[idx] = self.helper.params['target_label']

                            inputs = torch.from_numpy(data_np.transpose(0, 3, 1, 2)).to(self.device)
                            optimizer.zero_grad()
                            loss = F.cross_entropy(current_local_model(inputs), new_targets)
                            loss.backward()
                            optimizer.step()

                    # 累加 Delta
                    with torch.no_grad():
                        for name, param in current_local_model.state_dict().items():
                            diff = param.float() - global_model.state_dict()[name].float()
                            weight_accumulator[name].add_(diff)

                # 全局聚合
                with torch.no_grad():
                    curr_state = global_model.state_dict()
                    for name in curr_state:
                        update = weight_accumulator[name] * (eta / no_models)
                        curr_state[name].add_(update.to(curr_state[name].dtype))
                    global_model.load_state_dict(curr_state)

            # --- 评估逻辑 ---
            global_model.eval()
            correct_acc, total_acc = 0, 0
            orig_imgs, pois_imgs = [], []

            with torch.no_grad():
                # 1. 评估清洁准确率 (ACC)
                clean_loader = DataLoader(self.helper.test_dataset, batch_size=64, shuffle=False)
                for b_id, (data, target) in enumerate(clean_loader):
                    if b_id > 5: break
                    output = global_model(data.to(self.device))
                    correct_acc += (output.argmax(1) == target.to(self.device)).sum().item()
                    total_acc += len(data)

                    # 2. 准备 IQA 评估样本
                    batch_np = (data.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                    for i in range(len(batch_np)):
                        orig_imgs.append(batch_np[i])
                        # 假设 apply_da_rgb 接受整个 individual 配置进行 RGB 扰动
                        pois_imgs.append(apply_da_rgb(batch_np[i], individual))

                acc = correct_acc / total_acc

                # 3. 评估攻击成功率 (ASR)
                pois_t = torch.from_numpy(np.array(pois_imgs).transpose(0, 3, 1, 2)).float() / 255.0
                correct_asr = 0
                for i in range(0, len(pois_t), 64):
                    batch_t = pois_t[i:i + 64].to(self.device)
                    out = global_model(batch_t)
                    correct_asr += (out.argmax(1) == self.helper.params['target_label']).sum().item()
                asr = correct_asr / len(pois_t)

                # 4. 计算 IQA 惩罚
            #     lpips_vals = []
            #     # 抽取 50 张样本计算 IQA 以节省 GPU
            #     sample_indices = min(50, len(orig_imgs))
            #     c_t = torch.from_numpy(np.array(orig_imgs[:sample_indices]).transpose(0, 3, 1, 2)).float().to(
            #         self.device) / 127.5 - 1.0
            #     p_t = torch.from_numpy(np.array(pois_imgs[:sample_indices]).transpose(0, 3, 1, 2)).float().to(
            #         self.device) / 127.5 - 1.0
            #
            #     # 使用预加载的 self.lpips_vgg
            #     lp_dist = self.lpips_vgg(c_t, p_t).squeeze().mean().item()
            #
            #     # 计算 PSNR/SSIM/GMSD (建议在 CPU/Numpy 完成以减小显存)
            #     psnrs = [compute_psnr(orig_imgs[i], pois_imgs[i]) for i in range(sample_indices)]
            #     ssims = [sk_ssim(orig_imgs[i], pois_imgs[i], channel_axis=2, data_range=255) for i in range(sample_indices)]
            #     gmsds = [compute_gmsd(orig_imgs[i], pois_imgs[i], self.device) for i in range(sample_indices)]
            #     mean_psnr, mean_ssim, mean_gmsd = np.mean(psnrs), np.mean(ssims), np.mean(gmsds)
            #
            # # 归一化 Penalty
            # psnr_norm = np.clip((mean_psnr - self.psnr_min) / (self.psnr_max - self.psnr_min), 0, 1)
            # gmsd_norm = np.clip(mean_gmsd / self.gmsd_max, 0, 1)
            # iq_penalty = (self.w_psnr * (1 - psnr_norm) + self.w_ssim * (1 - mean_ssim) +self.w_gmsd * gmsd_norm + self.w_lpips * lp_dist)
            #
            # # 最终 Fitness 计算
            # fitness = (self.alpha * asr) + (self.beta * acc) - (self.gamma * iq_penalty)
            #
            # # ASR 突破奖励（可选）：如果 ASR 极低，给予微小惩罚或不奖励
            # if asr < 0.05: fitness -= 0.5

            fitness = (self.alpha * asr) + (self.beta * acc)

            record = {
                "asr": asr,
                "acc": acc,
                # "avg_ssim": mean_ssim,
                # "psnr": mean_psnr,
                # "gmsd": mean_gmsd,
                # "lpips": lp_dist,
                # "iq_penalty": iq_penalty
            }
            return fitness, record

        except Exception as e:
            print(f"\n[Error during evaluation]: {e}")
            return -1.0, {}

        # 彻底清理显存
        finally:
            if global_model is not None: del global_model
            if current_local_model is not None: del current_local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def search(self):
        print(f"开始进化搜索... [本地轮次: {self.local_epochs}, 联邦轮数: {self.fl_rounds}]")
        population = [random_combo_no_repeat(length=3) for _ in range(self.pop_size)]
        best_individual, best_fitness = None, -float('inf')

        for gen in range(self.n_gen):
            print(f"\n[Generation {gen + 1}/{self.n_gen}]")
            results = []

            for ind in tqdm(population, desc="Evaluating"):
                fit, rec = self.evaluate_combo_fitness(ind)
                results.append((fit, ind, rec))

            results.sort(key=lambda x: x[0], reverse=True)
            best_fit_gen, best_ind_gen, best_rec_gen = results[0]

            if best_fit_gen > best_fitness:
                best_fitness = best_fit_gen
                best_individual = copy.deepcopy(best_ind_gen)

            print(f"\n" + "=" * 50)
            print(f"GEN {gen + 1} BEST SUMMARY:")
            print(f"  Fitness: {best_fit_gen:.4f} (Global: {best_fitness:.4f})")
            print(f"  ASR: {best_rec_gen['asr']:.4f} | ACC: {best_rec_gen['acc']:.4f}")
            # print(f"  IQA -> SSIM: {best_rec_gen['avg_ssim']:.4f} | PSNR: {best_rec_gen['psnr']:.2f} | GMSD: {best_rec_gen['gmsd']:.4f} | LPIPS: {best_rec_gen['lpips']:.4f}")
            # print(f"  Penalty: {best_rec_gen['iq_penalty']:.4f}")
            print(f"  Combo: {[g[0] for g in best_ind_gen]}")
            print("=" * 50)

            # 进化算子 (保持原样)
            new_pop = [copy.deepcopy(results[0][1]), copy.deepcopy(results[1][1])]
            raw_pop = [r[1] for r in results]
            fits = [r[0] for r in results]

            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(raw_pop, fits)
                p2 = tournament_selection(raw_pop, fits)
                child1, child2 = uniform_crossover(p1, p2) if random.random() < 0.8 else (p1, p2)
                new_pop.append(mutate_no_repeat(child1, self.da_keys, pm=self.pm))
                if len(new_pop) < self.pop_size:
                    new_pop.append(mutate_no_repeat(child2, self.da_keys, pm=self.pm))

            population = new_pop

            # 每代结束后强制清理 Windows 显存碎片
            gc.collect()
            torch.cuda.empty_cache()

        # 保存逻辑... (略)
        # 保存结果
        out_path = Path(self.helper.params.get('out_dir', './ga_results/cifar_AM_GA/Demo26_01_07_gen15_pop25_fl45_AM_IQA_alpha10_beta0_1'))
        out_path.mkdir(parents=True, exist_ok=True)

        # 获取转换后的结果
        json_content = combo_to_json(best_individual)
        # 核心修改：如果 combo_to_json 返回的是字符串，直接写入
        if isinstance(json_content, str):
            (out_path / "best_combo.json").write_text(json_content, encoding="utf-8")
        else:
            # 如果是列表或字典，则序列化写入
            with open(out_path / "best_combo.json", "w", encoding="utf-8") as f:
                json.dump(json_content, f, indent=4, ensure_ascii=False)
        print(f"\n搜索完成！最优基因序列: {[g[0] for g in best_individual]}")


# ---------------------------
# 3. 运行入口
# ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DA-GA-DBA Search')
    parser.add_argument('--params', default='utils/cifar_params.yaml', help='配置文件路径')
    # 增加命令行直接修改参数的能力
    parser.add_argument('--ga_pop_size', type=int, default=25)
    parser.add_argument('--ga_gen', type=int, default=15)
    parser.add_argument('--ga_local_epochs', type=int, default=2, help='恶意客户端本地训练轮数')
    parser.add_argument('--ga_fl_rounds', type=int, default=45, help='评估时的联邦聚合轮数')
    parser.add_argument('--target_label', type=int, default=2)
    # 如果你想在命令行指定 name，可以取消下面这行的注释
    # parser.add_argument('--name', type=str, help='实验名称')

    args = parser.parse_args()

    # 1. 加载 YAML 配置
    with open(args.params, 'r', encoding='utf-8') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)

    # 2. 将命令行参数合并到配置字典中
    params_loaded.update(vars(args))

    # 3. 获取当前时间
    current_time = datetime.now().strftime('%m%d_%H%M')

    # 4. 参考 main.py 的写法：使用 .get() 避免 KeyError
    # 如果 params_loaded 里有 'name' 就用它的值，否则根据数据集类型给个默认名
    if params_loaded['type'] == 'cifar':
        default_name = 'cifar_ga'
    elif params_loaded['type'] == 'mnist':
        default_name = 'mnist_ga'
    else:
        default_name = 'ga_search_exp'

    exp_name = params_loaded.get('name', default_name)

    # 5. 初始化 ImageHelper
    helper = ImageHelper(
        current_time=current_time,
        params=params_loaded,
        name=exp_name
    )

    # 6. 加载数据和模型
    helper.load_data()
    helper.create_model()

    # 7. 构造用于 GA 快速评估的 DataLoader
    # helper.train_data 是 [(0, loader0), (1, loader1), ...]
    # 我们直接取第一个参与者的 DataLoader 即可
    if isinstance(helper.train_data, list) and len(helper.train_data) > 0:
        # 取出第一个元组 (participant_id, dataloader) 中的 dataloader
        helper.train_loader = helper.train_data[0][1]
    else:
        # 兜底方案：如果不是列表，则尝试直接赋值或包装
        helper.train_loader = helper.train_data

    # 同样的，检查 test_loader
    # 测试集：使用 helper.test_dataset 重新包装成一个 DataLoader
    # 这样才能保证它是清洁的数据（没有中毒），且能按 Batch 读取
    helper.test_loader = torch.utils.data.DataLoader(
        helper.test_dataset,
        batch_size=64,
        shuffle=False
    )
    # 8. 执行搜索
    searcher = FederatedGASearcher(helper)
    searcher.search()




