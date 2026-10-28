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
        self.fl_rounds = helper.params.get('ga_fl_rounds', 5)

        # 权重分配 (推荐 alpha 设高以提升 ASR)
        self.alpha = helper.params.get('ga_alpha', 5.0)
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
        模拟单轮投毒(3恶意+2良性)及后续持久良性衰减(另5个良性)过程
        数据设置：将训练集分为 8 个独立分片 (Shard 0-7)
        """
        global_model = None
        local_model = None

        try:
            # 初始化全局模型
            global_model = copy.deepcopy(self.helper.target_model).to(self.device)

            # --- 1. 数据准备：划分为 8 个完全独立的分片 ---
            all_train_data = list(self.helper.train_loader)
            num_shards = 8
            shard_size = len(all_train_data) // num_shards
            client_data_shards = [all_train_data[i * shard_size: (i + 1) * shard_size] for i in range(num_shards)]

            # 加载超参数
            poison_lr = 0.05
            benign_lr = 0.1
            internal_poison_epochs = 4
            clip_rate = 100.0  # 模型替换攻击缩放比例
            eta = 0.1
            target_label = self.helper.params['target_label']

            total_eval_rounds = self.fl_rounds  # 衰减测试的总轮数
            asr_history = []

            # --- 2. 联邦训练循环 ---
            for fl_round in range(total_eval_rounds):
                weight_accumulator = {name: torch.zeros_like(param).float()
                                      for name, param in global_model.state_dict().items()}

                # 核心逻辑：确定本轮参与的 5 个客户端索引
                if fl_round == 0:
                    # 第 0 轮：3个恶意(0,1,2) + 2个良性(3,4)
                    current_participants = [0, 1, 2, 3, 4]
                else:
                    # 后续轮次：5个全新的良性(3,4,5,6,7) - 其中3,4继续参加，5,6,7是新加入的冲刷力量
                    # 或者你也可以设置为完全不重合的 [3, 4, 5, 6, 7]
                    current_participants = [3, 4, 5, 6, 7]

                for client_idx in current_participants:
                    local_model = copy.deepcopy(global_model).to(self.device)
                    local_model.train()

                    # 判定当前身份
                    is_poison_round = (fl_round == 0)
                    is_malicious = (is_poison_round and client_idx < 3)

                    # 动态调整学习率与 Epoch
                    current_lr = poison_lr if is_malicious else benign_lr
                    current_epochs = internal_poison_epochs if is_malicious else self.local_epochs

                    optimizer = torch.optim.SGD(local_model.parameters(), lr=current_lr, momentum=0.9)

                    for epoch in range(current_epochs):
                        for batch_id, (data, target) in enumerate(client_data_shards[client_idx]):
                            # 控制计算开销，每轮每个客户端训练 10 个 batch
                            if batch_id > 16: break

                            if is_malicious:
                                # --- 恶意客户端：执行投毒 ---
                                data_for_train = data.clone()
                                data_np = data_for_train.cpu().numpy().transpose(0, 2, 3, 1)
                                num_poison = 5

                                for idx in range(min(len(data), num_poison)):
                                    # 应用 GA 搜索出的增强方法 (individual 对应索引 0,1,2)
                                    method, params = individual[client_idx]
                                    img_u8 = (np.clip(data_np[idx], 0, 1) * 255).astype(np.uint8)
                                    res_ch = apply_da_single_channel_safe(img_u8[:, :, client_idx], method, params)
                                    img_u8[:, :, client_idx] = res_ch
                                    data_np[idx] = img_u8.astype(np.float32) / 255.0

                                inputs = torch.from_numpy(data_np.transpose(0, 3, 1, 2)).to(self.device)
                                new_targets = target.clone().to(self.device)
                                for idx in range(min(len(data), num_poison)):
                                    new_targets[idx] = target_label
                            else:
                                # --- 良性客户端：正常训练 ---
                                inputs = data.to(self.device)
                                new_targets = target.to(self.device)

                            optimizer.zero_grad()
                            loss = F.cross_entropy(local_model(inputs), new_targets)
                            loss.backward()
                            optimizer.step()

                    # --- 更新聚合 ---
                    with torch.no_grad():
                        if is_poison_round and is_malicious:
                            # 模拟放缩攻击: W = W_target + (W_local - W_target) * 100
                            for name, param in local_model.named_parameters():
                                target_val = global_model.state_dict()[name].float()
                                scaled_val = target_val + (param.data.float() - target_val) * clip_rate
                                param.data.copy_(scaled_val)

                        # 计算增量并累加
                        for name, param in local_model.state_dict().items():
                            diff = param.float() - global_model.state_dict()[name].float()
                            weight_accumulator[name].add_(diff)

                    del local_model

                # 全局聚合
                with torch.no_grad():
                    curr_state = global_model.state_dict()
                    for name in curr_state:
                        update = weight_accumulator[name] * (eta / 5)
                        curr_state[name].add_(update.to(curr_state[name].dtype))
                    global_model.load_state_dict(curr_state)

                # 每轮结束记录 ASR 来看持久性 (为了加速，可以只在 R1 和 R10 评估)
                if fl_round == 0 or fl_round == (total_eval_rounds - 1):
                    current_asr = self.quick_eval_asr(global_model, individual)
                    asr_history.append(current_asr)

            # --- 最终评估逻辑 ---
            global_model.eval()
            r1_asr = asr_history[0]
            r10_asr = asr_history[-1]

            # 计算衰减率: (初始 - 最终) / 初始
            decay_rate = (r1_asr - r10_asr) / r1_asr if r1_asr > 0 else 0.0

            # 评估清洁准确率 (ACC)
            correct_acc, total_acc = 0, 0
            orig_imgs, pois_imgs = [], []
            with torch.no_grad():
                clean_loader = DataLoader(self.helper.test_dataset, batch_size=64, shuffle=False)
                for b_id, (data, target) in enumerate(clean_loader):
                    if b_id > 5: break
                    output = global_model(data.to(self.device))
                    correct_acc += (output.argmax(1) == target.to(self.device)).sum().item()
                    total_acc += len(data)

                    batch_np = (data.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                    for i in range(len(batch_np)):
                        orig_imgs.append(batch_np[i])
                        pois_imgs.append(apply_da_rgb(batch_np[i], individual))

                acc = correct_acc / total_acc

            #     # 计算 IQA 惩罚
            #     sample_indices = min(50, len(orig_imgs))
            #     c_t = torch.from_numpy(np.array(orig_imgs[:sample_indices]).transpose(0, 3, 1, 2)).float().to(
            #         self.device) / 127.5 - 1.0
            #     p_t = torch.from_numpy(np.array(pois_imgs[:sample_indices]).transpose(0, 3, 1, 2)).float().to(
            #         self.device) / 127.5 - 1.0
            #     lp_dist = self.lpips_vgg(c_t, p_t).squeeze().mean().item()
            #
            #     psnrs = [compute_psnr(orig_imgs[i], pois_imgs[i]) for i in range(sample_indices)]
            #     ssims = [sk_ssim(orig_imgs[i], pois_imgs[i], channel_axis=2, data_range=255) for i in
            #              range(sample_indices)]
            #     # --- 新增 GMSD 计算 ---
            #     # gmsds = [compute_gmsd(orig_imgs[i], pois_imgs[i], device=self.device) for i in range(sample_indices)]
            #
            #     mean_psnr = np.mean(psnrs) if len(psnrs) > 0 else self.psnr_min
            #     mean_ssim = np.mean(ssims) if len(ssims) > 0 else 0.0
            #     # mean_gmsd = np.mean(gmsds) if len(gmsds) > 0 else self.gmsd_max
            #
            #     # 归一化处理
            # psnr_norm = np.clip((mean_psnr - self.psnr_min) / (self.psnr_max - self.psnr_min), 0, 1)
            # # GMSD 越小质量越好，因此归一化通常使用 mean_gmsd / gmsd_max
            # # gmsd_norm = np.clip(mean_gmsd / max(1e-6, self.gmsd_max), 0, 1)
            #
            # # --- 聚合 IQ Penalty (Higher = Worse) ---
            # psnr_pen = (1.0 - psnr_norm) * self.w_psnr
            # ssim_pen = (1.0 - mean_ssim) * self.w_ssim
            # lpips_pen = lp_dist * self.w_lpips
            # # gmsd_pen = gmsd_norm * self.w_gmsd  # GMSD 本身就是值越大质量越差，直接作为 penalty 分量
            #
            # # iq_penalty = lpips_pen + psnr_pen + gmsd_pen + ssim_pen
            # iq_penalty = lpips_pen + psnr_pen + ssim_pen

            # --- 核心修改：加权持久性 Fitness ---
            persistence_asr_score = (r1_asr * 0.3) + (r10_asr * 0.7)
            fitness = (self.alpha * persistence_asr_score) + (self.beta * acc)
            # fitness = (self.alpha * persistence_asr_score) + (self.beta * acc) - (self.gamma * iq_penalty)

            record = {
                "asr": r1_asr,
                "asr_after_decay": r10_asr,
                "decay_rate": decay_rate,
                "acc": acc,
                # "avg_ssim": mean_ssim,
                # "psnr": mean_psnr,
                # "gmsd": mean_gmsd,  # 已添加
                # "lpips": lp_dist,
                # "iq_penalty": iq_penalty
            }
            return fitness, record

        except Exception as e:
            print(f"\n[Error during evaluation]: {e}")
            import traceback
            traceback.print_exc()
            return -1.0, {}
        finally:
            del global_model
            if 'local_model' in locals(): del local_model
            gc.collect()
            torch.cuda.empty_cache()

    def quick_eval_asr(self, model, individual):
        """辅助函数：快速评估当前模型的 ASR"""
        model.eval()
        correct_asr = 0
        total_pois = 0
        # 仅抽取少量样本快速评估，节省搜索时间
        clean_loader = DataLoader(self.helper.test_dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            for b_id, (data, _) in enumerate(clean_loader):
                if b_id > 5: break
                batch_np = (data.cpu().numpy().transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
                pois_batch_list = [apply_da_rgb(batch_np[i], individual) for i in range(len(batch_np))]
                pois_t = torch.from_numpy(np.array(pois_batch_list).transpose(0, 3, 1, 2)).float().to(
                    self.device) / 255.0

                out = model(pois_t)
                correct_asr += (out.argmax(1) == self.helper.params['target_label']).sum().item()
                total_pois += len(data)
        return correct_asr / total_pois if total_pois > 0 else 0


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

            # --- 增强打印区域 ---
            print(f"\n" + "=" * 60)
            print(f"GEN {gen + 1} BEST SUMMARY (Single-Shot Persistence):")
            print(f"  Fitness: {best_fit_gen:.4f} (Global: {best_fitness:.4f})")
            # 核心：展示持久性指标
            print(f"  ASR (R1 -> R10): {best_rec_gen['asr']:.4f} -> {best_rec_gen['asr_after_decay']:.4f}")
            print(f"  Decay Rate: {best_rec_gen['decay_rate']:.2%}")  # 以百分比显示衰减
            print(f"  ACC (After 10 Rounds): {best_rec_gen['acc']:.4f}")
            # print(f"  IQA -> SSIM: {best_rec_gen['avg_ssim']:.4f} | PSNR: {best_rec_gen['psnr']:.2f} | GMSD: {best_rec_gen['gmsd']:.4f} | LPIPS: {best_rec_gen['lpips']:.4f}")
            # print(f"  IQA -> SSIM: {best_rec_gen['avg_ssim']:.4f} | PSNR: {best_rec_gen['psnr']:.2f} | LPIPS: {best_rec_gen['lpips']:.4f}")
            # print(f"  Penalty: {best_rec_gen['iq_penalty']:.4f}")
            # 展示具体的增强算子
            print(f"  Combo: {[g[0] for g in best_ind_gen]}")
            print("=" * 60)

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

        # 保存逻辑... (略) model_cifar_Jan.04_14.59.42_DADBA_gen15_pop30_AS_20flr_cr100_alpha10_gama0_16_20
        # 保存结果
        out_path = Path(self.helper.params.get('out_dir', './ga_results/cifar_AS_GA/Demo26_01_05_gen20_pop25_AS_25flr_cr100_alpha5_gama0_noIQA'))
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
    parser.add_argument('--ga_gen', type=int, default=20)
    parser.add_argument('--ga_local_epochs', type=int, default=2, help='恶意客户端本地训练轮数')
    parser.add_argument('--ga_fl_rounds', type=int, default=25, help='评估时的联邦聚合轮数')
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