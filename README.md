# DA-GA-DBA

DA-GA-DBA 是一个面向联邦学习后门攻击研究的实验项目。项目在 DBA（Distributed Backdoor Attack）代码基础上，引入数据增强（DA）组合与遗传算法（GA）搜索，用于研究单次攻击（AS）和多次攻击（AM）场景中的攻击成功率、持续性、干净样本准确率与图像质量。

> 本仓库仅用于授权的安全研究与学术实验。公开版本不包含数据集、模型权重或完整实验结果。

## 主要内容

- 支持 CIFAR-10、MNIST、Tiny ImageNet、GTSRB 和 Loan 数据集的联邦学习实验
- 支持 Mean/FedAvg、Geometric Median 和 FoolsGold 聚合方式
- 支持分布式触发器与基于数据增强组合的触发器
- 提供 CIFAR-10、Tiny ImageNet、GTSRB 上的 AS/AM 遗传算法搜索入口
- 提供 GMSD、SSIM 等触发器图像质量评估工具

## 项目结构

```text
.
├── main.py                         # 联邦训练主入口
├── train.py / test.py              # 训练与评估流程
├── helper.py                       # 聚合、日志、检查点等公共逻辑
├── image_helper.py / image_train.py
├── loan_helper.py / loan_train.py
├── util.py                         # 数据增强、组合编码及图像质量工具
├── fed_ga_search_{as,am}_*.py      # 各数据集的 GA 搜索入口
├── ga_optimization.py              # 通用 GA 优化逻辑
├── compute_trigger_gmsd.py         # 触发器质量评估
├── models/                         # 模型定义
├── utils/                          # YAML 参数与数据预处理脚本
└── configs/trigger_combos/         # YAML 默认引用的增强组合
```

`data/`、`saved_models/`、`ga_results/` 等本地目录由程序下载或生成，已通过 `.gitignore` 排除。

## 环境安装

建议使用 Python 3.10，并根据本机 CUDA/CPU 环境先安装匹配的 PyTorch 与 torchvision。随后安装其余依赖：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch 安装方式请以其官方安装选择器为准，尤其是需要 GPU 加速时。

## 数据准备

- CIFAR-10、MNIST、GTSRB：首次运行时由 torchvision 下载到 `./data/`。
- Tiny ImageNet：将 `tiny-imagenet-200.zip` 放入 `utils/`，进入该目录执行 `bash process_tiny_data.sh`。程序最终期望目录为 `data/tiny-imagenet-200/{train,val}`。
- Loan：将 `lending-club-loan-data.zip` 放入 `utils/`，进入该目录执行 `bash process_loan_data.sh`。

数据集不要提交到 GitHub。

## 预训练模型

示例 YAML 默认设置了 `resumed_model: true`，并引用本地预训练权重。公开仓库不包含这些权重，因此运行前请选择一种方式：

1. 按 YAML 中的 `resumed_model_name` 放置对应检查点；或
2. 将 `resumed_model` 改为 `false`，从头训练。

模型检查点与训练输出会写入 `saved_models/`，该目录不会被 Git 跟踪。

## 运行联邦学习实验

代码默认启用 Visdom，可先启动服务：

```bash
python -m visdom.server -p 8098
```

另开终端运行实验，例如：

```bash
python main.py --params utils/cifar_params.yaml
python main.py --params utils/cifar_params_da_fcba14.yaml
python main.py --params utils/gtsrb_params.yaml
python main.py --params utils/tiny_params.yaml
python main.py --params utils/mnist_params.yaml
python main.py --params utils/loan_params.yaml
```

常用实验参数（联邦轮数、参与方数量、攻击轮次、聚合方式、Dirichlet 系数、检查点路径等）均在对应 YAML 中配置。

## 运行 GA 搜索

文件名中的 `as` 对应单次攻击搜索，`am` 对应多次攻击搜索。示例：

```bash
# CIFAR-10
python fed_ga_search_as_cifar.py --params utils/cifar_params.yaml --ga_pop_size 25 --ga_gen 20
python fed_ga_search_am_cifar.py --params utils/cifar_params_da_fcba14.yaml --ga_pop_size 25 --ga_gen 15

# GTSRB
python fed_ga_search_as_gtsrb.py --params utils/gtsrb_params.yaml
python fed_ga_search_am_gtsrb.py --params utils/gtsrb_params_da_fcba14.yaml

# Tiny ImageNet
python fed_ga_search_as_tiny.py --params utils/tiny_params.yaml
python fed_ga_search_am_tiny.py --params utils/tiny_params_da_fcba_14.yaml
```

搜索结果默认写入 `ga_results/`。若需要在普通训练中使用新的组合，请将生成的 `best_combo.json` 路径写入 YAML 的 `best_combo_path`。

随机组合基线与触发器质量评估示例：

```bash
python generate_random_combos.py --n 5 --seed 123 --out_dir ga_results/Random_DA
python compute_trigger_gmsd.py --params utils/tiny_params.yaml --split test --max_images 200 --backend pyiqa --device cuda
```

## GitHub 发布范围

当前 Git 暂存区仅包含核心源码、配置和文档。完整的纳入/排除清单与上传命令见 [UPLOAD_GUIDE.md](UPLOAD_GUIDE.md)。发布前还应根据上游代码许可和你的发布意愿补充合适的 `LICENSE`。

## 引用与致谢

本项目基于 DBA 项目继续开发。若相关代码对你的研究有帮助，请引用原论文：

```bibtex
@inproceedings{xie2020dba,
  title     = {DBA: Distributed Backdoor Attacks against Federated Learning},
  author    = {Chulin Xie and Keli Huang and Pin-Yu Chen and Bo Li},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  url       = {https://openreview.net/forum?id=rkgyS0VFvr}
}
```

上游与相关项目：

- [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/forum?id=rkgyS0VFvr)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
