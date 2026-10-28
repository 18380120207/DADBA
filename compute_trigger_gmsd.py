from __future__ import annotations

import argparse
import csv
import datetime
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from image_helper import ImageHelper
from utils import gmsd as gmsd_numpy
from utils import tensor_to_numpy_rgb

PYIQA_AVAILABLE = False
try:
    import torch
    import pyiqa

    PYIQA_AVAILABLE = True
except Exception:
    PYIQA_AVAILABLE = False

SKIMAGE_AVAILABLE = False
try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


def _pyiqa_metric(name: str, device: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if not PYIQA_AVAILABLE:
        raise RuntimeError("pyiqa is not available. Install it or use --backend numpy")

    import torch

    metric = pyiqa.create_metric(name).to(device)

    def _compute(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
        a = torch.from_numpy(a_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        b = torch.from_numpy(b_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            score = metric(a, b)
        return float(score.detach().cpu().item())

    return _compute


def _ssim_skimage(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    if not SKIMAGE_AVAILABLE:
        raise RuntimeError("scikit-image is required for SSIM when using --backend numpy")

    a = a_u8.astype(np.float32) / 255.0
    b = b_u8.astype(np.float32) / 255.0
    try:
        score = _skimage_ssim(a, b, channel_axis=2, data_range=1.0)
    except TypeError:
        score = _skimage_ssim(a, b, multichannel=True, data_range=1.0)
    return float(score)


@dataclass
class MetricRow:
    idx: int
    label: int
    gmsd_central: float
    ssim_central: float
    gmsd_dist_mean: float
    ssim_dist_mean: float
    gmsd_dist: Dict[int, float]
    ssim_dist: Dict[int, float]


def _load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _make_helper(params: dict) -> ImageHelper:
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    helper = ImageHelper(current_time=current_time, params=params, name=params.get("name", params.get("type", "exp")))
    helper.load_data()
    if hasattr(helper, "best_combo") is False and "best_combo_path" in helper.params:
        helper.load_best_combo()
    return helper


def _get_dataset(helper: ImageHelper, split: str):
    if split == "train":
        return helper.train_dataset
    if split in ("test", "val"):
        return helper.test_dataset
    raise ValueError("split must be 'train', 'test', or 'val'")


def _resolve_trigger_indices(helper: ImageHelper) -> List[int]:
    if hasattr(helper, "best_combo") and helper.best_combo:
        return list(range(len(helper.best_combo)))
    trigger_num = int(helper.params.get("trigger_num", 0))
    if trigger_num <= 0:
        raise RuntimeError("trigger count not found; set best_combo_path or trigger_num")
    return list(range(trigger_num))


def _compute_rows(helper: ImageHelper, split: str, max_images: int, backend: str, device: str) -> List[MetricRow]:
    dataset = _get_dataset(helper, split)
    trigger_indices = _resolve_trigger_indices(helper)
    channel_indices = [-1, 0, 1, 2]  # -1: global, 0/1/2: R/G/B

    if backend == "auto":
        backend = "pyiqa" if PYIQA_AVAILABLE else "numpy"

    if backend == "pyiqa":
        gmsd_fn = _pyiqa_metric("gmsd", device=device)
        ssim_fn = _pyiqa_metric("ssim", device=device)
    elif backend == "numpy":
        gmsd_fn = gmsd_numpy
        ssim_fn = _ssim_skimage
    else:
        raise ValueError("backend must be 'auto', 'numpy', or 'pyiqa'")

    n = min(max_images, len(dataset)) if max_images > 0 else len(dataset)
    rows: List[MetricRow] = []

    for idx in tqdm(range(n), desc=f"GMSD/SSIM ({split})"):
        image_t, label = dataset[idx]
        ref_rgb = tensor_to_numpy_rgb(image_t)

        gmsd_map: Dict[int, float] = {}
        ssim_map: Dict[int, float] = {}
        for ch in channel_indices:
            poisoned_t = helper.add_da_pattern(image_t, adversarial_index=ch)
            poisoned_rgb = tensor_to_numpy_rgb(poisoned_t)
            gmsd_map[ch] = float(gmsd_fn(ref_rgb, poisoned_rgb))
            ssim_map[ch] = float(ssim_fn(ref_rgb, poisoned_rgb))

        # Mean over individual R/G/B channels (excluding global)
        rgb_keys = [k for k in channel_indices if k != -1]
        gmsd_rgb_mean = float(np.mean([gmsd_map[k] for k in rgb_keys]))
        ssim_rgb_mean = float(np.mean([ssim_map[k] for k in rgb_keys]))

        rows.append(
            MetricRow(
                idx=idx,
                label=int(label),
                gmsd_central=gmsd_map[-1],
                ssim_central=ssim_map[-1],
                gmsd_dist_mean=gmsd_rgb_mean,
                ssim_dist_mean=ssim_rgb_mean,
                gmsd_dist={k: gmsd_map[k] for k in rgb_keys},
                ssim_dist={k: ssim_map[k] for k in rgb_keys},
            )
        )

    return rows


def _write_csv(rows: List[MetricRow], out_csv: str, trigger_indices: List[int]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    channel_indices = [-1, 0, 1, 2]
    rgb_keys = [k for k in channel_indices if k != -1]

    fieldnames = [
        "idx",
        "label",
        "gmsd_full",
        "ssim_full",
        "gmsd_r",
        "gmsd_g",
        "gmsd_b",
        "gmsd_rgb_mean",
        "ssim_r",
        "ssim_g",
        "ssim_b",
        "ssim_rgb_mean",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {
                "idx": r.idx,
                "label": r.label,
                "gmsd_full": r.gmsd_central,
                "ssim_full": r.ssim_central,
                "gmsd_r": r.gmsd_dist.get(0),
                "gmsd_g": r.gmsd_dist.get(1),
                "gmsd_b": r.gmsd_dist.get(2),
                "gmsd_rgb_mean": r.gmsd_dist_mean,
                "ssim_r": r.ssim_dist.get(0),
                "ssim_g": r.ssim_dist.get(1),
                "ssim_b": r.ssim_dist.get(2),
                "ssim_rgb_mean": r.ssim_dist_mean,
            }
            w.writerow(row)


def _print_summary(rows: List[MetricRow], trigger_indices: List[int]) -> None:
    g_full = np.array([r.gmsd_central for r in rows], dtype=np.float64)
    g_r = np.array([r.gmsd_dist.get(0) for r in rows], dtype=np.float64)
    g_g = np.array([r.gmsd_dist.get(1) for r in rows], dtype=np.float64)
    g_b = np.array([r.gmsd_dist.get(2) for r in rows], dtype=np.float64)
    g_mean = np.array([r.gmsd_dist_mean for r in rows], dtype=np.float64)

    s_full = np.array([r.ssim_central for r in rows], dtype=np.float64)
    s_r = np.array([r.ssim_dist.get(0) for r in rows], dtype=np.float64)
    s_g = np.array([r.ssim_dist.get(1) for r in rows], dtype=np.float64)
    s_b = np.array([r.ssim_dist.get(2) for r in rows], dtype=np.float64)
    s_mean = np.array([r.ssim_dist_mean for r in rows], dtype=np.float64)

    print("Summary")
    print(f"  images: {len(rows)}")
    print(f"  full(GMSD global): mean={g_full.mean():.6f}, std={g_full.std():.6f}, min={g_full.min():.6f}, max={g_full.max():.6f}")
    print(f"  R(GMSD): mean={g_r.mean():.6f}, std={g_r.std():.6f}, min={g_r.min():.6f}, max={g_r.max():.6f}")
    print(f"  G(GMSD): mean={g_g.mean():.6f}, std={g_g.std():.6f}, min={g_g.min():.6f}, max={g_g.max():.6f}")
    print(f"  B(GMSD): mean={g_b.mean():.6f}, std={g_b.std():.6f}, min={g_b.min():.6f}, max={g_b.max():.6f}")
    print(f"  RGB mean(GMSD): mean={g_mean.mean():.6f}, std={g_mean.std():.6f}, min={g_mean.min():.6f}, max={g_mean.max():.6f}")

    print(f"  full(SSIM global): mean={s_full.mean():.6f}, std={s_full.std():.6f}, min={s_full.min():.6f}, max={s_full.max():.6f}")
    print(f"  R(SSIM): mean={s_r.mean():.6f}, std={s_r.std():.6f}, min={s_r.min():.6f}, max={s_r.max():.6f}")
    print(f"  G(SSIM): mean={s_g.mean():.6f}, std={s_g.std():.6f}, min={s_g.min():.6f}, max={s_g.max():.6f}")
    print(f"  B(SSIM): mean={s_b.mean():.6f}, std={s_b.std():.6f}, min={s_b.min():.6f}, max={s_b.max():.6f}")
    print(f"  RGB mean(SSIM): mean={s_mean.mean():.6f}, std={s_mean.std():.6f}, min={s_mean.min():.6f}, max={s_mean.max():.6f}")


def _auto_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute GMSD/SSIM between clean and DA-GA-DBA triggers.")
    ap.add_argument("--params", type=str, default="utils/cifar_params.yaml", help="YAML params file for DA-GA-DBA")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "val"], help="Dataset split")
    ap.add_argument("--max_images", type=int, default=512, help="How many images to evaluate (<=0 means all)")
    ap.add_argument("--out_csv", type=str, default="gmsd_results/da_ga_dba_trigger_gmsd.csv", help="Output CSV path")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "pyiqa"], help="Metric backend")
    ap.add_argument("--metric", type=str, default=None, choices=["numpy", "pyiqa"], help="Deprecated alias for --backend")
    ap.add_argument("--device", type=str, default="auto", help="Device for pyiqa (cpu or cuda)")

    args = ap.parse_args()

    backend = args.backend if args.metric is None else args.metric
    device = _auto_device(args.device)

    params = _load_params(args.params)
    helper = _make_helper(params)
    trigger_indices = _resolve_trigger_indices(helper)

    rows = _compute_rows(helper, split=args.split, max_images=args.max_images, backend=backend, device=device)

    _write_csv(rows, args.out_csv, trigger_indices=trigger_indices)
    _print_summary(rows, trigger_indices=trigger_indices)
    print(f"CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
