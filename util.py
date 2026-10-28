"""
util.py

图像处理、触发器生成、相似度度量与 I/O 等工具集合。

功能包括：
- 多种图像增强基元（gamma, brightness/contrast, blur, sharpen, CLAHE, noise, bit-depth reduction 等）
- 将不同增强分别应用到 R/G/B 三通道：apply_da_rgb
- 触发器应用：apply_trigger（支持全图 / patch / masked）
- 相似度度量：ssim_batch（使用 skimage），gmsd_batch（近似实现）
- 图片加载/保存，写 txt 列表等 I/O helper
- 一个增强注册表（DA_REGISTRY），便于遗传算法选择/编码
"""

from typing import Tuple, Callable, Dict, Any, List, Sequence, Optional
import os
import json
import math
import random
import logging

import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float32, img_as_ubyte, img_as_float

from skimage import exposure, restoration, util
from typing import Callable, Dict
# Set up logger
logger = logging.getLogger("backdoor_repro.utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ---------------------------
# Basic image I/O helpers
# ---------------------------

def load_image(path: str, as_rgb: bool = True) -> np.ndarray:
    """
    Load an image and return as uint8 numpy array HxWx3 (RGB) or HxW (grayscale) depending on as_rgb.
    """
    img = Image.open(path).convert("RGB" if as_rgb else "L")
    arr = np.array(img)
    return arr


def save_image(path: str, img: np.ndarray) -> None:
    """
    Save numpy uint8 image to disk. Accepts HxWx3 or HxW arrays.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if img.dtype != np.uint8:
        # attempt to convert from float
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def write_txt_list(out_txt_path: str, list_of_tuples: Sequence[Tuple[str, int]]) -> None:
    """
    Write list of (relative_path, label) to txt (one per line: "relpath label").
    将一个包含路径和标签的列表写入文本文件。
    """
    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:    #out_txt_path：输出文本文件路径。
        for p, lbl in list_of_tuples:                       #list_of_tuples：一个列表，每个元素是一个元组，包含相对路径和标签。
            f.write(f"{p} {lbl}\n")


# ---------------------------
# Low-level helpers
# ---------------------------

def _to_gray_if_needed(img: np.ndarray) -> np.ndarray:
    """
    Ensure single channel return (2D array). If input is HxWx3, convert to grayscale float.
    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _clip_uint8(img: np.ndarray) -> np.ndarray:
    img = np.rint(img).astype(np.int32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------------------------
# Augmentation primitives
# Each function receives and returns numpy array with dtype uint8, range [0,255].
# For single-channel functions, input is HxW; for RGB ops, we handle channels outside.
# ---------------------------
#定义了一系列图像增强方法，每种方法接收一个 NumPy 数组作为输入，并返回处理后的 NumPy 数组。
import cv2
import numpy as np
import albumentations as A
from typing import Any, Callable, Dict, Tuple

# ---------------------------
# 工具函数
# ---------------------------
def _clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)

# ---------- Helper ----------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """确保返回 uint8 图片"""
    img = np.clip(img, 0, 255)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


# # -----------------------
# # e1: Gamma Correction
# # -----------------------
# def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
#     """
#     Gamma correction: gamma < 1 brightens, >1 darkens
#     """
#     # gamma = float(np.clip(gamma, 0.9, 1.1))
#     aug = A.RandomGamma(gamma_limit=(int(gamma * 80), int(gamma * 120)), p=1.0)
#     return aug(image=img)['image']
#
# # -----------------------
# # e2: Sharpen
# # -----------------------
# def sharpen(img: np.ndarray, alpha: float = 0.2, lightness: float = 0.8) -> np.ndarray:
#     """
#     Light sharpening using albumentations.Sharpen
#     """
#     aug = A.Sharpen(alpha=(alpha-0.1, alpha), lightness=(lightness-0.2, lightness), p=1.0)
#     return aug(image=img)['image']
# # -----------------------
# # e3: Change Brightness & Contrast
# # -----------------------
# def change_brightness_contrast(img: np.ndarray, brightness: float = 0.05, contrast: float = 0.05) -> np.ndarray:
#     """
#     brightness [-50, 50]; contrast [0.8, 1.2]
#     """
#     aug = A.RandomBrightnessContrast(
#         brightness_limit=(brightness),  # ±5% 亮度
#         contrast_limit=(contrast),  # ±5% 对比度
#         brightness_by_max=False, # 亮度变化更自然
#         ensure_safe_range=True,  # 防止溢出
#         p=1.0
#     )
#     return aug(image=img)['image']
# # -----------------------
# # e4: Unsharp Mask
# # -----------------------
# def unsharp_mask(img: np.ndarray, radius: int = 3, amount: float = 1.0) -> np.ndarray:
#     # 限制参数范围，避免过度锐化
#     # radius = np.clip(radius, 1, 2)
#     # amount = np.clip(amount, 0.3, 1.5)
#
#     # albumentations 内置 UnsharpMask
#     aug = A.UnsharpMask(
#         blur_limit=(radius, radius+8),  # 模糊核范围
#         sigma_limit=(0.1, 0.5),
#         alpha=(amount * 0.2, amount * 0.7),  # 锐化强度
#         threshold=15,  # 阈值避免放大噪声
#         p=1.0
#     )
#     return aug(image=img)['image']
#
# # -----------------------
# # e5: Wiener Filter (use cv2 bilateralFilter to simulate mild denoise)
# # -----------------------
# def wiener_filter(img: np.ndarray, d: int = 3) -> np.ndarray:
#     """
#     Approximate Wiener filter by mild bilateral filtering.
#     """
#     out = cv2.bilateralFilter(img, d, 25, 25)
#     return _clip_uint8(out)
# # -----------------------
# # e6: Histogram Equalization
# # -----------------------
# def hist_eq(img: np.ndarray) -> np.ndarray:
#     """
#     Global histogram equalization using Albumentations.
#     Equivalent to cv2.equalizeHist on Y channel.
#     """
#     transform = A.Equalize(mode='cv', by_channels=False, p=1.0)
#     # mode='cv' 使用 OpenCV 风格的实现，by_channels=False 代表只在亮度通道操作（与原版一致）
#     augmented = transform(image=img)
#     return augmented["image"]
# # -----------------------
# # e7: CLAHE
# # -----------------------
# def clahe(img: np.ndarray, clip_limit: float = 1.2, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
#     aug = A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=1.0)
#     return aug(image=img)['image']
#
#
# # -----------------------
# # e8: Gaussian Blur
# # -----------------------
# def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
#     aug = A.GaussianBlur(blur_limit=(ksize, ksize), sigma_limit=(0.1, 0.5), p=1.0)
#     return aug(image=img)['image']
# # -----------------------
# # e9: Median Blur
# # -----------------------
# def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
#     k = max(3, ksize)
#     if k % 2 == 0:
#         k += 1
#     aug = A.MedianBlur(blur_limit=k, p=1.0)
#     return aug(image=img)['image']
# # -----------------------
# # e10: Pixel Loss (Dropout)
# # -----------------------
# def pixel_dropout(img: np.ndarray, ) -> np.ndarray:
#     """
#     Randomly zero-out individual pixels with probability drop_prob.
#     """
#     aug = A.CoarseDropout(
#         num_holes_range=(1, 3),
#         hole_height_range=(1, 3),
#         hole_width_range=(1, 3),
#         fill=0,
#         p=1.0
#     )
#     return aug(image=img)['image']
# # -----------------------
# # e11: Rayleigh Noise (simulate via gamma noise)
# # -----------------------
# def rayleigh_noise(img: np.ndarray, scale: float = 5.0) -> np.ndarray:
#     """
#     Approximate Rayleigh noise via random distribution.
#     """
#     noise = np.random.rayleigh(scale, img.shape)
#     out = img.astype(np.float32) + noise
#     return _clip_uint8(out)
# # -----------------------
# # e12: Multiplicative Noise
# # -----------------------如果希望更强 → sigma 可增加到 0.05 或 0.08，但失真开始可见
# def multiplicative_noise(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
#     aug = A.MultiplicativeNoise(multiplier=(1 - sigma, 1 + sigma), per_channel=True, p=1.0)
#     return aug(image=img)['image']
# # -----------------------
# # e13: Gaussian Noise
# # -----------------------
# def gaussian_noise(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
#     aug = A.GaussNoise(std_range=(sigma*0.2, sigma*0.44),  p=1.0)
#     return aug(image=img)['image']
# # -----------------------
# # e14: Bit Depth Reduction
# # -----------------------
# def bit_depth_reduction(img: np.ndarray, bits: int = 7) -> np.ndarray:
#     bits = int(np.clip(bits, 1, 8))
#     levels = 2 ** bits
#     out = np.floor(img / (256 / levels)) * (256 / levels)
#     return _clip_uint8(out)
# # -----------------------
# # e15: Bit Depth Reduction
# # -----------------------
# # ---------------------
# def gaussian_blur_channel(channel: np.ndarray, sigma: float=0.5):
#     """Apply GaussianBlur to a single-channel image (numpy array float or uint8)."""
#     ksize = int(2 * round(3 * sigma) + 1)
#     if ksize < 3:
#         ksize = 3
#     # OpenCV expects HxW single channel
#     if channel.dtype != np.uint8:
#         # convert to 0-255 uint8 for cv2
#         ch = img_as_ubyte(channel)
#     else:
#         ch = channel
#     blurred = cv2.GaussianBlur(ch, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
#     return blurred
#
# # -----------------------
# # e16: Bit Depth Reduction
# # -----------------------
# def brightness_contrast_channel(channel: np.ndarray, alpha: float=1.2, beta: float=-0.2):
#     """Apply BC: out = alpha * channel + beta * 255"""
#     if channel.dtype != np.float32 and channel.dtype != np.float64:
#         chf = img_as_float(channel)
#     else:
#         chf = channel
#     out = alpha * chf + beta
#     # clamp
#     out = np.clip(out, 0.0, 1.0)
#     out_u8 = img_as_ubyte(out)
#     return out_u8
#
# # -----------------------
# # e17: Bit Depth Reduction
# # -----------------------
# def kernel_smooth_channel(channel: np.ndarray, delta: int=3):
#     """Apply simple uniform smoothing kernel of size delta x delta"""
#     if delta <= 1:
#         return channel
#     kernel = np.ones((delta, delta), dtype=np.float32) / (delta * delta)
#     # if channel is float 0-1, convert to 0-255 uint8 first for cv2.filter2D
#     if channel.dtype != np.uint8:
#         ch = img_as_ubyte(channel)
#     else:
#         ch = channel
#     smoothed = cv2.filter2D(ch, -1, kernel)
#     return smoothed
#
#
# # ---------- Augmentation functions ----------
# # e18
# def additive_noise(img: np.ndarray, var: float = 5.0) -> np.ndarray:
#     """
#     Additive Gaussian noise (small variance).
#     var: variance of gaussian noise (sigma^2). Default small (5.0).
#     """
#     # aug = A.GaussNoise(var_limit=(max(0.1, var - 2.0), var), per_channel=True, p=1.0)
#     aug = A.AdditiveNoise(noise_type="uniform",spatial_mode="constant",noise_params={"ranges": [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]})
#     out = aug(image=img)['image']
#     return _to_uint8(out)
#
# # e19
# def shot_noise(img: np.ndarray, prob: float = 0.005) -> np.ndarray:
#     """
#     Shot (Poisson) noise implementation that is robust to both single-channel
#     and multi-channel images.
#
#     Parameters
#     ----------
#     img : np.ndarray
#         Input image. Can be shape (H,W), (H,W,1) or (H,W,3) etc.
#     prob : float
#         Per-pixel probability that the pixel will be replaced by a Poisson sample.
#         Recommended small (e.g. 0.001).
#
#     Returns
#     -------
#     np.ndarray
#         uint8 image in range [0,255].
#     """
#     # ensure float in [0,1]
#     orig_dtype = img.dtype
#     img_f = img.astype(np.float32) / 255.0
#
#     # If image has a singleton channel dimension, remove it for simpler handling,
#     # but remember to restore channel axis if needed.
#     squeezed = False
#     if img_f.ndim == 3 and img_f.shape[2] == 1:
#         img_f = img_f[..., 0]
#         squeezed = True
#
#     # Create Poisson noisy version (same shape as img_f)
#     vals = 2 ** 8 - 1  # use 8-bit quantization baseline
#     noisy = np.random.poisson(np.clip(img_f * vals, 0, vals)) / float(vals)
#
#     # Generate mask per-pixel, then expand to channels if needed
#     if img_f.ndim == 2:
#         # single-channel
#         mask2d = (np.random.rand(img_f.shape[0], img_f.shape[1]) < prob)
#         out = img_f.copy()
#         out[mask2d] = noisy[mask2d]
#         # restore channel axis if original had size 1 channel
#         if squeezed:
#             out = out[:, :, None]
#     else:
#         # multi-channel: img_f shape (H, W, C)
#         h, w, c = img_f.shape
#         mask2d = (np.random.rand(h, w) < prob)  # per-pixel mask
#         mask = np.repeat(mask2d[:, :, None], c, axis=2)
#         out = img_f.copy()
#         out[mask] = noisy[mask]
#
#     # convert back to uint8
#     out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
#     return out
#
# # e20
# def iso_noise(img: np.ndarray, intensity: float = 0.03) -> np.ndarray:
#     """
#     Use albumentations.ISONoise, small intensity by default.
#     intensity: float in [0, 0.1] recommended.
#     """
#     aug = A.ISONoise(color_shift=(0.0, 0.02), intensity=(intensity, 0.07), p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e21
# def motion_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
#     """
#     Motion blur using albumentations.MotionBlur.
#     ksize: kernel size, small odd number 3..7 recommended.
#     """
#     # k = max(3, min(7, int(ksize) | 1))  # ensure odd and in [3,7]
#     aug = A.MotionBlur(blur_limit=(ksize, ksize), p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e22
# def glass_blur(img: np.ndarray, sigma: float = 0.3, max_delta: int = 1, iterations: int = 1) -> np.ndarray:
#     """
#     Albumentations GlassBlur - gentle by default.
#     sigma: blur sigma, max_delta: pixel shuffle amount
#     """
#     aug = A.GlassBlur(sigma=sigma, max_delta=max_delta, iterations=iterations, p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e23
# def defocus(img: np.ndarray, blur_limit: int = 3) -> np.ndarray:
#     """
#     Simulate slight defocus/blur using albumentations.Blur with small kernel.
#     blur_limit: 1..5
#     """
#     # bl = max(1, min(5, int(blur_limit)))
#     # aug = A.Blur(blur_limit=(bl, bl), p=1.0)
#     aug = A.Defocus(radius=(1,3),alias_blur=(0,0.1), p=1.0 )
#     return _to_uint8(aug(image=img)['image'])
#
# # e24
# def zoom_blur(img: np.ndarray, strength: float = 0.03, steps: int = 6) -> np.ndarray:
#     """
#     Radial zoom blur implemented by blending scaled images.
#     strength: how strong zoom effect (0.02..0.2 small)
#     steps: number of blended scales (6 default)
#     """
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     out = np.zeros_like(img, dtype=np.float32)
#     alpha_total = 0.0
#     for i in range(steps):
#         frac = 1.0 + (strength * (i / (steps - 1)))
#         # scale about center
#         M = cv2.getRotationMatrix2D(center, 0, frac)
#         scaled = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         weight = 1.0 / steps
#         out += scaled.astype(np.float32) * weight
#         alpha_total += weight
#     out = out / alpha_total
#     return _to_uint8(out)
#
# # e25
# def rotate_small(img: np.ndarray, angle: float = 3.0) -> np.ndarray:
#     """
#     Small rotation using albumentations.Rotate.
#     angle: degrees in [-15,15] recommended; default 5 degrees.
#     """
#     aug = A.Rotate(limit=( -abs(angle), abs(angle) ), border_mode=cv2.BORDER_REFLECT, p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e26
# def random_scale(img: np.ndarray, scale_limit: float = 0.04) -> np.ndarray:
#     """
#     Random small scale (zoom in/out) using albumentations.RandomScale + PadIfNeeded/Crop.
#     scale_limit: e.g., 0.08 => scale in [1-0.08, 1+0.08]
#     """
#     # Albumentations RandomScale expects scale_limit param as float
#     aug = A.Compose([
#         A.RandomScale(scale_limit=(max(0.01, -scale_limit), scale_limit), p=1.0),
#         A.PadIfNeeded(min_height=img.shape[0], min_width=img.shape[1], border_mode=cv2.BORDER_REFLECT, p=1.0),
#         A.CenterCrop(height=img.shape[0], width=img.shape[1], p=1.0)
#     ])
#     return _to_uint8(aug(image=img)['image'])
#
# # e27
# def perspective_small(img: np.ndarray, scale: float = 0.02) -> np.ndarray:
#     """
#     Small perspective warp using albumentations.Perspective.
#     scale: magnitude of perspective distortions (0.01..0.08 small)
#     """
#     aug = A.Perspective(scale=(0.01, scale), p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e28
# # def optical_distortion_small(img: np.ndarray, distort_limit: float = 0.01, shift_limit: float = 0.005) -> np.ndarray:
# #     """
# #     Albumentations OpticalDistortion with small limits.
# #     """
# #     aug = A.OpticalDistortion(distort_limit=( -abs(distort_limit), abs(distort_limit) ), shift_limit=( -abs(shift_limit), abs(shift_limit) ), p=1.0)
# #     return _to_uint8(aug(image=img)['image'])
#
# # e29
# def random_crop_from_borders(img: np.ndarray, max_border: int = 1) -> np.ndarray:
#     """
#     Crop a small number of pixels from a random border (top/bottom/left/right) and then pad back.
#     max_border: at most this many pixels will be cropped from one side.
#     """
#     h, w = img.shape[:2]
#     # choose side
#     side = np.random.choice(['top','bottom','left','right'])
#     crop = np.random.randint(1, max(1, max_border+1))
#     if side == 'top':
#         cropped = img[crop:, :]
#         padded = cv2.copyMakeBorder(cropped, crop, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
#     elif side == 'bottom':
#         cropped = img[:-crop, :]
#         padded = cv2.copyMakeBorder(cropped, 0, crop, 0, 0, borderType=cv2.BORDER_REFLECT)
#     elif side == 'left':
#         cropped = img[:, crop:]
#         padded = cv2.copyMakeBorder(cropped, 0, 0, crop, 0, borderType=cv2.BORDER_REFLECT)
#     else:  # right
#         cropped = img[:, :-crop]
#         padded = cv2.copyMakeBorder(cropped, 0, 0, 0, crop, borderType=cv2.BORDER_REFLECT)
#     return _to_uint8(padded)
#
# # e30
# def superpixels_small(img: np.ndarray, p_replace: float = 0.01, n_segments: int = 200) -> np.ndarray:
#     """
#     Use albumentations.Superpixels but with low replacement ratio.
#     p_replace: fraction of superpixels to replace with their average color (small).
#     n_segments: number of superpixels (higher -> smaller pieces); keep moderate.
#     """
#     aug = A.Superpixels(p_replace=(0.0, p_replace), n_segments=(100, n_segments), p=1.0)
#     return _to_uint8(aug(image=img)['image'])
#
# # e31
# def random_shadow(img: np.ndarray, shadow_intensity: float = 0.10) -> np.ndarray:
#     """
#     Create a single soft polygon shadow and darken underlying pixels slightly.
#     shadow_intensity: 0.1..0.6 (0.35 default), fraction to reduce brightness inside shadow.
#     """
#     h, w = img.shape[:2]
#     # polygon roughly across the image
#     x1 = np.random.randint(0, w // 4)
#     x2 = np.random.randint(3 * w // 4, w)
#     y1 = np.random.randint(0, h // 4)
#     y2 = np.random.randint(3 * h // 4, h)
#     poly = np.array([[
#         (x1, y1),
#         (int(0.5 * (x1 + x2)), max(0, y1 - np.random.randint(0, h//8))),
#         (x2, y1 + np.random.randint(0, h//8)),
#         (x2, y2),
#         (int(0.5 * (x1 + x2)), y2 + np.random.randint(0, h//8)),
#         (x1, y2)
#     ]], dtype=np.int32)
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(mask, poly, 255)
#     # blur mask to make soft shadow
#     k = max(3, min(h//10 + 1, 31))
#     mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)
#     mask_f = (mask.astype(np.float32) / 255.0) * shadow_intensity
#     out = img.astype(np.float32) * (1.0 - mask_f[:, :, None])
#     return _to_uint8(out)
#
# # e32
# def plasma_shadow(img: np.ndarray, strength: float = 0.12) -> np.ndarray:
#     """
#     Generate a 'plasma' style soft shadow using blurred noise, then darken.
#     strength: 0.1..0.4 small
#     """
#     h, w = img.shape[:2]
#     noise = np.random.randn(h, w).astype(np.float32)
#     # multi-scale blur to get smooth plasma
#     for k in (3,7,13):
#         noise = cv2.GaussianBlur(noise, (k | 1, k | 1), 0)
#     # normalize to 0..1
#     nmin, nmax = noise.min(), noise.max()
#     noise = (noise - nmin) / (nmax - nmin + 1e-8)
#     # threshold/scale to shadow mask (take dark areas)
#     mask = (noise < 0.5).astype(np.float32) * (0.5 - noise)
#     mask = cv2.GaussianBlur(mask, (31,31), 0)
#     mask = np.clip(mask, 0.0, 1.0) * strength
#     out = img.astype(np.float32) * (1.0 - mask[:, :, None])
#     return _to_uint8(out)
#
# # e33
# def ringing_overshoot(img: np.ndarray, amount: float = 0.3, lightness: float = 0.1) -> np.ndarray:
#     """
#     Simulate slight ringing/overshoot by applying a strong-ish Sharpen then blending back.
#     amount: how strong the sharpen is (0.2..1.0). Keep moderate.
#     """
#     aug = A.Sharpen(alpha=(0.2, amount), lightness=(max(0.5, lightness - 0.2), lightness), p=1.0)
#     sharp = aug(image=img)['image'].astype(np.float32)
#     out = cv2.addWeighted(img.astype(np.float32), 1.0 - 0.35, sharp, 0.35, 0.0)  # blend small amount
#     return _to_uint8(out)
#
# # ---------------------------
# # DA registry
# # Each entry: name -> (callable, default_param_dict)
# # callable signature: callable(img: np.ndarray, **params) -> np.ndarray
# # 将所有图像增强方法及其默认参数注册到一个字典中。
# # 用途：便于后续通过名称调用这些方法，并支持遗传算法等优化过程
# # ---------------------------
#
# DA_REGISTRY: Dict[str, Tuple[Callable[..., np.ndarray], Dict[str, Any]]] = {
#     # e1: Gamma Correction
#     "gamma": (
#         gamma_correction,
#         {"gamma": 1.0},  # 略微提亮
#     ),
#
#     # e2: Sharpen
#     "sharpen": (
#         sharpen,
#         {"alpha": 0.3, "lightness": 0.8},  # 轻微锐化
#     ),
#
#     # e3: Change Brightness Contrast
#     "brightness_contrast": (
#         change_brightness_contrast,
#         {"brightness": 0.05, "contrast": 0.05},  # 提亮一点对比度
#     ),
#
#     # e4: Unsharp Mask
#     "unsharp": (
#         unsharp_mask,
#         {"radius": 3, "amount": 1.0},  # 小范围增强
#     ),
#
#     # e5: Wiener Filter
#     "wiener_filter": (
#         wiener_filter,
#         {"d": 3},  # 双边滤波近似去噪
#     ),
#
#     # e6: Histogram Equalization
#     "hist_eq": (
#         hist_eq,
#         {},  # 无参数
#     ),
#
#     # e7: CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     "clahe": (
#         clahe,
#         {"clip_limit": 1.2, "tile_grid_size": (8, 8)},  # 默认轻度增强
#     ),
#
#     # e8: Gaussian Blur
#     "gaussian_blur": (
#         gaussian_blur,
#         {"ksize": 3},  # 轻微模糊
#     ),
#
#     # e9: Median Blur
#     "median_blur": (
#         median_blur,
#         {"ksize": 3},  # 小核去噪
#     ),
#
#     # e10: Pixel Loss (Dropout)
#     "pixel_loss": (
#         pixel_dropout,
#           {}, # 轻微像素丢失
#     ),
#
#     # e11: Rayleigh Noise
#     "rayleigh_noise": (
#         rayleigh_noise,
#         {"scale": 5.0},  # 轻度噪声
#     ),
#
#     # e12: Multiplicative Noise
#     "mult_noise": (
#         multiplicative_noise,
#         {"sigma": 0.05},  # 轻微乘性噪声
#     ),
#
#     # e13: Gaussian Noise
#     "gauss_noise": (
#         gaussian_noise,
#         {"sigma": 1.0},  # 轻微高斯噪声
#     ),
#
#     # e14: Bit-depth Reduction (BdR)
#     "bit_reduce": (
#         bit_depth_reduction,
#         {"bits": 7},  # 降低少量比特深度
#     ),
#
#     #e15
#     "gaussian_blur_channel":
#         (gaussian_blur_channel,
#          {"sigma": 0.5}
#          ),
#
#     # e16
#     "brightness_contrast_channel":
#         (brightness_contrast_channel,
#          {"alpha": 1.2, "beta":-0.2}
#          ),
#
#     # e17
#     "kernel_smooth_channel":
#         (kernel_smooth_channel,
#          {"delta": 3}
#          ),
#
#     # e18 轻微高斯噪声
#     "AdditiveNoise": (additive_noise, {"var": 5.0}),
#     # e19 Shot/Poisson 噪声
#     "ShotNoise": (shot_noise, {"prob": 0.005}),
#     # e20 ISO 噪声（摄像头风格）
#     "ISONoise": (iso_noise, {"intensity": 0.01}),
#     # e21 轻微运动模糊
#     "MotionBlur": (motion_blur, {"ksize": 3}),
#     # e22 轻微玻璃模糊
#     "GlassBlur": (glass_blur, {"sigma": 0.3, "max_delta": 1, "iterations": 1}),
#     # e23 小幅失焦模糊
#     "Defocus": (defocus, {"blur_limit": 3}),
#     # e24 轻微缩放模糊（radial）
#     "ZoomBlur": (zoom_blur, {"strength": 0.03, "steps": 6}),
#     # e25 小角度旋转
#     "Rotate": (rotate_small, {"angle": 3.0}),
#     # e26 小比例随机缩放
#     "RandomScale": (random_scale, {"scale_limit": 0.04}),
#     # e27 透视畸变（小）
#     "Perspective": (perspective_small, {"scale": 0.02}),
#     # e28 光学畸变（小）
#     # "OpticalDistortion": (optical_distortion_small, {"distort_limit": 0.01, "shift_limit": 0.005}),
#     # e29 从边界裁切少量像素并补回（模拟裁切偏移）
#     "RandomCropFromBorders": (random_crop_from_borders, {"max_border": 1}),
#     # e30 超像素小替换
#     "Superpixels": (superpixels_small, {"p_replace": 0.01, "n_segments": 200}),
#     # e31 随机单片阴影
#     "RandomShadow": (random_shadow, {"shadow_intensity": 0.10}),
#     # e32 等离子/模糊阴影
#     "PlasmaShadow": (plasma_shadow, {"strength": 0.12}),
#     # e33 轻微的锐化/振铃过冲效果
#     "RingingOvershoot": (ringing_overshoot, {"amount": 0.2, "lightness": 0.1}),
# }


# ---------------------------
# 图像增强函数（轻度增强）
# ---------------------------
# -----------------------
# e1: Gamma Correction
# -----------------------
def gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma correction: gamma < 1 brightens, >1 darkens
    """
    # gamma = float(np.clip(gamma, 0.9, 1.1))
    aug = A.RandomGamma(gamma_limit=(int(gamma * 70), int(gamma * 150)), p=1.0)
    return aug(image=img)['image']

# -----------------------
# e2: Sharpen
# -----------------------
def sharpen(img: np.ndarray, alpha: float = 0.5, lightness: float = 1.0) -> np.ndarray:
    """
    Light sharpening using albumentations.Sharpen
    """
    aug = A.Sharpen(alpha=(alpha-0.2, alpha), lightness=(lightness-0.5, lightness), p=1.0)
    return aug(image=img)['image']
# -----------------------
# e3: Change Brightness & Contrast
# -----------------------
def change_brightness_contrast(img: np.ndarray, brightness: float = 0.2, contrast: float = 0.2) -> np.ndarray:
    """
    brightness [-50, 50]; contrast [0.8, 1.2]
    """
    aug = A.RandomBrightnessContrast(
        brightness_limit=(brightness),  # ±5% 亮度
        contrast_limit=(contrast),  # ±5% 对比度
        brightness_by_max=False, # 亮度变化更自然
        ensure_safe_range=True,  # 防止溢出
        p=1.0
    )
    return aug(image=img)['image']
# -----------------------
# e4: Unsharp Mask
# -----------------------
def unsharp_mask(img: np.ndarray, radius: int = 5, amount: float = 1.2) -> np.ndarray:
    # 限制参数范围，避免过度锐化
    # radius = np.clip(radius, 1, 2)
    # amount = np.clip(amount, 0.3, 1.5)

    # albumentations 内置 UnsharpMask
    aug = A.UnsharpMask(
        blur_limit=(radius, radius+10),  # 模糊核范围
        sigma_limit=(0.5, 1.5),
        alpha=(amount * 0.4, amount * 0.8),  # 锐化强度
        threshold=8,  # 阈值避免放大噪声
        p=1.0
    )
    return aug(image=img)['image']

# -----------------------
# e5: Wiener Filter (use cv2 bilateralFilter to simulate mild denoise)
# -----------------------
def wiener_filter(img: np.ndarray, d: int = 5) -> np.ndarray:
    """
    Approximate Wiener filter by mild bilateral filtering.
    """
    out = cv2.bilateralFilter(img, d, 25, 25)
    return _clip_uint8(out)
# -----------------------
# e6: Histogram Equalization
# -----------------------
def hist_eq(img: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization using Albumentations.
    Equivalent to cv2.equalizeHist on Y channel.
    """
    transform = A.Equalize(mode='cv', by_channels=False, p=1.0)
    # mode='cv' 使用 OpenCV 风格的实现，by_channels=False 代表只在亮度通道操作（与原版一致）
    augmented = transform(image=img)
    return augmented["image"]
# -----------------------
# e7: CLAHE
# -----------------------
def clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    aug = A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, p=1.0)
    return aug(image=img)['image']


# -----------------------
# e8: Gaussian Blur
# -----------------------
def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    aug = A.GaussianBlur(blur_limit=(ksize, ksize), sigma_limit=(0.5, 3), p=1.0)
    return aug(image=img)['image']
# -----------------------
# e9: Median Blur
# -----------------------
def median_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = max(3, ksize)
    if k % 2 == 0:
        k += 1
    aug = A.MedianBlur(blur_limit=k, p=1.0)
    return aug(image=img)['image']
# -----------------------
# e10: Pixel Loss (Dropout)
# -----------------------
def pixel_dropout(img: np.ndarray, ) -> np.ndarray:
    """
    Randomly zero-out individual pixels with probability drop_prob.
    """
    aug = A.CoarseDropout(
        num_holes_range=(4, 4),
        hole_height_range=(4, 4),
        hole_width_range=(4, 4),
        fill=0,
        p=1.0
    )
    return aug(image=img)['image']
# -----------------------
# e11: Rayleigh Noise (simulate via gamma noise)
# -----------------------
def rayleigh_noise(img: np.ndarray, scale: float = 10.0) -> np.ndarray:
    """
    Approximate Rayleigh noise via random distribution.
    """
    noise = np.random.rayleigh(scale, img.shape)
    out = img.astype(np.float32) + noise
    return _clip_uint8(out)
# -----------------------
# e12: Multiplicative Noise
# -----------------------如果希望更强 → sigma 可增加到 0.05 或 0.08，但失真开始可见
def multiplicative_noise(img: np.ndarray, sigma: float = 0.08) -> np.ndarray:
    aug = A.MultiplicativeNoise(multiplier=(1 - sigma, 1 + sigma), per_channel=True, p=1.0)
    return aug(image=img)['image']
# -----------------------
# e13: Gaussian Noise
# -----------------------
def gaussian_noise(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    aug = A.GaussNoise(std_range=(sigma*0.2, sigma*0.44),  p=1.0)
    return aug(image=img)['image']
# -----------------------
# e14: Bit Depth Reduction
# -----------------------
def bit_depth_reduction(img: np.ndarray, bits: int = 5) -> np.ndarray:
    bits = int(np.clip(bits, 1, 8))
    levels = 2 ** bits
    out = np.floor(img / (256 / levels)) * (256 / levels)
    return _clip_uint8(out)
# -----------------------
# e15: Bit Depth Reduction
# -----------------------
# ---------------------
def gaussian_blur_channel(channel: np.ndarray, sigma: float=1.0):
    """Apply GaussianBlur to a single-channel image (numpy array float or uint8)."""
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize < 3:
        ksize = 3
    # OpenCV expects HxW single channel
    if channel.dtype != np.uint8:
        # convert to 0-255 uint8 for cv2
        ch = img_as_ubyte(channel)
    else:
        ch = channel
    blurred = cv2.GaussianBlur(ch, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred

# -----------------------
# e16: Bit Depth Reduction
# -----------------------
def brightness_contrast_channel(channel: np.ndarray, alpha: float=1.4, beta: float=-0.3):
    """Apply BC: out = alpha * channel + beta * 255"""
    if channel.dtype != np.float32 and channel.dtype != np.float64:
        chf = img_as_float(channel)
    else:
        chf = channel
    out = alpha * chf + beta
    # clamp
    out = np.clip(out, 0.0, 1.0)
    out_u8 = img_as_ubyte(out)
    return out_u8

# -----------------------
# e17: Bit Depth Reduction
# -----------------------
def kernel_smooth_channel(channel: np.ndarray, delta: int=5):
    """Apply simple uniform smoothing kernel of size delta x delta"""
    if delta <= 1:
        return channel
    kernel = np.ones((delta, delta), dtype=np.float32) / (delta * delta)
    # if channel is float 0-1, convert to 0-255 uint8 first for cv2.filter2D
    if channel.dtype != np.uint8:
        ch = img_as_ubyte(channel)
    else:
        ch = channel
    smoothed = cv2.filter2D(ch, -1, kernel)
    return smoothed


# ---------- Augmentation functions ----------
# e18
def additive_noise(img: np.ndarray, var: float = 5.0) -> np.ndarray:
    """
    Additive Gaussian noise (small variance).
    var: variance of gaussian noise (sigma^2). Default small (5.0).
    """
    # aug = A.GaussNoise(var_limit=(max(0.1, var - 2.0), var), per_channel=True, p=1.0)
    aug = A.AdditiveNoise(noise_type="uniform",spatial_mode="constant",noise_params={"ranges": [(-0.4, 0.4), (-0.3, 0.3), (-0.3, 0.3)]})
    out = aug(image=img)['image']
    return _to_uint8(out)

# e19
def shot_noise(img: np.ndarray, prob: float = 0.02) -> np.ndarray:
    """
    Shot (Poisson) noise implementation that is robust to both single-channel
    and multi-channel images.

    Parameters
    ----------
    img : np.ndarray
        Input image. Can be shape (H,W), (H,W,1) or (H,W,3) etc.
    prob : float
        Per-pixel probability that the pixel will be replaced by a Poisson sample.
        Recommended small (e.g. 0.001).

    Returns
    -------
    np.ndarray
        uint8 image in range [0,255].
    """
    # ensure float in [0,1]
    orig_dtype = img.dtype
    img_f = img.astype(np.float32) / 255.0

    # If image has a singleton channel dimension, remove it for simpler handling,
    # but remember to restore channel axis if needed.
    squeezed = False
    if img_f.ndim == 3 and img_f.shape[2] == 1:
        img_f = img_f[..., 0]
        squeezed = True

    # Create Poisson noisy version (same shape as img_f)
    vals = 2 ** 8 - 1  # use 8-bit quantization baseline
    noisy = np.random.poisson(np.clip(img_f * vals, 0, vals)) / float(vals)

    # Generate mask per-pixel, then expand to channels if needed
    if img_f.ndim == 2:
        # single-channel
        mask2d = (np.random.rand(img_f.shape[0], img_f.shape[1]) < prob)
        out = img_f.copy()
        out[mask2d] = noisy[mask2d]
        # restore channel axis if original had size 1 channel
        if squeezed:
            out = out[:, :, None]
    else:
        # multi-channel: img_f shape (H, W, C)
        h, w, c = img_f.shape
        mask2d = (np.random.rand(h, w) < prob)  # per-pixel mask
        mask = np.repeat(mask2d[:, :, None], c, axis=2)
        out = img_f.copy()
        out[mask] = noisy[mask]

    # convert back to uint8
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

# e20
def iso_noise(img: np.ndarray, intensity: float = 0.2) -> np.ndarray:
    """
    Use albumentations.ISONoise, small intensity by default.
    intensity: float in [0, 0.1] recommended.
    """
    aug = A.ISONoise(color_shift=(0.0, 0.02), intensity=(intensity, 0.4), p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e21
def motion_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Motion blur using albumentations.MotionBlur.
    ksize: kernel size, small odd number 3..7 recommended.
    """
    # k = max(3, min(7, int(ksize) | 1))  # ensure odd and in [3,7]
    aug = A.MotionBlur(blur_limit=(ksize, ksize+2), p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e22
def glass_blur(img: np.ndarray, sigma: float = 0.5, max_delta: int = 3, iterations: int = 2) -> np.ndarray:
    """
    Albumentations GlassBlur - gentle by default.
    sigma: blur sigma, max_delta: pixel shuffle amount
    """
    aug = A.GlassBlur(sigma=sigma, max_delta=max_delta, iterations=iterations, p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e23
def defocus(img: np.ndarray, blur_limit: int = 3) -> np.ndarray:
    """
    Simulate slight defocus/blur using albumentations.Blur with small kernel.
    blur_limit: 1..5
    """
    # bl = max(1, min(5, int(blur_limit)))
    # aug = A.Blur(blur_limit=(bl, bl), p=1.0)
    aug = A.Defocus(radius=(3,6),alias_blur=(0,0.1), p=1.0 )
    return _to_uint8(aug(image=img)['image'])

# e24
def zoom_blur(img: np.ndarray, strength: float = 0.1, steps: int = 10) -> np.ndarray:
    """
    Radial zoom blur implemented by blending scaled images.
    strength: how strong zoom effect (0.02..0.2 small)
    steps: number of blended scales (6 default)
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    out = np.zeros_like(img, dtype=np.float32)
    alpha_total = 0.0
    for i in range(steps):
        frac = 1.0 + (strength * (i / (steps - 1)))
        # scale about center
        M = cv2.getRotationMatrix2D(center, 0, frac)
        scaled = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        weight = 1.0 / steps
        out += scaled.astype(np.float32) * weight
        alpha_total += weight
    out = out / alpha_total
    return _to_uint8(out)

# e25
def rotate_small(img: np.ndarray, angle: float = 7.0) -> np.ndarray:
    """
    Small rotation using albumentations.Rotate.
    angle: degrees in [-15,15] recommended; default 5 degrees.
    """
    aug = A.Rotate(limit=( -abs(angle), abs(angle) ), border_mode=cv2.BORDER_REFLECT, p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e26
def random_scale(img: np.ndarray, scale_limit: float = 0.2) -> np.ndarray:
    """
    Random small scale (zoom in/out) using albumentations.RandomScale + PadIfNeeded/Crop.
    scale_limit: e.g., 0.08 => scale in [1-0.08, 1+0.08]
    """
    # Albumentations RandomScale expects scale_limit param as float
    aug = A.Compose([
        A.RandomScale(scale_limit=(max(0.01, -scale_limit), scale_limit), p=1.0),
        A.PadIfNeeded(min_height=img.shape[0], min_width=img.shape[1], border_mode=cv2.BORDER_REFLECT, p=1.0),
        A.CenterCrop(height=img.shape[0], width=img.shape[1], p=1.0)
    ])
    return _to_uint8(aug(image=img)['image'])

# e27
def perspective_small(img: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """
    Small perspective warp using albumentations.Perspective.
    scale: magnitude of perspective distortions (0.01..0.08 small)
    """
    aug = A.Perspective(scale=(0.05, scale), p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e28
def optical_distortion_small(img: np.ndarray, distort_limit: float = 0.05, shift_limit: float = 0.005) -> np.ndarray:
    """
    Albumentations OpticalDistortion with small limits.
    """
    aug = A.OpticalDistortion(distort_limit=( -abs(distort_limit), abs(distort_limit) ), p=1.0)
    return _to_uint8(aug(image=img)['image'])

# e29
def random_crop_from_borders(img: np.ndarray, max_border: int = 3) -> np.ndarray:
    """
    Crop a small number of pixels from a random border (top/bottom/left/right) and then pad back.
    max_border: at most this many pixels will be cropped from one side.
    """
    h, w = img.shape[:2]
    # choose side
    side = np.random.choice(['top','bottom','left','right'])
    crop = np.random.randint(1, max(1, max_border+1))
    if side == 'top':
        cropped = img[crop:, :]
        padded = cv2.copyMakeBorder(cropped, crop, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
    elif side == 'bottom':
        cropped = img[:-crop, :]
        padded = cv2.copyMakeBorder(cropped, 0, crop, 0, 0, borderType=cv2.BORDER_REFLECT)
    elif side == 'left':
        cropped = img[:, crop:]
        padded = cv2.copyMakeBorder(cropped, 0, 0, crop, 0, borderType=cv2.BORDER_REFLECT)
    else:  # right
        cropped = img[:, :-crop]
        padded = cv2.copyMakeBorder(cropped, 0, 0, 0, crop, borderType=cv2.BORDER_REFLECT)
    return _to_uint8(padded)

# e30
# def superpixels_small(img: np.ndarray, p_replace: float = 0.1, n_segments: int = 50) -> np.ndarray:
#     """
#     Use albumentations.Superpixels but with low replacement ratio.
#     p_replace: fraction of superpixels to replace with their average color (small).
#     n_segments: number of superpixels (higher -> smaller pieces); keep moderate.
#     """
#     aug = A.Superpixels(p_replace=(0.0, p_replace), n_segments=(20, n_segments), p=1.0)
#     return _to_uint8(aug(image=img)['image'])

# e31
def random_shadow(img: np.ndarray, shadow_intensity: float = 0.35) -> np.ndarray:
    """
    Create a single soft polygon shadow and darken underlying pixels slightly.
    shadow_intensity: 0.1..0.6 (0.35 default), fraction to reduce brightness inside shadow.
    """
    h, w = img.shape[:2]
    # polygon roughly across the image
    x1 = np.random.randint(0, w // 4)
    x2 = np.random.randint(3 * w // 4, w)
    y1 = np.random.randint(0, h // 4)
    y2 = np.random.randint(3 * h // 4, h)
    poly = np.array([[
        (x1, y1),
        (int(0.5 * (x1 + x2)), max(0, y1 - np.random.randint(0, h//8))),
        (x2, y1 + np.random.randint(0, h//8)),
        (x2, y2),
        (int(0.5 * (x1 + x2)), y2 + np.random.randint(0, h//8)),
        (x1, y2)
    ]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, poly, 255)
    # blur mask to make soft shadow
    k = max(3, min(h//10 + 1, 31))
    mask = cv2.GaussianBlur(mask, (k | 1, k | 1), 0)
    mask_f = (mask.astype(np.float32) / 255.0) * shadow_intensity
    out = img.astype(np.float32) * (1.0 - mask_f[:, :, None])
    return _to_uint8(out)

# e32
def plasma_shadow(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """
    Generate a 'plasma' style soft shadow using blurred noise, then darken.
    strength: 0.1..0.4 small
    """
    h, w = img.shape[:2]
    noise = np.random.randn(h, w).astype(np.float32)
    # multi-scale blur to get smooth plasma
    for k in (3,7,13):
        noise = cv2.GaussianBlur(noise, (k | 1, k | 1), 0)
    # normalize to 0..1
    nmin, nmax = noise.min(), noise.max()
    noise = (noise - nmin) / (nmax - nmin + 1e-8)
    # threshold/scale to shadow mask (take dark areas)
    mask = (noise < 0.5).astype(np.float32) * (0.5 - noise)
    mask = cv2.GaussianBlur(mask, (31,31), 0)
    mask = np.clip(mask, 0.0, 1.0) * strength
    out = img.astype(np.float32) * (1.0 - mask[:, :, None])
    return _to_uint8(out)

# e33
def ringing_overshoot(img: np.ndarray, amount: float = 0.5, lightness: float = 0.2) -> np.ndarray:
    """
    Simulate slight ringing/overshoot by applying a strong-ish Sharpen then blending back.
    amount: how strong the sharpen is (0.2..1.0). Keep moderate.
    """
    aug = A.Sharpen(alpha=(0.2, amount), lightness=(max(0.5, lightness - 0.2), lightness), p=1.0)
    sharp = aug(image=img)['image'].astype(np.float32)
    out = cv2.addWeighted(img.astype(np.float32), 1.0 - 0.35, sharp, 0.35, 0.0)  # blend small amount
    return _to_uint8(out)


# e34: 修正 ImageCompression
def image_compression(img: np.ndarray, quality_lower: int = 60, quality_upper: int = 80) -> np.ndarray:
    # 最新版使用 quality_range 代替 quality_lower/upper
    aug = A.ImageCompression(quality_range=(quality_lower, quality_upper), p=1.0)
    return aug(image=img)['image']

# e35: 模拟镜头色散（边缘彩虹纹）
# e35: 修正 ChromaticAberration (色差/色散)
def chromatic_aberration(img: np.ndarray, primary_distortion: float = 0.01) -> np.ndarray:
    """
    根据源码定义：
    primary_distortion_limit: 控制图像中心的畸变。
    secondary_distortion_limit: 控制图像边缘的畸变。
    """
    # 按照源码 Example 的方式传入参数
    aug = A.ChromaticAberration(
        primary_distortion_limit=primary_distortion,   # 传入 float 自动变为 (-val, val)
        secondary_distortion_limit=0.05,               # 默认值
        mode='green_purple',
        p=1.0
    )
    return aug(image=img)['image']

# e36: 下采样再上采样（丢失细节）
def downscale(img: np.ndarray, scale_min: float = 0.5, scale_max: float = 0.7) -> np.ndarray:
    # 最新版使用 scale_range 代替 scale_min/max
    aug = A.Downscale(scale_range=(scale_min, scale_max), p=1.0)
    return aug(image=img)['image']

# e37: 高级模糊（非均匀模糊）
def advanced_blur(img: np.ndarray, blur_limit: int = 3) -> np.ndarray:
    aug = A.AdvancedBlur(blur_limit=blur_limit, p=1.0)
    return aug(image=img)['image']

# e38: 曝光过度效果 (修正版)
def solarize(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # 注意：最新版参数名是 threshold_range
    aug = A.Solarize(threshold_range=(threshold, threshold), p=1.0)
    return aug(image=img)['image']

# # e39: 颜色通道打乱
# def channel_shuffle(img: np.ndarray) -> np.ndarray:
#     aug = A.ChannelShuffle(p=1.0)
#     return aug(image=img)['image']

# e40: 色相/饱和度/亮度扰动
def hsv_perturb(img: np.ndarray, h_limit: int = 20, s_limit: int = 30, v_limit: int = 20) -> np.ndarray:
    aug = A.HueSaturationValue(hue_shift_limit=h_limit, sat_shift_limit=s_limit, val_shift_limit=v_limit, p=1.0)
    return aug(image=img)['image']

# e41: 粗粒度区域丢弃 (Albumentations 新版字段修正)
def coarse_dropout(img: np.ndarray, max_holes: int = 1, fill_value: int = 0) -> np.ndarray:
    # 最新版将 max_holes 等移除了，改为用 range 控制大小和数量
    # 针对 CIFAR (32x32)，设定孔洞在 4到8 像素之间
    aug = A.CoarseDropout(
        num_holes_range=(max_holes, max_holes), 
        hole_height_range=(4, 8), 
        hole_width_range=(4, 8), 
        fill=fill_value, 
        p=1.0
    )
    return aug(image=img)['image']

# e42: 网格畸变
def grid_distortion(img: np.ndarray, dist_limit: float = 0.2) -> np.ndarray:
    aug = A.GridDistortion(distort_limit=dist_limit, p=1.0)
    return aug(image=img)['image']


# ---------------------------
# DA registry
# Each entry: name -> (callable, default_param_dict)
# callable signature: callable(img: np.ndarray, **params) -> np.ndarray
# 将所有图像增强方法及其默认参数注册到一个字典中。
# 用途：便于后续通过名称调用这些方法，并支持遗传算法等优化过程
# ---------------------------

DA_REGISTRY: Dict[str, Tuple[Callable[..., np.ndarray], Dict[str, Any]]] = {
    # e1: Gamma Correction
    "gamma": (
        gamma_correction,
        {"gamma": 1.0},  # 略微提亮
    ),

    # e2: Sharpen
    "sharpen": (
        sharpen,
        {"alpha": 0.5, "lightness": 1.0},  # 轻微锐化
    ),

    # e3: Change Brightness Contrast
    "brightness_contrast": (
        change_brightness_contrast,
        {"brightness": 0.2, "contrast": 0.2},  # 提亮一点对比度
    ),

    # e4: Unsharp Mask
    "unsharp": (
        unsharp_mask,
        {"radius": 5, "amount": 1.2},  # 小范围增强
    ),

    # e5: Wiener Filter
    "wiener_filter": (
        wiener_filter,
        {"d": 5},  # 双边滤波近似去噪
    ),

    # e6: Histogram Equalization
    "hist_eq": (
        hist_eq,
        {},  # 无参数
    ),

    # e7: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    "clahe": (
        clahe,
        {"clip_limit": 3.0, "tile_grid_size": (8, 8)},  # 默认轻度增强
    ),

    # e8: Gaussian Blur
    "gaussian_blur": (
        gaussian_blur,
        {"ksize": 5},  # 轻微模糊
    ),

    # e9: Median Blur
    "median_blur": (
        median_blur,
        {"ksize": 5},  # 小核去噪
    ),

    # e10: Pixel Loss (Dropout)
    "pixel_loss": (
        pixel_dropout,
          {}, # 轻微像素丢失
    ),

    # e11: Rayleigh Noise
    "rayleigh_noise": (
        rayleigh_noise,
        {"scale": 10.0},  # 轻度噪声
    ),

    # e12: Multiplicative Noise
    "mult_noise": (
        multiplicative_noise,
        {"sigma": 0.08},  # 轻微乘性噪声
    ),

    # e13: Gaussian Noise
    "gauss_noise": (
        gaussian_noise,
        {"sigma": 1.0},  # 轻微高斯噪声
    ),

    # e14: Bit-depth Reduction (BdR)
    "bit_reduce": (
        bit_depth_reduction,
        {"bits": 5},  # 降低少量比特深度
    ),

    #e15
    "gaussian_blur_channel":
        (gaussian_blur_channel,
         {"sigma": 1.0}
         ),

    # e16
    "brightness_contrast_channel":
        (brightness_contrast_channel,
         {"alpha": 1.4, "beta":-0.3}
         ),

    # e17
    "kernel_smooth_channel":
        (kernel_smooth_channel,
         {"delta": 5}
         ),

    # e18 轻微高斯噪声
    "AdditiveNoise": (additive_noise, {"var": 5.0}),
    # e19 Shot/Poisson 噪声
    "ShotNoise": (shot_noise, {"prob": 0.02}),
    # e20 ISO 噪声（摄像头风格）
    "ISONoise": (iso_noise, {"intensity": 0.2}),
    # e21 轻微运动模糊
    "MotionBlur": (motion_blur, {"ksize": 7}),
    # e22 轻微玻璃模糊
    "GlassBlur": (glass_blur, {"sigma": 0.5, "max_delta": 3, "iterations": 2}),
    # e23 小幅失焦模糊
    "Defocus": (defocus, {"blur_limit": 3}),
    # e24 轻微缩放模糊（radial）
    "ZoomBlur": (zoom_blur, {"strength": 0.03, "steps": 6}),
    # e25 小角度旋转
    "Rotate": (rotate_small, {"angle": 7.0}),
    # e26 小比例随机缩放
    "RandomScale": (random_scale, {"scale_limit": 0.2}),
    # e27 透视畸变（小）
    "Perspective": (perspective_small, {"scale": 0.1}),
    # e28 光学畸变（小）
    "OpticalDistortion": (optical_distortion_small, {"distort_limit": 0.05, "shift_limit": 0.005}),
    # e29 从边界裁切少量像素并补回（模拟裁切偏移）
    "RandomCropFromBorders": (random_crop_from_borders, {"max_border": 3}),
    # e30 超像素小替换
    # "Superpixels": (superpixels_small, {"p_replace": 0.1, "n_segments": 100}),
    # e31 随机单片阴影
    "RandomShadow": (random_shadow, {"shadow_intensity": 0.35}),
    # e32 等离子/模糊阴影
    "PlasmaShadow": (plasma_shadow, {"strength": 0.25}),
    # e33 轻微的锐化/振铃过冲效果
    "RingingOvershoot": (ringing_overshoot, {"amount": 0.5, "lightness": 0.2}),
    
    # e34: 模拟 JPEG 压缩损伤 (对破坏像素级特征、保留语义极度有效)
    "image_compression": (image_compression, {"quality_lower": 60, "quality_upper": 80}),

    # e35: 模拟镜头色散 (造成颜色通道微小位移)
    "chromatic_aberration": (chromatic_aberration, {"primary_distortion": 0.01}),

    # e36: 下采样 (丢失高频信号的利器)
    "downscale": (downscale, {"scale_min": 0.5, "scale_max": 0.7}),

    # e37: 高级模糊 (模拟更复杂的退化)
    "advanced_blur": (advanced_blur, {"blur_limit": 3}),

    # e38: 曝光过度 (剧烈改变亮度拓扑)
    "solarize": (solarize, {"threshold": 0.5}),

    # e39: 通道打乱 (如果你的触发器不是强依赖特定颜色，此项持久性极强)
    # "channel_shuffle": (channel_shuffle, {}),

    # e40: HSV 色彩抖动
    "hsv_perturb": (hsv_perturb, {"h_limit": 20, "s_limit": 30, "v_limit": 20}),

    # e41: 区域遮挡 (强制模型根据局部剩余特征激活)
    "coarse_dropout": (coarse_dropout, {"max_holes": 1, "fill_value": 0}),

    # e42: 网格畸变 (改变触发器几何结构)
    "grid_distortion": (grid_distortion, {"dist_limit": 0.2}),


}

def list_da_methods() -> List[str]:
    """Return registered DA method names."""
    return list(DA_REGISTRY.keys())


# def apply_da_single_channel(img_channel: np.ndarray, method_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
#     """
#     Apply a single DA method to a single-channel image (HxW).
#     """
#     if params is None:
#         params = {}
#     if method_name not in DA_REGISTRY:
#         raise KeyError(f"Unknown DA method {method_name}")
#     func, default_params = DA_REGISTRY[method_name]
#     # merge params
#     call_params = dict(default_params)
#     call_params.update(params)
#     return func(img_channel, **call_params)

def apply_da_single_channel_safe(ch: np.ndarray, method: str, params=None):
    """
    安全单通道增强：
      - 自动转换为3通道RGB（对要求RGB的增强）
      - 自动转回单通道
      - 确保uint8输出
    """
    if params is None:
        params = {}

    try:
        func, default_args = DA_REGISTRY[method]
        call_params = {**default_args, **params}

        img = np.clip(ch, 0, 255).astype(np.uint8)

        # 某些增强要求RGB输入 → 转成伪RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        out = func(img_rgb, **call_params)

        # 若输出是RGB → 转回灰度
        if out.ndim == 3 and out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    except Exception as e:
        print(f"[WARN] apply_da_single_channel failed on {method}: {e}")
        return ch

def apply_da_rgb(img_rgb: np.ndarray, combo: Tuple[Tuple[str, Optional[Dict[str, Any]]],
                                                    Tuple[str, Optional[Dict[str, Any]]],
                                                    Tuple[str, Optional[Dict[str, Any]]]]) -> np.ndarray:
    """
    Apply three DA operations separately to R, G, B channels respectively.

    combo: ((method_R, params_R), (method_G, params_G), (method_B, params_B))
    Each method is a string key in DA_REGISTRY; params may be None or a dict.
    Returns HxWx3 uint8 numpy array.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("apply_da_rgb expects an RGB image HxWx3")
    r, g, b = cv2.split(img_rgb)
    (mR, pR), (mG, pG), (mB, pB) = combo
    r2 = apply_da_single_channel_safe(r, mR, pR)
    g2 = apply_da_single_channel_safe(g, mG, pG)
    b2 = apply_da_single_channel_safe(b, mB, pB)
    out = cv2.merge([r2, g2, b2])

    # --- 保底修复阶段 ---
    # 确保输出为合法RGB图像
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    if out.ndim == 2:
        # 单通道图像 -> 转3通道
        out = np.repeat(out[:, :, None], 3, axis=2)
    elif out.ndim == 3 and out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    elif out.ndim == 3 and out.shape[2] > 3:
        # 通道超出3个时截断
        out = out[:, :, :3]
    # 修正尺寸异常
    h, w = img_rgb.shape[:2]
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    # 转换为 uint8 范围 [0,255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ---------------------------
# Trigger application
# Supports:
# - full image replacement with applied DA
# - patch injection (at bottom-right or random pos)
# - mask-based blending if mask provided
# ---------------------------

#？这个是如何应用的？为何有这一步？
def apply_trigger(img: np.ndarray,
                  combo: Tuple[Tuple[str, Optional[Dict[str, Any]]],
                               Tuple[str, Optional[Dict[str, Any]]],
                               Tuple[str, Optional[Dict[str, Any]]]],
                  mode: str = "global",
                  patch_size: Optional[int] = None,
                  location: str = "bottom_right",
                  mask: Optional[np.ndarray] = None,
                  alpha: float = 1.0,
                  random_state: Optional[int] = None) -> np.ndarray:
    """
    Apply trigger (DA combo) to the image.

    mode:
     - "global": apply DA to entire image (default)
     - "patch": apply DA to a square patch of size patch_size (int)
     - "mask": use provided mask (same HxW, float in [0,1]) to blend DA-applied image on top of original

    location (for patch mode): "bottom_right", "top_left", "center", or "random"
    alpha: blending factor (for mask mode; alpha of trigger image)
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    img = img.copy()
    H, W = img.shape[:2]
    trigger_img = apply_da_rgb(img, combo)

    if mode == "global":
        return trigger_img

    elif mode == "patch":
        if patch_size is None:
            patch_size = int(min(H, W) * 0.2)  # default 20% of min dim
        ph = pw = patch_size
        if location == "bottom_right":
            y0 = H - ph
            x0 = W - pw
        elif location == "top_left":
            y0 = 0
            x0 = 0
        elif location == "center":
            y0 = (H - ph) // 2
            x0 = (W - pw) // 2
        elif location == "random":
            y0 = np.random.randint(0, max(1, H - ph + 1))
            x0 = np.random.randint(0, max(1, W - pw + 1))
        else:
            raise ValueError("unknown location")
        out = img.copy()
        out[y0:y0+ph, x0:x0+pw, :] = trigger_img[y0:y0+ph, x0:x0+pw, :]
        return out

    elif mode == "mask":
        if mask is None:
            raise ValueError("mask must be provided for 'mask' mode")
        if mask.shape[:2] != (H, W):
            raise ValueError("mask shape mismatch")
        # ensure mask float in [0,1]
        m = mask.astype(np.float32)
        if m.max() > 1.1:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        out = (1.0 - m[..., None] * alpha) * img.astype(np.float32) + (m[..., None] * alpha) * trigger_img.astype(np.float32)
        return _clip_uint8(out)

    else:
        raise ValueError("unknown mode")



# ---------------------------
# Similarity metrics
# ---------------------------

def _img_to_float(img: np.ndarray) -> np.ndarray:
    return img_as_float32(img)

def pil_to_numpy_rgb(pil_img):
    arr = np.array(pil_img)  # PIL gives RGB
    return arr

def ssim_batch(originals: Sequence[np.ndarray], poisoned: Sequence[np.ndarray], multichannel: bool = True) -> float:
    """
    Compute average SSIM over a batch of image pairs.
    Each input is a sequence/list of numpy arrays (HxWxC). Returns mean SSIM in [0,1].
    """
    if len(originals) != len(poisoned):
        raise ValueError("length mismatch")
    scores = []
    for a, b in zip(originals, poisoned):
        # convert to float in [0,1]
        af = _img_to_float(a)
        bf = _img_to_float(b)
        try:
            s = ssim(af, bf, multichannel=multichannel, data_range=1.0)
        except Exception:
            # fallback: convert to grayscale and compute
            s = ssim(_to_gray_if_needed((af*255).astype(np.uint8)), _to_gray_if_needed((bf*255).astype(np.uint8)), data_range=255)
        scores.append(float(s))
    return float(np.mean(scores))

def _prewitt_filters():
    hx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32) / 3.0
    hy = hx.T
    return hx, hy

def gmsd(ref_img: np.ndarray, dist_img: np.ndarray, c: float = 170.0):
    """
    Compute GMSD between reference and distorted images.
    ref_img, dist_img: HxW or HxWxC numpy arrays in uint8 (0-255) or float (0-1).
    Returns scalar GMSD score.
    """
    # convert to grayscale float
    def to_gray_float(im):
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # convert to float in range [0,255]
        if im.dtype == np.uint8:
            f = im.astype(np.float32)
        else:
            f = img_as_ubyte(im).astype(np.float32)
        return f

    r = to_gray_float(ref_img)
    d = to_gray_float(dist_img)
    hx, hy = _prewitt_filters()
    # gradient magnitude
    rx = cv2.filter2D(r, -1, hx)
    ry = cv2.filter2D(r, -1, hy)
    dx = cv2.filter2D(d, -1, hx)
    dy = cv2.filter2D(d, -1, hy)
    mr = np.sqrt(rx ** 2 + ry ** 2)
    md = np.sqrt(dx ** 2 + dy ** 2)
    # GMS map
    num = 2 * mr * md + c
    den = mr ** 2 + md ** 2 + c
    # avoid division by zero
    gms = num / den
    # compute std dev of gms map as GMSD
    gmsm = np.mean(gms)
    gmsd_score = np.sqrt(np.mean((gms - gmsm) ** 2))
    return float(gmsd_score)

def gmsd_map_single(img_ref: np.ndarray, img_dist: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute a local gradient magnitude similarity map (approximation).
    Returns similarity map in [0,1] roughly.
    Implementation:
      - compute gradient magnitude (Sobel) for both images (grayscale)
      - sim_map = 1 - abs(gm_ref - gm_dist) / (gm_ref + gm_dist + eps)
    """
    ref_gray = _to_gray_if_needed(img_ref).astype(np.float32)
    dist_gray = _to_gray_if_needed(img_dist).astype(np.float32)
    # Sobel gradients
    gx_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    gm_ref = np.sqrt(gx_ref**2 + gy_ref**2)

    gx_dist = cv2.Sobel(dist_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_dist = cv2.Sobel(dist_gray, cv2.CV_32F, 0, 1, ksize=3)
    gm_dist = np.sqrt(gx_dist**2 + gy_dist**2)

    sim_map = 1.0 - np.abs(gm_ref - gm_dist) / (gm_ref + gm_dist + eps)
    # clip to [0,1]
    sim_map = np.clip(sim_map, 0.0, 1.0)
    return sim_map


def gmsd_batch(originals: Sequence[np.ndarray], poisoned: Sequence[np.ndarray]) -> float:
    """
    Approximate GMSD: compute similarity map for each pair and return standard deviation of the similarity map means.
    The original GMSD metric is a single scalar; here we compute a usable surrogate:
      For each pair, compute mean(sim_map). Return std across images (lower is better -> convert to similarity score).
    To keep semantics similar to SSIM (higher better), we will return 1 / (1 + gmsd) where gmsd is computed std(mean_sim).
    """
    if len(originals) != len(poisoned):
        raise ValueError("length mismatch")
    means = []
    for a, b in zip(originals, poisoned):
        sm = gmsd_map_single(a, b)
        means.append(float(np.mean(sm)))
    means = np.array(means, dtype=np.float32)
    gmsd_val = float(np.std(means))
    # convert to similarity-like number in (0,1], higher means more similar
    sim = 1.0 / (1.0 + gmsd_val)
    return sim


# ---------------------------
# Utilities for DA combos handling (for GA)
# ---------------------------

def combo_to_json(combo: Tuple[Tuple[str, Optional[Dict[str, Any]]],
                              Tuple[str, Optional[Dict[str, Any]]],
                              Tuple[str, Optional[Dict[str, Any]]]]) -> str:
    """Serialize combo to JSON string.将图像增强组合序列化为 JSON 字符串"""
    return json.dumps([
        [combo[0][0], combo[0][1] or {}],
        [combo[1][0], combo[1][1] or {}],
        [combo[2][0], combo[2][1] or {}],
    ])


def combo_from_json(s: str):
    """Deserialize combo JSON (list of [name, params]) to combo tuple.从 JSON 字符串反序列化图像增强组合。"""
    arr = json.loads(s)
    return ((arr[0][0], arr[0][1] or {}),
            (arr[1][0], arr[1][1] or {}),
            (arr[2][0], arr[2][1] or {}))


# def random_combo(seed: Optional[int] = None) -> Tuple[Tuple[str, Dict[str, Any]],
#                                                      Tuple[str, Dict[str, Any]],
#                                                      Tuple[str, Dict[str, Any]]]:
#     """
#     Sample a random combo from DA_REGISTRY using default params.
#     随机生成一个图像增强组合
#     """
#     if seed is not None:
#         random.seed(seed)
#         np.random.seed(seed)
#     keys = list(DA_REGISTRY.keys())
#     choice = random.choices(keys, k=3)
#     combo = []
#     for k in choice:
#         func, params = DA_REGISTRY[k]
#         combo.append((k, dict(params)))  # copy defaults
#     return (combo[0], combo[1], combo[2])
def random_combo_no_repeat(seed: Optional[int] = None, length: int = 3) -> Tuple[Tuple[str, Dict[str, Any]],
                                                                               Tuple[str, Dict[str, Any]],
                                                                               Tuple[str, Dict[str, Any]]]:
    """
    Sample a random combo from DA_REGISTRY using default params, without repeating methods.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    keys = list(DA_REGISTRY.keys())
    choice = random.sample(keys, k=min(length, len(keys)))  # sample without replacement
    combo = []
    for k in choice:
        func, params = DA_REGISTRY[k]
        combo.append((k, dict(params)))  # copy defaults
    # pad with None if less than length
    while len(combo) < length:
        combo.append((choice[-1], dict(DA_REGISTRY[choice[-1]][1])))
    return tuple(combo)


def _random_combo_no_repeat_with_rng(rng: random.Random, length: int = 3):
    """Generate a random combo using a local RNG (no global seeding side-effects)."""
    keys = list(DA_REGISTRY.keys())
    if not keys:
        raise RuntimeError("DA_REGISTRY is empty; cannot sample combo")
    length = int(length)
    if length <= 0:
        raise ValueError("length must be a positive integer")

    choice = rng.sample(keys, k=min(length, len(keys)))
    combo = [(k, dict(DA_REGISTRY[k][1])) for k in choice]
    while len(combo) < length:
        combo.append((choice[-1], dict(DA_REGISTRY[choice[-1]][1])))
    return tuple(combo)


def generate_random_combos(n: int = 5,
                           seed: Optional[int] = None,
                           length: int = 3,
                           ensure_unique: bool = True,
                           max_tries: int = 10000):
    """Generate multiple random DA combos for comparison experiments.

    Args:
        n: number of combos.
        seed: fixed seed for reproducibility (uses a local RNG).
        length: combo length (default 3 for R/G/B).
        ensure_unique: ensure combos are not duplicated (by JSON signature).
        max_tries: safety guard to avoid infinite loops when unique space is small.

    Returns:
        List[combo] where each combo is a tuple of (method, params) pairs.
    """
    n = int(n)
    if n <= 0:
        return []
    rng = random.Random(seed)

    combos = []
    seen = set()
    tries = 0
    while len(combos) < n:
        tries += 1
        if tries > int(max_tries):
            raise RuntimeError(
                f"Unable to generate {n} unique combos after {max_tries} tries; "
                f"try ensure_unique=False or reduce n/length."
            )
        combo = _random_combo_no_repeat_with_rng(rng, length=length)
        if ensure_unique:
            sig = combo_to_json(combo)
            if sig in seen:
                continue
            seen.add(sig)
        combos.append(combo)
    return combos



# ---------------------------
# Small demo / self-test utilities
# ---------------------------

def visualize_pair(original: np.ndarray, poisoned: np.ndarray, out_path: Optional[str] = None):
    """
    Stack original and poisoned horizontally for quick visualization.
    将原始图像和处理后的图像水平堆叠，便于可视化比较
    - 将两个图像堆叠到一个画布上。
    - 如果指定输出路径，将结果保存为图像文件。
    - 返回堆叠后的图像
    """
    if original.dtype != np.uint8:
        orig = _clip_uint8(original)
    else:
        orig = original
    if poisoned.dtype != np.uint8:
        pois = _clip_uint8(poisoned)
    else:
        pois = poisoned
    H = max(orig.shape[0], pois.shape[0])
    W = orig.shape[1] + pois.shape[1]
    if orig.ndim == 2:
        canvas = np.zeros((H, W), dtype=np.uint8)
        canvas[:orig.shape[0], :orig.shape[1]] = orig
        canvas[:pois.shape[0], orig.shape[1]:] = pois
        img = Image.fromarray(canvas)
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:orig.shape[0], :orig.shape[1], :] = orig
        canvas[:pois.shape[0], orig.shape[1]:, :] = pois
        img = Image.fromarray(canvas)
    if out_path:
        save_image(out_path, np.array(img))
    return img


# ---------------------------
# End of file
# ---------------------------



