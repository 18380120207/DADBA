from __future__ import annotations

import numpy as np


def pil_to_numpy_rgb(pil_img):
    """Convert a PIL image to uint8 RGB numpy array (H, W, 3)."""
    from PIL import Image

    if not isinstance(pil_img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(pil_img)!r}")

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.asarray(pil_img, dtype=np.uint8)


def tensor_to_numpy_rgb(tensor) -> np.ndarray:
    """Convert a torch Tensor image (C,H,W) in [0,1] to uint8 RGB (H,W,3)."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

    arr = tensor.detach().cpu().numpy()

    if arr.ndim == 2:
        gray = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {arr.shape}")

    if arr.shape[0] in (1, 3):
        if arr.shape[0] == 1:
            gray = np.clip(arr[0] * 255.0, 0, 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

        rgb = np.transpose(arr, (1, 2, 0))
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb

    if arr.shape[2] in (1, 3):
        if arr.shape[2] == 1:
            gray = np.clip(arr[..., 0] * 255.0, 0, 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)
        rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return rgb

    raise ValueError(f"Unrecognized image tensor shape {arr.shape} (expected CHW or HWC)")


def _rgb_to_gray_float01(rgb_u8: np.ndarray) -> np.ndarray:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected RGB uint8 image HxWx3, got shape {rgb_u8.shape}")

    rgb = rgb_u8.astype(np.float32) / 255.0
    return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    try:
        import cv2

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    except Exception:
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        def conv2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
            h, w = img.shape
            out = np.zeros((h, w), dtype=np.float32)
            pad = 1
            padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
            for i in range(h):
                for j in range(w):
                    patch = padded[i : i + 3, j : j + 3]
                    out[i, j] = float(np.sum(patch * k))
            return out

        gx = conv2(gray, kx)
        gy = conv2(gray, ky)
        return np.sqrt(gx * gx + gy * gy)


def gmsd(ref_u8: np.ndarray, dist_u8: np.ndarray, c: float = 0.0026) -> float:
    """Compute GMSD between two uint8 RGB images (lower is better)."""
    if ref_u8.shape != dist_u8.shape:
        raise ValueError(f"Shape mismatch: ref {ref_u8.shape} vs dist {dist_u8.shape}")

    g1 = _grad_mag(_rgb_to_gray_float01(ref_u8))
    g2 = _grad_mag(_rgb_to_gray_float01(dist_u8))

    num = 2.0 * g1 * g2 + c
    den = g1 * g1 + g2 * g2 + c
    gms_map = num / den

    return float(np.std(gms_map))
