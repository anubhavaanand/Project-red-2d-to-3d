import numpy as np

from ..model_weights import asset, ensure

_lpips_vgg = None

# EVAL-06: a perfect prediction has infinite PSNR. Returning float('inf')
# poisons every plain sum()/len() mean downstream, and 100.0 was silently
# treated as a real score. Cap at a documented ceiling instead and record it.
PSNR_CAP_DB = 100.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes PSNR (dB) between two RGB images in [0, 255].

    A perfect match returns ``PSNR_CAP_DB`` (100.0) rather than infinity: the
    cap is documented here and in the README so it can be excluded or handled
    explicitly when aggregating.
    """
    a = np.asarray(img1, dtype=np.float32)[..., :3]
    b = np.asarray(img2, dtype=np.float32)[..., :3]
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return PSNR_CAP_DB
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _to_lpips_tensor(img):
    import torch

    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t
    arr = np.asarray(img, dtype=np.float32)[..., :3]
    if arr.max() > 1.05:
        arr = arr / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def compute_lpips(img1, img2) -> float:
    """Computes VGG LPIPS perceptual distance between two RGB images."""
    global _lpips_vgg
    import lpips
    import torch

    if _lpips_vgg is None:
        # SEC-12: verify the VGG16 weights against a pinned SHA-256 before lpips
        # loads them, so a substituted CDN cannot silently change the metric.
        ensure(asset("vgg16-397923af.pth"))
        _lpips_vgg = lpips.LPIPS(net="vgg").eval()
    t1 = _to_lpips_tensor(img1)
    t2 = _to_lpips_tensor(img2)
    with torch.no_grad():
        dist = _lpips_vgg(t1, t2)
    return float(dist.item())
