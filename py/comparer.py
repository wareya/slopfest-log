import numpy as np
from PIL import Image
import argparse
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Skimage metrics
from skimage.metrics import (
    peak_signal_noise_ratio as psnr_func,
    structural_similarity as ssim_func,
    normalized_mutual_information as nmi_func,
    variation_of_information as voi_func
)

def load_lpips():
    try:
        import torch
        import lpips
        return torch, lpips
    except ImportError as e:
        print(f"Error: LPIPS requires 'torch' and 'lpips' packages. ({e})")
        sys.exit(1)


def mae_func(gt, pred, data_range=1.0):
    gt = gt.astype(np.float32) / data_range
    pred = pred.astype(np.float32) / data_range
    return np.sum(np.abs(gt - pred)) / gt.size
    
def fft2_power_of_2(image):
    h, w = image.shape
    ret = np.fft.fft2(image, s=(h, w), norm="ortho")
    ret = np.fft.fftshift(ret)
    return ret


def run_comparison(gt_path, est_path, args):
    try:
        img_gt = Image.open(gt_path)
        img_est = Image.open(est_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if img_gt.size != img_est.size:
        print("Error: Dimension mismatch.")
        sys.exit(1)

    gt_arr = np.array(img_gt.convert('RGB')).astype(float)
    est_arr = np.array(img_est.convert('RGB')).astype(float)
    
    gt_gray = np.array(img_gt.convert('L')).astype(float)
    est_gray = np.array(img_est.convert('L')).astype(float)
    
    gt_gray1 = fft2_power_of_2(gt_gray).real
    est_gray1 = fft2_power_of_2(est_gray).real
    
    img = Image.fromarray(gt_gray1.astype(int)).convert('RGB')
    #img.save('gtest.png') # for debugging

    print(gt_gray1.shape)
    print(gt_gray1[5][4])
    
    # Calculate standard metrics
    psnr_val = psnr_func(gt_arr, est_arr, data_range=255)
    ssim_val = ssim_func(gt_gray, est_gray, data_range=255)
    
    dftmse_val = np.mean((gt_gray1/255.0 - est_gray1/255.0)**2)
    dftpsnr_val = 10.0 * np.log10(1.0 / (dftmse_val + 1e-30))
    
    nmi_val = nmi_func(gt_arr, est_arr)

    # VOI (Raw) & B-VOI (Binned)
    voi_total = sum(voi_func(gt_arr.astype(int), est_arr.astype(int)))
    gt_binned = (gt_arr // 16).astype(int)
    est_binned = (est_arr // 16).astype(int)
    bvoi_total = sum(voi_func(gt_binned, est_binned))

    mae_val = mae_func(gt_arr, est_arr, data_range=255)

    lpips_val = None
    if args.lpips:
        torch, lpips_lib = load_lpips()
        loss_fn = lpips_lib.LPIPS(net='alex')
        # LPIPS expects tensors in [-1, 1], shape (N, C, H, W)
        def to_lpips_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)
        lpips_val = loss_fn(to_lpips_tensor(gt_arr), to_lpips_tensor(est_arr)).item()
    
    # Output Formatting
    print("-" * 50)
    print(f"Metrics for: {est_path}")
    print(f"PSNR:     {psnr_val:7.2f}  (Better: Higher)")
    print(f"DFT PSNR: {dftpsnr_val:7.2f}  (Better: Higher)")
    print(f"SSIM:     {ssim_val:7.4f}  (Better: Higher)")
    print(f"NMI:      {nmi_val:7.4f}  (Better: Higher)")
    print(f"MAE:      {mae_val:7.4f}  (Better: Lower)")
    print(f"VOI:      {voi_total:7.4f}  (Better: Lower)")
    print(f"B-VOI:    {bvoi_total:7.4f}  (Better: Lower)")
    if lpips_val is not None:
        print(f"LPIPS:    {lpips_val:7.4f}  (Better: Lower)")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth")
    parser.add_argument("estimate")
    parser.add_argument("--lpips", action="store_true",
                        help="Compute LPIPS score (requires torch and lpips packages)")
    args = parser.parse_args()
    run_comparison(args.ground_truth, args.estimate, args)