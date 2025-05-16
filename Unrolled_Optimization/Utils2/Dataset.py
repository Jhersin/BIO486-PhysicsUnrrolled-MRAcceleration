import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Unrolled_Dataset(Dataset):
    def __init__(self, unroll_root_dir, transform=None):
        self.unroll_root_dir = unroll_root_dir
        self.transform = transform

        # Define paths
        zero_fill_dir        = os.path.join(unroll_root_dir, '*', 'zerofill', '*.npy')
        coil_sensitivity_dir = os.path.join(unroll_root_dir, '*', 'coil_sensitivity', '*.npy')
        undersample_dir      = os.path.join(unroll_root_dir, '*', 'undersample', '*.npy')
        ground_truth_Cs_dir  = os.path.join(unroll_root_dir, '*', 'ground_truth_Cs', '*.npy')

        # Gather files
        zero_fill_files        = sorted(glob.glob(zero_fill_dir))
        coil_sensitivity_files = sorted(glob.glob(coil_sensitivity_dir))
        undersample_files      = sorted(glob.glob(undersample_dir))
        ground_truth_Cs_files  = sorted(glob.glob(ground_truth_Cs_dir))

        print("Found files:")
        print("  Zero-filled:        ", len(zero_fill_files))
        print("  Coil sensitivity:   ", len(coil_sensitivity_files))
        print("  Undersampled:       ", len(undersample_files))
        print("  Ground truth (Cs):  ", len(ground_truth_Cs_files))

        # Safety check
        assert len(zero_fill_files) == len(coil_sensitivity_files) == len(undersample_files) == len(ground_truth_Cs_files), \
            "Mismatch in number of files across modalities"

        # Pair files together
        self.paired_files = list(zip(zero_fill_files, coil_sensitivity_files, undersample_files, ground_truth_Cs_files))

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        zf_path, cs_path, us_path, gt_path = self.paired_files[idx]

        # Load .npy data
        zf = np.load(zf_path)
        cs = np.load(cs_path)
        us = np.load(us_path)
        gt = np.load(gt_path)

         # Convert to tensors
        zf_tensor = torch.from_numpy(zf).unsqueeze(0)      # (1, 320, 320)
        cs_tensor = torch.from_numpy(cs)                   # (15, 320, 320)
        us_tensor = torch.from_numpy(us)                   # (15, 644, 372)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)      # (1, 320, 320

        # Optionally apply transform (e.g., cropping or augmentation)
        if self.transform:
            zf_tensor = self.extract_central_patch(zf_tensor)
            cs_tensor = self.extract_central_patch(cs_tensor)
            gt_tensor = self.extract_central_patch(gt_tensor)

        return zf_tensor, us_tensor, cs_tensor, gt_tensor

    @staticmethod
    def extract_central_patch(x: torch.Tensor, patch_size: int = 320) -> torch.Tensor:
        """Crop middle part (320,320)"""
        H = x.shape[-2]
        W = x.shape[-1]
        start_x = (H - patch_size) // 2
        start_y = (W - patch_size) // 2
        return x[..., start_x:start_x + patch_size, start_y:start_y + patch_size]