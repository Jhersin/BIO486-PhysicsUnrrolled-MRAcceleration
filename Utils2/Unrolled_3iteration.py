import torch.fft
import torch

class Physics(torch.nn.Module):
    def __init__(self, alpha=0.1, sampler=None):
        super().__init__()
        self.alpha = alpha
        self.sampler = sampler  # SamplingFunction instance

    def forward(self, x):
        """Optional: Implement if Physics is called directly."""
        return x

    def fourier_transform(self, img, dim=(-2, -1)):
        """Image → k-space (orthonormal FFT)."""
        img = torch.fft.ifftshift(img, dim=dim)
        k = torch.fft.fftn(img, dim=dim, norm='ortho')
        return torch.fft.fftshift(k, dim=dim)

    def inverse_fourier_transform(self, k, dim=(-2, -1)):
        """k-space → image (orthonormal FFT)."""
        k = torch.fft.ifftshift(k, dim=dim)
        img = torch.fft.ifftn(k, dim=dim, norm='ortho')
        return torch.fft.fftshift(img, dim=dim)

    @staticmethod
    def extract_central_patch(x: torch.Tensor, patch_size: int = 320) -> torch.Tensor:
        """Crop middle part (320,320)"""
        H = x.shape[-2]
        W = x.shape[-1]
        start_x = (H - patch_size) // 2
        start_y = (W - patch_size) // 2
        return x[..., start_x:start_x + patch_size, start_y:start_y + patch_size]

    def _compute_S(self, C_k, Cs):
        """
        Compute S = C_k - alpha * E^H(E(C_k))
        """
        Cs_conj = torch.conj(Cs)

        # E(C_k) = Masked FFT(C_k * Cs)
        C_x = C_k * Cs  # [B, 15, H, W]
        #print(f'C_x.shape {C_x.shape}')
        CF_x = self.fourier_transform(C_x)  # [B, 15, H, W]
        #print(f'CF_x.shape {CF_x.shape}')

        # E^H(E_x) = IFFT(zero-filled E_x) * Csᴴ
        Z_Ex = self.sampler.apply_mask(CF_x)  # [B, 15, H, W]
        #print(f'Z_Ex.shape {Z_Ex.shape}')
        IFZ_Ex = self.inverse_fourier_transform(Z_Ex)  # [B, 15, H, W]
        #print(f'IFZ_Ex.shape {IFZ_Ex.shape}')
        Eh_Ex = torch.sum(IFZ_Ex * Cs_conj , dim = -3, keepdim = True) # [B, 1, H, W]
        #print(f'Eh_Ex.shape {Eh_Ex.shape}')

        return C_k - self.alpha * Eh_Ex

    def _compute_W_e(self, undersampled_ksp, Cs):
        """
        Compute W_e = alpha * E^H(undersampled_ksp)
        """
        Cs_conj = torch.conj(Cs)

        Z_x = self.sampler.zero_fill(undersampled_ksp)  # [B, 15, H, W]
        #print(f'Zx_we {Z_x.shape}')
        IFZ_x = self.inverse_fourier_transform(Z_x)  # [B, 15, H, W]
        #print(f'IFZ_x {IFZ_x.shape}')
        CIFZ_X = self.extract_central_patch(IFZ_x)
        #print(f'CIFZ_X {CIFZ_X.shape}')
        E_hx = torch.sum(CIFZ_X * Cs_conj , dim = -3, keepdim = True) # [B, 1, H, W]
        #print(f'E_hx {E_hx.shape}')

        return self.alpha * E_hx

    def _final_sum(self, S, W_e):
        """
        Combine S and W_e
        """
        combined = S + W_e
        combined_final = torch.sum(combined, dim=-3, keepdim = True)  # [B, H, W]

        return combined_final

import torch
import torch.nn as nn
import torch.fft

class SamplingFunction(nn.Module):
    def __init__(self, accel_factor=4, num_central_lines=30, mask_width=320, zero_fill_width=368, height=320):
        super().__init__()
        self.accel_factor = accel_factor
        self.num_central_lines = num_central_lines

        # Precompute separate ky_positions
        self.mask_width = mask_width
        self.zero_fill_width = zero_fill_width

        self.mask_ky_positions = self._compute_sampling_positions(self.mask_width)
        self.zero_fill_ky_positions = self._compute_sampling_positions(self.zero_fill_width)

        # Create and store fixed mask for apply_mask() (assume height = 320)
        self.fixed_mask = self._create_mask(height=height, width=self.mask_width)

    def _compute_sampling_positions(self, width):
        center = width // 2
        half_width = self.num_central_lines // 2

        central_lines = torch.arange(center - half_width,
                                     center + half_width + (self.num_central_lines % 2))

        all_lines = torch.arange(width)
        accel_lines = all_lines[::self.accel_factor]
        accel_lines = accel_lines[~torch.isin(accel_lines, central_lines)]

        ky_positions = torch.sort(torch.cat([central_lines, accel_lines]))[0]
        return ky_positions

    def apply_mask(self, kspace_data):
        """
        Apply fixed mask [320x320] for undersampling simulated FFT data.
        Input shape: [B, 15, 320, 320]
        """
        assert kspace_data.shape[-2:] == (320, self.mask_width), \
            f"Expected shape [..., 320, {self.mask_width}], got {kspace_data.shape[-2:]}"
        mask = self.fixed_mask.to(kspace_data.device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return kspace_data * mask

    def zero_fill(self, undersampled_ksp):
        """
        Zero-fill data using ky_positions computed from 368-width full k-space.
        """
        expected = self.zero_fill_ky_positions.shape[0]
        actual = undersampled_ksp.shape[-1]
        assert actual == expected, f"Expected {expected} ky lines, got {actual}"

        full_shape = list(undersampled_ksp.shape[:-1]) + [self.zero_fill_width]
        full_kspace = torch.zeros(full_shape, dtype=undersampled_ksp.dtype, device=undersampled_ksp.device)
        full_kspace[..., self.zero_fill_ky_positions] = undersampled_ksp
        return full_kspace

    def _create_mask(self, height, width):
        """
        Create a binary undersampling mask of shape [height, width].
        Based on mask_ky_positions.
        """
        mask = torch.zeros((height, width), dtype=torch.float32)
        mask[:, self.mask_ky_positions] = 1.0
        return mask

import torch.nn as nn

class UnrolledReconstructor(nn.Module):
    def __init__(self, model1, model2, model3, alpha=0.1):
        super().__init__()
        self.model1 = model1  # Refinement network 1: [B,1,H,W] -> [B,1,H,W]
        self.model2 = model2  # Refinement network 2: [B,1,H,W] -> [B,1,H,W]
        self.model3 = model3
        self.alpha = alpha

        # Initialize physics components
        self.sampler = SamplingFunction()
        self.physics = Physics(alpha=self.alpha, sampler=self.sampler)

    def forward(self, zerofill, undersampled_ksp, cs_maps):
        """Unrolled reconstruction with data consistency.

        Args:
            zerofill (torch.Tensor): [B,1,H,W]
            undersampled_ksp (torch.Tensor): [B,15,H,W]
            cs_maps (torch.Tensor): [B,15,H,W]

        Returns:
            torch.Tensor: Reconstructed image [B,1,H,W]
        """

        # Step 1: Physics
        W_e = self.physics._compute_W_e(undersampled_ksp, cs_maps)
        S = self.physics._compute_S(zerofill, cs_maps)
        input1 = self.physics._final_sum(S, W_e)

        #print(f"recon1 shape: {input1.shape}")

        # Step 2: Unrolled1 (DL + Physics)
        C_1 = self.model1(input1)
        S1 = self.physics._compute_S(C_1, cs_maps)
        input2 = self.physics._final_sum(S1, W_e)

        # print(f"C_1 shape: {C_1.shape}")
        # print(f"S1 shape: {S1.shape}")
        # print(f"recon2 shape: {input2.shape}")

        # Step 3: Unrolled2 (DL + Physics)
        C_2 = self.model2(input2)
        S2 = self.physics._compute_S(C_2, cs_maps)
        input3 = self.physics._final_sum(S2, W_e)

        # print(f"S2 shape: {S2.shape}")
        # print(f"C_2 shape: {C_2.shape}")
        # print(f"recon3 shape: {recon3.shape}")

        #Step 4: Unrolled3 (DL + Physics)
        C_3 = self.model3(input3)

        return C_3
