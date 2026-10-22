import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMorphPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, metric='SMO'):
        super(AdaptiveMorphPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.metric = metric

    def forward(self, x):
        # x shape: [B, 3, H, W] -> Se procesa sobre 3 canales RGB
        B, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride

        x_unfold = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw))
        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1
        
        x_unfold = x_unfold.view(B, C, kh * kw, H_out, W_out)
        x_win = x_unfold.permute(0, 1, 3, 4, 2)  # [B, C, H_out, W_out, K*K]

        # 1. Descriptor estadístico (SMO o Varianza)
        var = torch.var(x_win, dim=-1, keepdim=True)
        if self.metric == 'SMO':
            phi = 1.0 - (1.0 / (1.0 + var))
        else:
            phi = var

        # 2. Vector de pesos psi con manejo de regiones planas (varianza 0 -> 1/3)
        sum_phi = torch.sum(phi, dim=1, keepdim=True)
        psi = torch.where(sum_phi > 1e-8, phi / (sum_phi + 1e-8), torch.full_like(phi, 1.0 / C))

        # 3. Puntuación y selección del píxel ganador intacto
        score = torch.sum(x_win * psi, dim=1)  # [B, H_out, W_out, K*K]
        max_idx = torch.argmax(score, dim=-1, keepdim=True)
        max_idx_expanded = max_idx.unsqueeze(1).expand(B, C, H_out, W_out, 1)
        
        out = torch.gather(x_win, dim=-1, index=max_idx_expanded).squeeze(-1)
        return out  # [B, 3, H_out, W_out]