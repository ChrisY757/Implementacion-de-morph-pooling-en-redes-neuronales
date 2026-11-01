import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMorphPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, metric='SMO', stat_window=None):
        super(AdaptiveMorphPooling2d, self).__init__()
        self.kernel_size = kernel_size     # Pooling siempre 2
        self.stride = stride               # Pooling siempre 2
        self.metric = metric
        self.stat_window = stat_window     # None/2, 3, 5, 7 o 'global'

    def forward(self, x):
        # x shape: [B, 3, H, W] -> Se procesa sobre 3 canales RGB
        B, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride

        # --- 1. Extraer ventanas del pooling 2x2 ---
        x_unfold = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw))
        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1
        
        x_unfold = x_unfold.view(B, C, kh * kw, H_out, W_out)
        x_win = x_unfold.permute(0, 1, 3, 4, 2)  # [B, C, H_out, W_out, 4]

        # --- 2. Determinar el tensor sobre el que se calcula el descriptor ---
        if self.stat_window == 'global':
            # Caso especial: Imagen completa
            # Calculamos la varianza/media/etc. global por canal
            x_stat = x.view(B, C, -1).unsqueeze(-1)  # [B, C, H*W, 1] -> Aplanamos la imagen
            # Nota: para que el descriptor funcione, calculamos sobre los H*W valores
            
        elif self.stat_window is None or self.stat_window == self.kernel_size:
            # Caso base: misma ventana que el pooling (2x2)
            x_stat = x_win                            # [B, C, H_out, W_out, 4]
            
        else:
            # Caso ventana grande: 3x3, 5x5, 7x7...
            sw_int = int(self.stat_window)
            total_pad = sw_int - kh                   # Padding total para mantener H_out x W_out
            pad_tl = total_pad // 2
            pad_br = total_pad - pad_tl
            x_padded = F.pad(x, (pad_tl, pad_br, pad_tl, pad_br))
            
            x_stat_unfold = F.unfold(x_padded, kernel_size=(sw_int, sw_int), stride=(sh, sw))
            x_stat_unfold = x_stat_unfold.view(B, C, sw_int * sw_int, H_out, W_out)
            x_stat = x_stat_unfold.permute(0, 1, 3, 4, 2)  # [B, C, H_out, W_out, sw_int^2]

        # --- 3. Calcular el descriptor phi sobre x_stat ---
        if self.metric == 'SMO':
            var = torch.var(x_stat, dim=-1, keepdim=True, unbiased=False)
            phi = 1.0 - (1.0 / (1.0 + var))
        elif self.metric == 'VAR':
            phi = torch.var(x_stat, dim=-1, keepdim=True, unbiased=False)
        elif self.metric == 'MEAN':
            phi = torch.mean(x_stat, dim=-1, keepdim=True)
        elif self.metric == 'MIN':
            phi = torch.min(x_stat, dim=-1, keepdim=True).values
        elif self.metric == 'MAX':
            phi = torch.max(x_stat, dim=-1, keepdim=True).values
        elif self.metric == 'MO1':
            phi = self._mode_low(x_stat)
        elif self.metric == 'MO2':
            phi = self._mode_high(x_stat)
        else:
            raise ValueError(f"Descriptor desconocido: {self.metric}")

        # --- 4. Asegurar que phi tenga la forma correcta para el broadcasting ---
        # Si es global, phi es [B, C, 1, 1] (o [B, C, H*W, 1] si no lo ajustamos).
        # Lo expandimos a [B, C, H_out, W_out, 1] para que coincida con x_win
        if self.stat_window == 'global':
            phi = phi.mean(dim=-1, keepdim=True)  # [B, C, 1, 1]
            phi = phi.expand(B, C, H_out, W_out, 1)

        # --- 5. Vector de pesos psi con manejo de regiones planas ---
        sum_phi = torch.sum(phi, dim=1, keepdim=True)
        psi = torch.where(sum_phi > 1e-8, phi / (sum_phi + 1e-8), torch.full_like(phi, 1.0 / C))

        # --- 6. Puntuación y selección del píxel ganador intacto ---
        score = torch.sum(x_win * psi, dim=1)  # [B, H_out, W_out, 4]
        max_idx = torch.argmax(score, dim=-1, keepdim=True)
        max_idx_expanded = max_idx.unsqueeze(1).expand(B, C, H_out, W_out, 1)
        
        out = torch.gather(x_win, dim=-1, index=max_idx_expanded).squeeze(-1)
        return out  # [B, 3, H_out, W_out]

    # --- Métodos auxiliares para la moda real (MO1 = moda menor en empate, MO2 = moda mayor en empate) ---
    def _compute_mode(self, x_win, high=False):
        # Ordenamos ascendente (MO1) o descendente (MO2) para desempatar determinísticamente
        sorted_vals, _ = torch.sort(x_win, dim=-1, descending=high)
        # Cuantizamos a niveles de 8 bits para contar frecuencias reales sin ruido numérico
        q = torch.round(sorted_vals * 255.0)
        n_elem = q.size(-1)

        if n_elem <= 64:
            counts = torch.zeros_like(q, dtype=torch.int32)
            for k in range(n_elem):
                counts += (q == q[..., k:k+1]).int()
            best_idx = torch.argmax(counts, dim=-1, keepdim=True)
            return torch.gather(sorted_vals, dim=-1, index=best_idx)

        # Caso ventana global: usar torch.unique por imagen y canal
        B, C = sorted_vals.shape[:2]
        out = torch.empty((B, C, 1, 1), dtype=x_win.dtype, device=x_win.device)
        for b in range(B):
            for c in range(C):
                vals, cnts = torch.unique(q[b, c], sorted=True, return_counts=True)
                if high:
                    vals = vals.flip(0)
                    cnts = cnts.flip(0)
                out[b, c, 0, 0] = vals[torch.argmax(cnts)] / 255.0
        return out

    def _mode_low(self, x_win):
        return self._compute_mode(x_win, high=False)

    def _mode_high(self, x_win):
        return self._compute_mode(x_win, high=True)