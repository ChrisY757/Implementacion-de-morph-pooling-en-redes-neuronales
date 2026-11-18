import math
import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_PARTITIONS = ("W0", "W1", "W4", "W9", "W16", "W25", "W36", "W49", "W64")


def normalize_partition(partition="W0", stat_window=None):
    """
    Normaliza el identificador de partición de dominio según Vázquez Noguera et al. (2019):
      - 'W0'  : Estrategia 1 (Vecindad local definida por la ventana B / stat_window).
      - 'W1'  : Estrategia 2 caso especial (1 sub-región = imagen completa / global).
      - 'WX'  : Estrategia 2 (X = M' * N' sub-regiones con unión de sub-regiones en contacto con B,
                ej. W4=2x2, W9=3x3, W16=4x4, W25=5x5, W49=7x7).
      - 'MxN' : Cuadrícula explícita M' x N' sub-regiones (ej. '3x3', '5x5').
    """
    if str(stat_window).strip().lower() == "global" and (
        partition is None or str(partition).strip().upper() in ("", "W0")
    ):
        return "W1"

    if partition is None:
        return "W0"

    raw = str(partition).strip().upper()
    if raw in ("", "0", "W0", "NEIGHBORHOOD", "VECINDAD", "LOCAL"):
        return "W0"
    if raw in ("1", "W1", "GLOBAL", "1X1"):
        return "W1"

    if raw.isdigit():
        raw = f"W{int(raw)}"

    if raw.startswith("W") and raw[1:].isdigit():
        n_reg = int(raw[1:])
        if n_reg == 0:
            return "W0"
        if n_reg == 1:
            return "W1"
        root = math.isqrt(n_reg)
        if root * root != n_reg or root < 1:
            raise ValueError(
                f"Partición '{partition}' inválida: el número de sub-regiones X en WX debe ser un cuadrado perfecto "
                f"(ej. W0, W1, W4, W9, W16, W25, W49) o formato MxN."
            )
        return f"W{n_reg}"

    if "X" in raw:
        parts = raw.split("X")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            m_grid, n_grid = int(parts[0]), int(parts[1])
            if m_grid < 1 or n_grid < 1:
                raise ValueError(f"Dimensiones de cuadrícula inválidas en '{partition}'.")
            if m_grid == 1 and n_grid == 1:
                return "W1"
            if m_grid == n_grid:
                return f"W{m_grid * n_grid}"
            return f"{m_grid}x{n_grid}"

    raise ValueError(
        f"Estrategia de partición desconocida: '{partition}'. "
        f"Use W0 (vecindad), W1 (global), W4, W9, W16, W25, W49 o MxN."
    )


def parse_partition_grid(partition_norm):
    """Retorna (M', N') para una partición normalizada distinta de W0."""
    if partition_norm == "W1":
        return (1, 1)
    if partition_norm.startswith("W"):
        n_reg = int(partition_norm[1:])
        root = math.isqrt(n_reg)
        return (root, root)
    m_str, n_str = partition_norm.lower().split("x")
    return (int(m_str), int(n_str))


def format_domain_label(partition="W0", stat_window=2, kernel_size=2):
    """Genera una etiqueta legible con la nomenclatura del paper (ej. 'B2W9 (3x3 sub-regiones)')."""
    p_norm = normalize_partition(partition, stat_window)
    sw_str = "global" if str(stat_window).strip().lower() == "global" else str(stat_window or kernel_size)

    if p_norm == "W0":
        b_size = kernel_size if sw_str == "global" else int(sw_str)
        return f"B{b_size}W0 (Vecindad {b_size}x{b_size})"
    if p_norm == "W1":
        b_size = kernel_size if sw_str == "global" else int(sw_str)
        return f"B{b_size}W1 (Global 1x1)"

    m_grid, n_grid = parse_partition_grid(p_norm)
    b_size = kernel_size if sw_str == "global" else int(sw_str)
    n_total = m_grid * n_grid
    return f"B{b_size}{p_norm} ({n_total} sub-regiones {m_grid}x{n_grid})"


class AdaptiveMorphPooling2d(nn.Module):
    """
    Capa de Pooling Morfológico Adaptativo Vectorial basada en el marco de ordenamiento
    de Vázquez Noguera et al. (IEEE Access, 2019).

    Soporta las dos estrategias de partición de dominio del artículo (Sección II-A):
      1) partition='W0' (Vecindad / Neighborhood):
         El dominio D_m corresponde a la ventana local B centrada/ubicada en la posición
         de filtrado (de tamaño stat_window x stat_window: 2, 3, 5, 7).
      2) partition='WX' (Partición de la imagen en sub-regiones: W1, W4, W9, W16, W25, W49...):
         La imagen M x N se divide en una matriz de M' x N' sub-regiones con floor(M/M') filas
         y floor(N/N') columnas. Para cada ventana B, cualquier sub-región en contacto con al
         menos un píxel de B forma parte de D_m (unión de sub-regiones en contacto, Figs. 2 y 3).
         El caso especial W1 corresponde a 1 sola región (imagen completa / global).
    """

    def __init__(self, kernel_size=2, stride=2, metric='SMO', stat_window=None, partition='W0'):
        super(AdaptiveMorphPooling2d, self).__init__()
        self.kernel_size = kernel_size     # Ventana de pooling (2x2)
        self.stride = stride               # Stride de pooling (2)
        self.metric = metric
        self.stat_window = stat_window     # None/2, 3, 5, 7 o 'global'
        self.partition = normalize_partition(partition, stat_window)
        self._layout_cache = {}

    def _get_subregion_layout(self, H, W, device):
        """
        Precalcula y almacena en caché los rangos de sub-regiones en contacto con cada ventana B
        según la Estrategia 2 del artículo (Figuras 2 y 3).
        """
        m_grid, n_grid = parse_partition_grid(self.partition)
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride

        if self.stat_window is None or str(self.stat_window).strip().lower() == "global":
            bw_h, bw_w = kh, kw
        else:
            bw_val = int(self.stat_window)
            bw_h, bw_w = max(kh, bw_val), max(kw, bw_val)

        cache_key = (H, W, kh, kw, sh, sw, bw_h, bw_w, m_grid, n_grid, device)
        if cache_key in self._layout_cache:
            return self._layout_cache[cache_key]

        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        # Tamaño base floor(M / M') y floor(N / N') según Sección II-A.2 del paper
        h_step = max(1, H // m_grid)
        w_step = max(1, W // n_grid)

        pad_top = (bw_h - kh) // 2
        pad_bottom = (bw_h - kh) - pad_top
        pad_left = (bw_w - kw) // 2
        pad_right = (bw_w - kw) - pad_left

        # Determinar para cada fila de salida qué sub-regiones verticales entran en contacto con B
        u_rows = []
        row_lookup = {}
        row_map_list = []
        for i in range(H_out):
            y0 = max(0, i * sh - pad_top)
            y1 = min(H - 1, i * sh + kh - 1 + pad_bottom)
            p0 = min(y0 // h_step, m_grid - 1)
            p1 = min(y1 // h_step, m_grid - 1)
            r_start = p0 * h_step
            r_end = (p1 + 1) * h_step if p1 < m_grid - 1 else H
            interval = (r_start, r_end)
            if interval not in row_lookup:
                row_lookup[interval] = len(u_rows)
                u_rows.append(interval)
            row_map_list.append(row_lookup[interval])

        # Determinar para cada columna de salida qué sub-regiones horizontales entran en contacto con B
        u_cols = []
        col_lookup = {}
        col_map_list = []
        for j in range(W_out):
            x0 = max(0, j * sw - pad_left)
            x1 = min(W - 1, j * sw + kw - 1 + pad_right)
            q0 = min(x0 // w_step, n_grid - 1)
            q1 = min(x1 // w_step, n_grid - 1)
            c_start = q0 * w_step
            c_end = (q1 + 1) * w_step if q1 < n_grid - 1 else W
            interval = (c_start, c_end)
            if interval not in col_lookup:
                col_lookup[interval] = len(u_cols)
                u_cols.append(interval)
            col_map_list.append(col_lookup[interval])

        # Agrupar dominios D_m únicos por cantidad de píxeles para vectorizar el cálculo de phi
        groups_by_area = {}
        for ur_idx, (r0, r1) in enumerate(u_rows):
            for uc_idx, (c0, c1) in enumerate(u_cols):
                area = (r1 - r0) * (c1 - c0)
                groups_by_area.setdefault(area, []).append((ur_idx, uc_idx, r0, r1, c0, c1))

        compiled_groups = []
        for area, items in groups_by_area.items():
            ur_t = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
            uc_t = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
            slices = [(it[2], it[3], it[4], it[5]) for it in items]
            compiled_groups.append((area, ur_t, uc_t, slices))

        row_map = torch.tensor(row_map_list, dtype=torch.long, device=device)
        col_map = torch.tensor(col_map_list, dtype=torch.long, device=device)

        layout = (len(u_rows), len(u_cols), compiled_groups, row_map, col_map)
        self._layout_cache[cache_key] = layout
        return layout

    def _compute_phi(self, x_stat):
        """Evalúa la función estadística phi sobre la última dimensión de x_stat."""
        if self.metric == 'SMO':
            var = torch.var(x_stat, dim=-1, keepdim=True, unbiased=False)
            return 1.0 - (1.0 / (1.0 + var))
        elif self.metric == 'VAR':
            return torch.var(x_stat, dim=-1, keepdim=True, unbiased=False)
        elif self.metric == 'MEAN':
            return torch.mean(x_stat, dim=-1, keepdim=True)
        elif self.metric == 'MIN':
            return torch.min(x_stat, dim=-1, keepdim=True).values
        elif self.metric == 'MAX':
            return torch.max(x_stat, dim=-1, keepdim=True).values
        elif self.metric == 'MO1':
            return self._mode_low(x_stat)
        elif self.metric == 'MO2':
            return self._mode_high(x_stat)
        elif self.metric == 'ENT':
            return self._compute_entropy(x_stat)
        else:
            raise ValueError(f"Descriptor desconocido: {self.metric}")

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

        # --- 2. Calcular descriptor phi según Estrategia de Partición de Dominio ---
        p_norm = normalize_partition(self.partition, self.stat_window)

        if p_norm == "W1":
            # Caso global / 1 sola sub-región (imagen completa)
            x_stat = x.reshape(B, C, 1, 1, H * W)
            phi = self._compute_phi(x_stat).expand(B, C, H_out, W_out, 1)

        elif p_norm == "W0":
            # Estrategia 1 del paper: Partición basada en vecindad (ventana local)
            if self.stat_window is None or int(self.stat_window) == self.kernel_size:
                x_stat = x_win  # [B, C, H_out, W_out, 4]
            else:
                sw_int = int(self.stat_window)
                total_pad = sw_int - kh
                pad_tl = total_pad // 2
                pad_br = total_pad - pad_tl
                x_padded = F.pad(x, (pad_tl, pad_br, pad_tl, pad_br))

                x_stat_unfold = F.unfold(x_padded, kernel_size=(sw_int, sw_int), stride=(sh, sw))
                x_stat_unfold = x_stat_unfold.view(B, C, sw_int * sw_int, H_out, W_out)
                x_stat = x_stat_unfold.permute(0, 1, 3, 4, 2)  # [B, C, H_out, W_out, sw_int^2]

            phi = self._compute_phi(x_stat)

        else:
            # Estrategia 2 del paper: Partición en sub-regiones W_1..W_lambda con unión por contacto
            n_urows, n_ucols, compiled_groups, row_map, col_map = self._get_subregion_layout(H, W, x.device)
            phi_grid = torch.empty((B, C, n_urows, n_ucols, 1), dtype=x.dtype, device=x.device)

            for area, ur_t, uc_t, slices in compiled_groups:
                patches = torch.stack(
                    [x[:, :, r0:r1, c0:c1].reshape(B, C, area) for (r0, r1, c0, c1) in slices],
                    dim=2
                )  # [B, C, n_domains_in_group, area]
                phi_grp = self._compute_phi(patches)  # [B, C, n_domains_in_group, 1]
                phi_grid[:, :, ur_t, uc_t, :] = phi_grp

            phi = phi_grid[:, :, row_map[:, None], col_map[None, :], :]  # [B, C, H_out, W_out, 1]

        # --- 3. Vector de pesos psi con manejo de regiones planas ---
        sum_phi = torch.sum(phi, dim=1, keepdim=True)
        psi = torch.where(sum_phi > 1e-8, phi / (sum_phi + 1e-8), torch.full_like(phi, 1.0 / C))

        # --- 4. Puntuación escalar T(v) = <v, psi> y selección del píxel ganador intacto ---
        score = torch.sum(x_win * psi, dim=1)  # [B, H_out, W_out, 4]
        max_idx = torch.argmax(score, dim=-1, keepdim=True)
        max_idx_expanded = max_idx.unsqueeze(1).expand(B, C, H_out, W_out, 1)

        out = torch.gather(x_win, dim=-1, index=max_idx_expanded).squeeze(-1)
        return out  # [B, 3, H_out, W_out]

    # --- Métodos auxiliares para la moda real (MO1 = moda menor en empate, MO2 = moda mayor en empate) ---
    def _compute_mode(self, x_stat, high=False):
        # Cuantizamos a niveles enteros de 8 bits (Escala [0, 255] del histograma del paper)
        q = torch.round(x_stat * 255.0)
        n_elem = q.size(-1)

        if n_elem <= 64:
            sorted_q, _ = torch.sort(q, dim=-1, descending=high)
            counts = torch.zeros_like(sorted_q, dtype=torch.int32)
            for k in range(n_elem):
                counts += (sorted_q == sorted_q[..., k:k + 1]).int()
            best_idx = torch.argmax(counts, dim=-1, keepdim=True)
            return torch.gather(sorted_q, dim=-1, index=best_idx) / 255.0

        # Caso dominios grandes (sub-regiones W4..W49 o global W1): histograma vectorizado con scatter_add_
        q_min = q.min()
        q_span = int((q.max() - q_min).item()) + 1
        if q_span <= 4096:
            q_idx = (q - q_min).to(torch.long)
            hist = torch.zeros((*q.shape[:-1], q_span), dtype=torch.int32, device=x_stat.device)
            hist.scatter_add_(-1, q_idx, torch.ones_like(q_idx, dtype=torch.int32))
            if high:
                best_bin = (q_span - 1) - torch.argmax(hist.flip(-1), dim=-1, keepdim=True)
            else:
                best_bin = torch.argmax(hist, dim=-1, keepdim=True)
            return (best_bin.to(x_stat.dtype) + q_min) / 255.0

        # Fallback general en caso de rangos extremos fuera de escala
        flat_q = q.reshape(-1, n_elem)
        out_flat = torch.empty((flat_q.size(0), 1), dtype=x_stat.dtype, device=x_stat.device)
        for idx in range(flat_q.size(0)):
            vals, cnts = torch.unique(flat_q[idx], sorted=True, return_counts=True)
            if high:
                vals = vals.flip(0)
                cnts = cnts.flip(0)
            out_flat[idx, 0] = vals[torch.argmax(cnts)] / 255.0
        return out_flat.reshape(*q.shape[:-1], 1)

    def _quantize_255(self, x_stat):
        """
        Convierte las intensidades de x_stat a niveles enteros discretos {0, 1, ..., 255}
        del histograma por componente RGB. Si el tensor proviene de normalización ImageNet
        (modelos preentrenados con valores fuera de [0, 1]), restituye primero la escala [0, 1].
        """
        x_min = float(x_stat.min().item())
        x_max = float(x_stat.max().item())
        if (x_min < -1e-4 or x_max > 1.0 + 1e-4) and x_stat.size(1) == 3:
            shape = [1, 3] + [1] * (x_stat.ndim - 2)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x_stat.dtype, device=x_stat.device).view(*shape)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x_stat.dtype, device=x_stat.device).view(*shape)
            x_01 = (x_stat * std + mean).clamp(0.0, 1.0)
        else:
            x_01 = x_stat.clamp(0.0, 1.0)
        return torch.round(x_01 * 255.0).to(torch.long)

    def _compute_entropy(self, x_stat):
        """
        Calcula la Entropía de Shannon phi_Ent sobre el histograma de 8 bits (niveles 0 a 255)
        para cada canal RGB en el dominio D_m (ventana local, sub-región o imagen global):
            phi_Ent(h_{f_j}^{D_m}) = - sum_{i=0..255, p_j(i)>0} p_j(i) * log2(p_j(i))
        donde p_j(i) = h_{f_j}^{D_m}(i) / n.
        """
        q = self._quantize_255(x_stat)
        n_elem = q.size(-1)

        if n_elem <= 64:
            # Identidad exacta sin alojar 256 bins por cada ventana espacial:
            # -sum_{v} (c_v / n) log2(c_v / n) == log2(n) - (1/n) * sum_{m=1..n} log2(c(q_m))
            counts = torch.zeros(q.shape, dtype=torch.float32, device=x_stat.device)
            for k in range(n_elem):
                counts += (q == q[..., k:k + 1]).to(torch.float32)
            ent = math.log2(float(n_elem)) - torch.mean(torch.log2(counts), dim=-1, keepdim=True)
            return ent.clamp_min(0.0).to(x_stat.dtype)

        # Para dominios amplios (sub-regiones W4..W64 o global W1): histograma exacto de 256 niveles (0..255)
        hist = torch.zeros((*q.shape[:-1], 256), dtype=torch.float32, device=x_stat.device)
        hist.scatter_add_(-1, q, torch.ones_like(q, dtype=torch.float32))
        probs = hist / float(n_elem)
        log_probs = torch.where(probs > 0.0, torch.log2(probs.clamp_min(1e-12)), torch.zeros_like(probs))
        ent = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        return ent.clamp_min(0.0).to(x_stat.dtype)

    def _mode_low(self, x_stat):
        return self._compute_mode(x_stat, high=False)

    def _mode_high(self, x_stat):
        return self._compute_mode(x_stat, high=True)
