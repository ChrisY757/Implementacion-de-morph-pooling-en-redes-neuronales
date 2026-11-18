import os
import gc
import math
import time
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

import argparse
from morph_pool_layer import normalize_partition, format_domain_label
from models import CNNClassifier, PretrainedWithPool
from report_generator import generate_html_report
from report_aggregator import aggregate_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experimento A/B: Pooling Morfológico Adaptativo vs MaxPool2d en Imágenes Dermatoscópicas (Melanoma)"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=5,
        help="Número de etapas/épocas de entrenamiento por modelo (default: 5)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Tamaño del lote / batch size (default automático según VRAM y arquitectura)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje / Learning rate (default: 0.001 en simple, 0.0001 en preentrenadas)"
    )
    parser.add_argument(
        "--metric",
        choices=["SMO", "VAR", "ENT", "MEAN", "MIN", "MAX", "MO1", "MO2"],
        default="SMO",
        help="Descriptor estadístico phi por canal RGB (incluye ENT = Entropía 0-255; default: SMO)"
    )
    parser.add_argument(
        "-d", "--data-dir", "--dataset",
        dest="data_dir",
        type=str,
        default="archive",
        help="Carpeta del dataset a utilizar: 'archive', 'archive2', 'archive11k_2', etc. (default: archive)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad A/B (default: 42)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="reporte_experimento_melanoma.html",
        help="Nombre o ruta del informe HTML de salida; se guarda en 'reportes/' (default: reporte_experimento_melanoma.html)"
    )
    parser.add_argument(
        "--stat-window",
        type=str,
        default="2",
        choices=["2", "3", "5", "7", "global"],
        help="Tamaño de la ventana B para vecindad (W0) o ventana de contacto en sub-regiones (WX) (default: 2)"
    )
    parser.add_argument(
        "-p", "--partition", "--domain",
        dest="partition",
        type=str,
        default="W0",
        help=(
            "Estrategia de partición de dominio según el paper (Sección II-A): "
            "W0 (vecindad local), W1 (global 1x1), W4 (2x2), W9 (3x3), "
            "W16 (4x4), W25 (5x5), W49 (7x7) o MxN (default: W0)"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "vgg16", "resnet18"],
        help="Arquitectura: simple (CNN), vgg16 o resnet18 (default: simple)"
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="Pooling baseline (default: max). El propuesto siempre es morph."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Lista de semillas para ejecutar múltiples experimentos (ej: --seeds 42 7 123)"
    )
    parser.add_argument(
        "--epochs-list",
        type=int,
        nargs="+",
        default=None,
        help="Lista de épocas para ejecutar múltiples experimentos (ej: --epochs-list 5 10)"
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Activa precisión mixta FP16 (Tensor Cores en Tesla T4 / GPUs modernas). Por defecto automático si VRAM >= 8 GB."
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Desactiva precisión mixta FP16 forzando FP32 puro."
    )

    return parser.parse_args()


# 1. Dataset precargado en RAM con decodificación JPEG paralela multihilo
class FastRAMDataset(Dataset):
    def __init__(self, folder_path, transform):
        print(f"Cargando {folder_path} directamente a memoria RAM...")
        raw_dataset = datasets.ImageFolder(folder_path)
        samples = raw_dataset.samples
        loader = raw_dataset.loader
        total = len(samples)

        first_img = transform(loader(samples[0][0]))
        self.images = torch.empty((total, *first_img.shape), dtype=first_img.dtype)
        self.labels = torch.empty(total, dtype=torch.float32)
        self.images[0] = first_img
        self.labels[0] = samples[0][1]

        def _load_one(idx):
            path, label = samples[idx]
            return idx, transform(loader(path)), label

        num_workers = min(8, max(2, (os.cpu_count() or 2) * 2))
        done_count = 1
        if total > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                for idx, t_img, label in ex.map(_load_one, range(1, total), chunksize=32):
                    self.images[idx] = t_img
                    self.labels[idx] = label
                    done_count += 1
                    if done_count % 500 == 0 or done_count == total:
                        print(f"\r    Leídas {done_count}/{total} imágenes...", end="", flush=True)
        print(f"\r    Leídas {total}/{total} imágenes.          ")
        print(f"--> {total} imágenes listas en RAM.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Detección automática de dispositivo y VRAM disponible
def get_best_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)} - {vram_gb:.1f} GB VRAM)", vram_gb
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "Intel XPU", 0.0
    return torch.device("cpu"), f"CPU ({torch.get_num_threads()} hilos con oneDNN/MKLDNN)", 0.0


DEVICE, DEVICE_NAME, TOTAL_VRAM_GB = get_best_device()

transform_simple = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

transform_pretrained = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _can_store_in_vram(needed_bytes, reserve_gb=1.8):
    """Verifica si el tensor precalculado cabe holgadamente directo en la VRAM de la GPU."""
    if DEVICE.type != "cuda":
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return (free_bytes - needed_bytes) > int(reserve_gb * (1024 ** 3))
    except Exception:
        return False


@torch.inference_mode()
def _precompute_static_features(model, model_name, images_tensor, batch_size, desc="", use_amp=False):
    """
    Precalcula una sola vez la etapa estática/congelada (pooling inicial en FP32 exacto + bloques congelados).
    Si hay suficiente VRAM libre (como los 15 GB de Tesla T4), aloja el tensor directamente en VRAM
    para eliminar al 100% las transferencias CPU->GPU durante todas las épocas y semillas.
    """
    model.eval()
    out_tensor = None
    target_dev = torch.device("cpu")
    n = images_tensor.size(0)
    # En inferencia estática sin gradientes podemos usar lotes grandes para exprimir la GPU
    if DEVICE.type == "cuda" and TOTAL_VRAM_GB >= 8.0:
        step = max(batch_size, 128)
    elif DEVICE.type == "cuda":
        step = max(batch_size, 32)
    else:
        step = max(batch_size, 64)

    for i in range(0, n, step):
        batch = images_tensor[i:i + step].to(DEVICE, non_blocking=True)
        if model_name == "simple":
            # El pooling RGB siempre corre en FP32 exacto para máxima precisión estadística
            out = model.morph_pool(batch) if model.use_morph_pool else model.max_pool_rgb(batch)
        else:
            # El pooling RGB corre en FP32 exacto; las convoluciones congeladas pueden usar Tensor Cores
            pooled = model.pool(batch)
            if use_amp and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model.frozen_features(pooled)
            else:
                out = model.frozen_features(pooled)

        if out_tensor is None:
            needed_bytes = n * math.prod(out.shape[1:]) * out.element_size()
            if _can_store_in_vram(needed_bytes):
                target_dev = DEVICE
                out_tensor = torch.empty((n, *out.shape[1:]), dtype=out.dtype, device=DEVICE)
            else:
                target_dev = torch.device("cpu")
                out_tensor = torch.empty((n, *out.shape[1:]), dtype=out.dtype, pin_memory=(DEVICE.type == "cuda"))

        if target_dev.type == "cuda":
            out_tensor[i:i + out.size(0)].copy_(out)
        else:
            out_tensor[i:i + out.size(0)].copy_(out.cpu())

        if (i // step) % 4 == 0 or (i + step) >= n:
            loc_str = "VRAM GPU" if target_dev.type == "cuda" else "RAM CPU"
            print(f"\r    [{desc} -> {loc_str}] Procesadas {min(i + step, n)}/{n} imágenes...", end="", flush=True)
    print()
    return out_tensor


def _build_model(model_name, pool_type, use_morph, metric, stat_window, partition):
    if model_name == "simple":
        return CNNClassifier(
            use_morph_pool=use_morph, metric=metric,
            stat_window=stat_window, pool_type=pool_type, partition=partition
        ).to(DEVICE)
    return PretrainedWithPool(
        model_name, use_morph=use_morph, metric=metric,
        stat_window=stat_window, pool_type=pool_type, partition=partition
    ).to(DEVICE)


def run_experiment(model_name, pool_type, use_morph, name, epochs, batch_size, lr,
                   metric, stat_window, train_dataset, test_dataset, seed=42,
                   partition="W0", use_amp=False, precomputed=None):
    set_seed(seed)
    model = _build_model(model_name, pool_type, use_morph, metric, stat_window, partition)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_tag = "FP16 TensorCores" if (use_amp and DEVICE.type == "cuda") else "FP32"
    print(f"[Parámetros {model_name} | {amp_tag}] Entrenables: {trainable_params:,} / Total: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    use_foreach = DEVICE.type in ("cuda", "cpu")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, foreach=use_foreach)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and DEVICE.type == "cuda"))

    print(f"\n================ Ejecutando: {name} ({epochs} épocas | Seed: {seed}) ================")
    start_time = time.time()

    own_precomputed = False
    if precomputed is not None:
        train_x, test_x = precomputed
        loc_label = "VRAM GPU" if train_x.is_cuda else "RAM"
        print(f" -> Reutilizando características estáticas ya alojadas en {loc_label} (0.0s overhead).")
    else:
        own_precomputed = True
        print(" -> Precalculando etapa estática (Pooling RGB + bloques congelados)...")
        train_x = _precompute_static_features(model, model_name, train_dataset.images, batch_size, desc="Train", use_amp=use_amp)
        test_x = _precompute_static_features(model, model_name, test_dataset.images, batch_size, desc="Test", use_amp=use_amp)

    train_y = train_dataset.labels.to(DEVICE) if train_x.is_cuda else train_dataset.labels
    test_y_cpu = test_dataset.labels
    train_size = len(train_dataset.labels)
    test_size = len(test_y_cpu)
    x_in_vram = train_x.is_cuda

    # Reiniciar semilla antes del bucle de épocas para garantizar idéntico barajado A/B
    set_seed(seed)
    epoch_losses = []
    num_batches = max(1, (train_size + batch_size - 1) // batch_size)
    log_every = max(1, num_batches // 4)

    for epoch in range(epochs):
        ep_t0 = time.time()
        model.train()
        running_loss_t = torch.zeros((), dtype=torch.float32, device=DEVICE)
        # Mantener torch.randperm en CPU preserva exactamente la misma secuencia de lotes entre CPU/GPU
        perm = torch.randperm(train_size)
        if x_in_vram:
            perm_dev = perm.to(DEVICE, non_blocking=True)

        for b_idx, i in enumerate(range(0, train_size, batch_size)):
            if x_in_vram:
                idx = perm_dev[i:i + batch_size]
                images = train_x[idx]
                labels = train_y[idx].unsqueeze(1)
            else:
                idx = perm[i:i + batch_size]
                images = train_x[idx].to(DEVICE, non_blocking=True)
                labels = train_y[idx].unsqueeze(1).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and DEVICE.type == "cuda")):
                if model_name == "simple":
                    outputs = model.classifier(model.features(images))
                else:
                    outputs = model.forward_trainable(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss_t += loss.detach().float() * images.size(0)
            if (b_idx + 1) % log_every == 0 or (b_idx + 1) == num_batches:
                print(f"\r    Época [{epoch+1}/{epochs}] Lote {min(i + batch_size, train_size)}/{train_size}...", end="", flush=True)

        epoch_loss = (running_loss_t / train_size).item()
        epoch_losses.append(epoch_loss)
        print(f"\rÉpoca [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} ({time.time() - ep_t0:.2f}s)          ")

    elapsed = time.time() - start_time
    print(f"Tiempo de entrenamiento: {elapsed:.2f} segundos.")

    # Evaluación en Test
    model.eval()
    all_probs = []
    eval_step = max(batch_size, 128) if (DEVICE.type == "cuda" and TOTAL_VRAM_GB >= 8.0) else batch_size
    with torch.inference_mode():
        for i in range(0, test_size, eval_step):
            images = test_x[i:i + eval_step] if test_x.is_cuda else test_x[i:i + eval_step].to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_amp and DEVICE.type == "cuda")):
                if model_name == "simple":
                    outputs = model.classifier(model.features(images))
                else:
                    outputs = model.forward_trainable(images)
            probs = torch.sigmoid(outputs.float())
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy().ravel()
    all_preds = (all_probs > 0.5).astype(np.float32)
    all_labels = test_y_cpu.numpy().ravel()

    acc = accuracy_score(all_labels, all_preds)
    sens = recall_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    spec = tn / (tn + fp)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    print(f"\n--- Resultados Finales ({name}) ---")
    print(f"Accuracy:      {acc*100:.2f}%")
    print(f"Sensibilidad:  {sens*100:.2f}%")
    print(f"Especificidad: {spec*100:.2f}%")
    print(f"AUC-ROC:       {auc:.4f}")

    del model
    if own_precomputed:
        del train_x, test_x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "name": name,
        "use_morph": use_morph,
        "epoch_losses": epoch_losses,
        "elapsed": elapsed,
        "acc": acc,
        "sens": sens,
        "spec": spec,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist()
    }


if __name__ == '__main__':
    args = parse_args()
    # Si hay >= 8 GB de VRAM (ej. Tesla T4 15 GB en Colab), usamos batch 64 por defecto también en ResNet/VGG;
    # en GPUs de 4 GB (GTX 1650) mantenemos 32/16 para cuidar VRAM salvo que el usuario indique -b.
    if DEVICE.type == "cuda" and TOTAL_VRAM_GB < 7.5:
        default_bs = {"simple": 64, "resnet18": 32, "vgg16": 16}
    else:
        default_bs = {"simple": 64, "resnet18": 64, "vgg16": 64}

    batch_size = args.batch_size if args.batch_size is not None else default_bs[args.model]
    lr = args.lr if args.model == "simple" else 1e-4
    use_amp = args.amp if args.amp is not None else (DEVICE.type == "cuda" and TOTAL_VRAM_GB >= 7.5)
    selected_transform = transform_simple if args.model == "simple" else transform_pretrained
    resolution = "112x112" if args.model == "simple" else "224x224"
    stat_window = args.stat_window if args.stat_window == "global" else int(args.stat_window)
    try:
        partition_norm = normalize_partition(args.partition, stat_window)
    except ValueError as exc:
        print(f"\n[!] Error en argumento --partition: {exc}\n")
        exit(1)
    domain_label = format_domain_label(partition_norm, stat_window)
    short_domain_tag = domain_label.split(" ")[0]

    raw_dir = args.data_dir.strip()
    candidates_map = {
        "1": ["archive", "archive1", "archive 1"],
        "archive": ["archive", "archive1", "archive 1"],
        "archive1": ["archive", "archive1", "archive 1"],
        "archive 1": ["archive", "archive1", "archive 1"],
        "2": ["archive2", "archive 2"],
        "archive2": ["archive2", "archive 2"],
        "archive 2": ["archive2", "archive 2"],
        "3": ["archive3", "archive 3"],
        "archive3": ["archive3", "archive 3"],
        "archive 3": ["archive3", "archive 3"],
    }
    candidates = candidates_map.get(raw_dir.lower(), [raw_dir, raw_dir.replace(" ", "")])
    data_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])
    dataset_label = os.path.basename(os.path.normpath(data_dir))

    seeds_list = args.seeds if args.seeds is not None else [args.seed]
    epochs_list = args.epochs_list if args.epochs_list is not None else [args.epochs]
    multi_run = (len(seeds_list) > 1 or len(epochs_list) > 1)

    print("\n" + "="*65)
    print(f" INICIO DEL EXPERIMENTO COMPARATIVO A/B (MELANOMA) - {args.model.upper()}")
    print(f" Dataset: {dataset_label} | Épocas: {epochs_list} | Batch Size: {batch_size} | LR: {lr}")
    print(f" Métrica: {args.metric} | Dominio: {domain_label} | Pool Baseline: {args.pool.upper()} | Seeds: {seeds_list}")
    print(f" Dispositivo: {DEVICE_NAME} | Precisión: {'AMP FP16 (Tensor Cores)' if use_amp else 'FP32'}")
    print("="*65)

    if args.model != "simple" and args.metric not in ("SMO", "VAR", "ENT"):
        print(f"\n[!] ADVERTENCIA METODOLÓGICA: '{args.metric}' se evalúa sobre tensores normalizados con ImageNet")
        print("    (valores centrados en 0 y negativos). Se recomienda usar 'SMO', 'VAR' o 'ENT' para")
        print("    comparaciones consistentes entre arquitecturas.\n")

    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"\n[!] Error: No se encontraron las carpetas 'train' o 'test' en: {data_dir}")
        print("    Ejemplos de uso: -d archive | -d \"archive 2\" | -d \"archive 3\"\n")
        exit(1)

    print("\n=== PREPARANDO DATASETS EN RAM (Precarga paralela) ===")
    train_dataset = FastRAMDataset(train_dir, transform=selected_transform)
    test_dataset = FastRAMDataset(test_dir, transform=selected_transform)

    # Precalcular UNA SOLA VEZ las características estáticas para Baseline y MorphPool
    # (son deterministas e independientes de la semilla, ahorrando minutos enormes en multi-seed)
    print("\n=== PRECALCULANDO CARACTERÍSTICAS ESTÁTICAS EN VRAM/RAM (1 sola vez para todas las semillas) ===")
    t_pre0 = time.time()
    tmp_base_model = _build_model(args.model, args.pool, False, args.metric, stat_window, partition_norm)
    base_train_x = _precompute_static_features(tmp_base_model, args.model, train_dataset.images, batch_size, desc="Baseline Train", use_amp=use_amp)
    base_test_x = _precompute_static_features(tmp_base_model, args.model, test_dataset.images, batch_size, desc="Baseline Test", use_amp=use_amp)
    del tmp_base_model

    tmp_prop_model = _build_model(args.model, "morph", True, args.metric, stat_window, partition_norm)
    prop_train_x = _precompute_static_features(tmp_prop_model, args.model, train_dataset.images, batch_size, desc="MorphPool Train", use_amp=use_amp)
    prop_test_x = _precompute_static_features(tmp_prop_model, args.model, test_dataset.images, batch_size, desc="MorphPool Test", use_amp=use_amp)
    del tmp_prop_model

    # Liberar los tensores crudos 224x224 de la RAM de sistema (libera hasta 6.6 GB en Colab!)
    train_dataset.images = None
    test_dataset.images = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"--> Precomputación completada en {time.time() - t_pre0:.1f}s | VRAM activa con tensores cacheados: {used_vram:.2f} GB")

    base_out_name, base_out_ext = os.path.splitext(args.output)
    if not base_out_ext:
        base_out_ext = ".html"

    for ep in epochs_list:
        ep_results = []
        for seed in seeds_list:
            set_seed(seed)
            baseline_results = run_experiment(
                model_name=args.model,
                pool_type=args.pool,
                use_morph=False,
                name=f"Baseline ({args.model.upper()} + {args.pool.upper()}Pool RGB)",
                epochs=ep,
                batch_size=batch_size,
                lr=lr,
                metric=args.metric,
                stat_window=stat_window,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                seed=seed,
                partition=partition_norm,
                use_amp=use_amp,
                precomputed=(base_train_x, base_test_x)
            )

            proposed_results = run_experiment(
                model_name=args.model,
                pool_type="morph",
                use_morph=True,
                name=f"Propuesto ({args.model.upper()} + MorphPool {args.metric} {short_domain_tag})",
                epochs=ep,
                batch_size=batch_size,
                lr=lr,
                metric=args.metric,
                stat_window=stat_window,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                seed=seed,
                partition=partition_norm,
                use_amp=use_amp,
                precomputed=(prop_train_x, prop_test_x)
            )

            run_output = (
                f"{base_out_name}_s{seed}_e{ep}{base_out_ext}"
                if multi_run else args.output
            )
            run_config = {
                "dataset": f"ISIC ({dataset_label})",
                "model": args.model,
                "pool": args.pool,
                "seed": seed,
                "batch_size": batch_size,
                "epochs": ep,
                "lr": lr,
                "metric": args.metric,
                "stat_window": args.stat_window,
                "partition": partition_norm,
                "domain_label": domain_label,
                "device": DEVICE_NAME,
                "resolution": resolution,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            report_path = generate_html_report(
                baseline=baseline_results,
                proposed=proposed_results,
                config=run_config,
                output_path=run_output
            )

            print("\n" + "="*65)
            print(f" [OK] INFORME INDIVIDUAL GENERADO (Seed: {seed} | Épocas: {ep})")
            print(f" Archivo HTML: {report_path}")
            print("="*65)

            ep_results.append({
                "seed": seed,
                "epochs": ep,
                "baseline": baseline_results,
                "proposed": proposed_results
            })

        if len(seeds_list) > 1:
            agg_out = f"{base_out_name}_agregado_e{ep}{base_out_ext}"
            agg_path = aggregate_report(
                results=ep_results,
                output_path=agg_out,
                config={
                    "dataset": f"ISIC ({dataset_label})",
                    "model": args.model,
                    "pool": args.pool,
                    "epochs": ep,
                    "metric": args.metric,
                    "stat_window": args.stat_window,
                    "partition": partition_norm,
                    "domain_label": domain_label,
                    "device": DEVICE_NAME,
                }
            )
            print("\n" + "="*65)
            print(f" [OK] INFORME ESTADÍSTICO AGREGADO ({len(seeds_list)} seeds | Épocas: {ep})")
            print(f" Archivo HTML Agregado: {agg_path}")
            print("="*65 + "\n")
