import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve

import argparse
from models import CNNClassifier, PretrainedWithPool
from report_generator import generate_html_report

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
        help="Tamaño del lote / batch size (default automático: 64 en simple, 32 en resnet18, 16 en vgg16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje / Learning rate (default: 0.001)"
    )
    parser.add_argument("--metric", 
    choices=["SMO", "VAR", "MEAN", "MIN", "MAX", "MO1", "MO2"], 
    default="SMO")
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default="./archive",
        help="Ruta al directorio de datos que contiene 'train' y 'test' (default: ./archive)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="reporte_experimento_melanoma.html",
        help="Nombre o ruta del informe HTML de salida; se guarda en la carpeta 'reportes/' (default: reporte_experimento_melanoma.html)"
    )
    parser.add_argument(
        "--stat-window",
        type=str,
        default="2",
        choices=["2", "3", "5", "7", "global"],
        help="Tamaño de la ventana estadística para psi (default: 2)"
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
    
    return parser.parse_args()

# 1. Dataset ultra rápido que precarga todo en memoria RAM
class FastRAMDataset(Dataset):
    def __init__(self, folder_path, transform):
        print(f"Cargando {folder_path} directamente a memoria RAM...")
        raw_dataset = datasets.ImageFolder(folder_path)
        total = len(raw_dataset)
        self.images = []
        self.labels = []

        for idx, (img, label) in enumerate(raw_dataset, 1):
            self.images.append(transform(img))
            self.labels.append(label)
            if idx % 250 == 0 or idx == total:
                print(f"\r    Leídas {idx}/{total} imágenes...", end="", flush=True)
        print()

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(f"--> {len(self.labels)} imágenes listadas en RAM.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Detección automática de dispositivo (CUDA / XPU / CPU)
def get_best_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "Intel XPU"
    return torch.device("cpu"), f"CPU ({torch.get_num_threads()} hilos con oneDNN/MKLDNN)"

DEVICE, DEVICE_NAME = get_best_device()

transform_simple = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

transform_pretrained = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalización de ImageNet (nota: usar descriptores SMO o VAR, invariantes a desplazamiento)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.inference_mode()
def _precompute_static_features(model, model_name, images_tensor, batch_size, desc=""):
    """
    Precalcula una sola vez la etapa estática/congelada (pooling inicial + bloques congelados)
    ya que no posee parámetros entrenables y su salida es invariante entre épocas.
    """
    model.eval()
    outputs = []
    n = images_tensor.size(0)
    step = max(batch_size, 64)

    for i in range(0, n, step):
        batch = images_tensor[i:i + step].to(DEVICE, non_blocking=True)
        if model_name == "simple":
            out = model.morph_pool(batch) if model.use_morph_pool else model.max_pool_rgb(batch)
        else:
            out = model.extract_frozen(batch)
        outputs.append(out.cpu())
        print(f"\r    [{desc}] Procesadas {min(i + step, n)}/{n} imágenes...", end="", flush=True)
    print()
    return torch.cat(outputs, dim=0)


def run_experiment(model_name, pool_type, use_morph, name, epochs, batch_size, lr,
                   metric, stat_window, train_dataset, test_dataset):
    set_seed(42)
    if model_name == "simple":
        model = CNNClassifier(use_morph_pool=use_morph, metric=metric,
                              stat_window=stat_window, pool_type=pool_type).to(DEVICE)
    else:
        model = PretrainedWithPool(model_name, use_morph=use_morph, metric=metric,
                                   stat_window=stat_window, pool_type=pool_type).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Parámetros {model_name}] Entrenables: {trainable_params:,} / Total: {total_params:,}")
    criterion = nn.BCEWithLogitsLoss()
    use_foreach = DEVICE.type in ("cuda", "cpu")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, foreach=use_foreach)

    print(f"\n================ Ejecutando: {name} ({epochs} épocas) ================")
    start_time = time.time()

    # Precalcular etapa sin parámetros entrenables (Pooling RGB + capas congeladas)
    print(" -> Precalculando etapa estática (Pooling RGB + bloques congelados)...")
    train_x = _precompute_static_features(model, model_name, train_dataset.images, batch_size, desc="Train")
    test_x = _precompute_static_features(model, model_name, test_dataset.images, batch_size, desc="Test")
    train_y = train_dataset.labels
    test_y = test_dataset.labels
    train_size = len(train_y)
    test_size = len(test_y)

    # Reiniciar semilla antes del bucle de épocas para garantizar idéntico barajado A/B
    set_seed(42)
    epoch_losses = []

    for epoch in range(epochs):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        perm = torch.randperm(train_size)

        for i in range(0, train_size, batch_size):
            idx = perm[i:i + batch_size]
            images = train_x[idx].to(DEVICE, non_blocking=True)
            labels = train_y[idx].unsqueeze(1).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if model_name == "simple":
                outputs = model.classifier(model.features(images))
            else:
                outputs = model.forward_trainable(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            print(f"\r    Época [{epoch+1}/{epochs}] Lote {min(i + batch_size, train_size)}/{train_size}...", end="", flush=True)

        epoch_loss = running_loss / train_size
        epoch_losses.append(epoch_loss)
        print(f"\rÉpoca [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} ({time.time() - ep_t0:.1f}s)          ")

    elapsed = time.time() - start_time
    print(f"Tiempo de entrenamiento: {elapsed:.2f} segundos.")

    # Evaluación en Test
    model.eval()
    all_probs = []
    with torch.inference_mode():
        for i in range(0, test_size, batch_size):
            images = test_x[i:i + batch_size].to(DEVICE, non_blocking=True)
            if model_name == "simple":
                outputs = model.classifier(model.features(images))
            else:
                outputs = model.forward_trainable(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy().ravel()
    all_preds = (all_probs > 0.5).astype(np.float32)
    all_labels = test_y.numpy().ravel()

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
    # En GPU con VRAM limitada usar 32/16; en CPU (RAM de sistema) usar 64 para minimizar pasos de AdamW
    if DEVICE.type == "cuda":
        default_bs = {"simple": 64, "resnet18": 32, "vgg16": 16}
    else:
        default_bs = {"simple": 64, "resnet18": 64, "vgg16": 64}
    batch_size = args.batch_size if args.batch_size is not None else default_bs[args.model]
    lr = args.lr if args.model == "simple" else 1e-4
    selected_transform = transform_simple if args.model == "simple" else transform_pretrained
    resolution = "112x112" if args.model == "simple" else "224x224"
    stat_window = args.stat_window if args.stat_window == "global" else int(args.stat_window)

    print("\n" + "="*65)
    print(f" INICIO DEL EXPERIMENTO COMPARATIVO A/B (MELANOMA) - {args.model.upper()}")
    print(f" Épocas: {args.epochs} | Batch Size: {batch_size} | LR: {lr} | Métrica: {args.metric} | Ventana: {args.stat_window}")
    print(f" Dispositivo: {DEVICE_NAME}")
    print("="*65)

    if args.model != "simple" and args.metric not in ("SMO", "VAR"):
        print(f"\n[!] ADVERTENCIA METODOLÓGICA: '{args.metric}' se evalúa sobre tensores normalizados con ImageNet")
        print("    (valores centrados en 0 y negativos). Se recomienda usar 'SMO' o 'VAR' para")
        print("    comparaciones consistentes entre arquitecturas.\n")

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"\n[!] Error: No se encontraron las carpetas 'train' o 'test' en: {args.data_dir}")
        print("    Asegúrate de que la ruta sea correcta o especifícala con --data-dir <ruta>\n")
        exit(1)

    print("\n=== PREPARANDO DATASETS EN RAM (Precarga de aceleración) ===")
    train_dataset = FastRAMDataset(train_dir, transform=selected_transform)
    test_dataset = FastRAMDataset(test_dir, transform=selected_transform)

    baseline_results = run_experiment(
        model_name=args.model,
        pool_type=args.pool,
        use_morph=False,
        name=f"Baseline ({args.model.upper()} + {args.pool.upper()}Pool RGB)",
        epochs=args.epochs,
        batch_size=batch_size,
        lr=lr,
        metric=args.metric,
        stat_window=stat_window,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    proposed_results = run_experiment(
        model_name=args.model,
        pool_type="morph",
        use_morph=True,
        name=f"Propuesto ({args.model.upper()} + MorphPool RGB)",
        epochs=args.epochs,
        batch_size=batch_size,
        lr=lr,
        metric=args.metric,
        stat_window=stat_window,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )

    report_path = generate_html_report(
        baseline=baseline_results,
        proposed=proposed_results,
        config={
            "dataset": "ISIC Dermoscopy (Benign vs Malignant)",
            "model": args.model,
            "pool": args.pool,
            "batch_size": batch_size,
            "epochs": args.epochs,
            "lr": lr,
            "metric": args.metric,
            "stat_window": args.stat_window,
            "device": str(DEVICE),
            "resolution": resolution,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        output_path=args.output
    )

    print("\n" + "="*65)
    print(" [OK] INFORME CIENTÍFICO GENERADO CON ÉXITO")
    print(f" Archivo HTML: {report_path}")
    print(" Puedes abrirlo en tu navegador web para ver los resultados.")
    print("="*65 + "\n")