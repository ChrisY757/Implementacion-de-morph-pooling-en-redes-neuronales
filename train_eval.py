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
from models import CNNClassifier
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
        default=64,
        help="Tamaño del lote / batch size (default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Tasa de aprendizaje / Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="SMO",
        choices=["SMO", "VAR"],
        help="Métrica estadística para el descriptor morfológico (default: SMO)"
    )
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
        help="Ruta/nombre del informe HTML de salida (default: reporte_experimento_melanoma.html)"
    )
    return parser.parse_args()

# 1. Dataset ultra rápido que precarga todo en memoria RAM
class FastRAMDataset(Dataset):
    def __init__(self, folder_path, transform):
        print(f"Cargando {folder_path} directamente a memoria RAM...")
        raw_dataset = datasets.ImageFolder(folder_path)
        self.images = []
        self.labels = []
        
        for img, label in raw_dataset:
            self.images.append(transform(img))
            self.labels.append(label)
            
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        print(f"--> {len(self.labels)} imágenes listadas en RAM.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Configuración base de rutas y dispositivo
DATA_DIR = "./archive"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reducción a 112x112 para acelerar 4 veces el cálculo
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

def run_experiment(use_morph, name, epochs, batch_size, lr, metric, train_loader, test_loader, train_size):
    set_seed(42)
    model = CNNClassifier(use_morph_pool=use_morph, metric=metric).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print(f"\n================ Ejecutando: {name} ({epochs} épocas) ================")
    start_time = time.time()
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / train_size
        epoch_losses.append(epoch_loss)
        print(f"Época [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Tiempo de entrenamiento: {elapsed:.2f} segundos.")

    # Evaluación en Test
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs).ravel()
    all_preds = np.array(all_preds).ravel()
    all_labels = np.array(all_labels).ravel()

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

    print("\n" + "="*65)
    print(" INICIO DEL EXPERIMENTO COMPARATIVO A/B (MELANOMA)")
    print(f" Épocas: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Métrica: {args.metric}")
    print(f" Dispositivo: {DEVICE}")
    print("="*65)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"\n[!] Error: No se encontraron las carpetas 'train' o 'test' en: {args.data_dir}")
        print("    Asegúrate de que la ruta sea correcta o especifícala con --data-dir <ruta>\n")
        exit(1)

    print("\n=== PREPARANDO DATASETS EN RAM (Precarga de aceleración) ===")
    train_dataset = FastRAMDataset(train_dir, transform=transform)
    test_dataset = FastRAMDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    baseline_results = run_experiment(
        use_morph=False,
        name="Baseline (MaxPool RGB)",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        metric=args.metric,
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=len(train_dataset)
    )

    proposed_results = run_experiment(
        use_morph=True,
        name="Propuesto (MorphPool RGB)",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        metric=args.metric,
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=len(train_dataset)
    )

    report_path = generate_html_report(
        baseline=baseline_results,
        proposed=proposed_results,
        config={
            "dataset": "ISIC Dermoscopy (Benign vs Malignant)",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "metric": args.metric,
            "device": str(DEVICE),
            "resolution": "112x112",
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        output_path=args.output
    )

    print("\n" + "="*65)
    print(" [✓] INFORME CIENTÍFICO GENERADO CON ÉXITO")
    print(f" Archivo HTML: {report_path}")
    print(" Puedes abrirlo en tu navegador web para ver los resultados.")
    print("="*65 + "\n")