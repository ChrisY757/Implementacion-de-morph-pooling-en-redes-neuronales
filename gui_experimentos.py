import os
import sys
import time
import queue
import random
import subprocess
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from morph_pool_layer import normalize_partition, format_domain_label, parse_partition_grid


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_DIR, "reportes")


MODEL_OPTIONS = [
    ("simple", "simple — CNN Clasificadora desde cero (112×112)"),
    ("resnet18", "resnet18 — ResNet-18 Preentrenada ImageNet (224×224)"),
    ("vgg16", "vgg16 — VGG-16 Preentrenada ImageNet (224×224)"),
]

POOL_OPTIONS = [
    ("max", "max — MaxPool2d Estándar (Marginal)"),
    ("avg", "avg — AvgPool2d Estándar (Promedio)"),
]

METRIC_OPTIONS = [
    ("SMO", "SMO — Suavidad Relativa (φ_R)"),
    ("ENT", "ENT — Entropía de Shannon 0-255 (φ_Ent)"),
    ("MEAN", "MEAN — Media Aritmética (φ_Me)"),
    ("VAR", "VAR — Varianza (φ_Var)"),
    ("MO1", "MO1 — Moda Mínima (φ_minMo)"),
    ("MO2", "MO2 — Moda Máxima (φ_maxMo)"),
    ("MIN", "MIN — Mínimo (φ_Min)"),
    ("MAX", "MAX — Máximo (φ_Max)"),
]

PARTITION_OPTIONS = [
    ("W0", "W0 — Estrategia 1: Vecindad Local (sin sub-regiones)"),
    ("W1", "W1 — Estrategia 2: Global (1 sub-región 1×1, imagen completa)"),
    ("W4", "W4 — Estrategia 2: 4 Sub-regiones (2×2, Figs. 2 y 3 del paper)"),
    ("W9", "W9 — Estrategia 2: 9 Sub-regiones (3×3, Figs. 11 y 12 del paper)"),
    ("W16", "W16 — Estrategia 2: 16 Sub-regiones (4×4)"),
    ("W25", "W25 — Estrategia 2: 25 Sub-regiones (5×5, Figs. 7, 11 y 12 del paper)"),
    ("W36", "W36 — Estrategia 2: 36 Sub-regiones (6×6)"),
    ("W49", "W49 — Estrategia 2: 49 Sub-regiones (7×7, Figs. 11 y 12 del paper)"),
    ("W64", "W64 — Estrategia 2: 64 Sub-regiones (8×8)"),
]

WINDOW_OPTIONS = [
    ("2", "2 — 2×2 (Ventana base del pooling)"),
    ("3", "3 — 3×3"),
    ("5", "5 — 5×5"),
    ("7", "7 — 7×7"),
    ("global", "global — Imagen completa (equivale a W1)"),
]

BATCH_OPTIONS = [
    ("auto", "Automático (según arquitectura y GPU/CPU)"),
    ("16", "16"),
    ("32", "32"),
    ("64", "64"),
    ("128", "128"),
]

LR_OPTIONS = [
    "Automático (0.001 simple / 0.0001 preentrenada)",
    "0.001",
    "0.0005",
    "0.0001",
    "0.00005",
]

EPOCHS_PRESETS = ["5", "10", "15", "25", "50", "100", "5 10", "10 25 50"]

SEED_MODES = [
    ("single", "Semilla única (1 corrida comparativa A/B)"),
    ("multi", "Múltiples semillas (Informe estadístico con t-test)"),
]

SEED_COUNT_PRESETS = ["3", "5", "9", "10", "11", "13", "15"]


def count_images_in_folder(folder):
    """Cuenta archivos de imagen dentro de las subcarpetas de clases sin bloquear la interfaz."""
    if not os.path.isdir(folder):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    total = 0
    try:
        with os.scandir(folder) as class_entries:
            for entry in class_entries:
                if entry.is_dir():
                    with os.scandir(entry.path) as files:
                        for f in files:
                            if f.is_file() and os.path.splitext(f.name)[1].lower() in exts:
                                total += 1
    except OSError:
        return 0
    return total


def discover_datasets(base_dir):
    """Detecta automáticamente todas las carpetas del proyecto que contengan 'train' y 'test'."""
    found = []
    try:
        for entry in sorted(os.scandir(base_dir), key=lambda e: e.name.lower()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in ("venv", "__pycache__", "reportes", "presentaciones"):
                continue
            train_p = os.path.join(entry.path, "train")
            test_p = os.path.join(entry.path, "test")
            if os.path.isdir(train_p) and os.path.isdir(test_p):
                found.append(entry.name)
    except OSError:
        pass
    if not found:
        found = ["archive"]
    return found


class MorphPoolExperimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Panel Experimental de Tesis — Adaptive MorphPooling RGB vs MaxPool2d")
        self.root.geometry("1180x820")
        self.root.minsize(980, 700)
        self.root.configure(bg="#f1f5f9")

        self.process = None
        self.log_queue = queue.Queue()
        self.generated_reports = []
        self.start_ts = None
        self._last_was_cr = False
        self._updating_vars = False

        self._configure_styles()
        self._init_variables()
        self._build_layout()
        self._refresh_datasets()
        self._on_any_param_change()
        self._poll_log_queue()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        for theme_candidate in ("vista", "clam", "default"):
            if theme_candidate in style.theme_names():
                style.theme_use(theme_candidate)
                break

        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Bg.TFrame", background="#f1f5f9")
        style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 13, "bold"))
        style.configure("SubHeader.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("SectionTitle.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", background="#ffffff", foreground="#334155", font=("Segoe UI", 9, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 8))
        style.configure("Warn.TLabel", background="#fffbeb", foreground="#92400e", font=("Segoe UI", 8, "bold"))
        style.configure("Badge.TLabel", background="#eef2ff", foreground="#3730a3", font=("Consolas", 9, "bold"))

    def _init_variables(self):
        self.dataset_var = tk.StringVar(value="archive")
        self.model_var = tk.StringVar(value=MODEL_OPTIONS[0][1])
        self.pool_var = tk.StringVar(value=POOL_OPTIONS[0][1])
        self.metric_var = tk.StringVar(value=METRIC_OPTIONS[0][1])
        self.partition_var = tk.StringVar(value=PARTITION_OPTIONS[3][1])  # W9 por defecto para mostrar la nueva opción
        self.window_var = tk.StringVar(value=WINDOW_OPTIONS[0][1])     # 2x2
        self.epochs_var = tk.StringVar(value="5")
        self.batch_var = tk.StringVar(value=BATCH_OPTIONS[0][1])
        self.lr_var = tk.StringVar(value=LR_OPTIONS[0])
        self.seed_mode_var = tk.StringVar(value=SEED_MODES[0][1])
        self.seeds_text_var = tk.StringVar(value="42")
        self.rand_seed_count_var = tk.StringVar(value="9")
        self.auto_output_var = tk.BooleanVar(value=True)
        self.output_var = tk.StringVar(value="reporte_experimento_melanoma.html")

        self.dataset_info_var = tk.StringVar(value="")
        self.domain_summary_var = tk.StringVar(value="")
        self.domain_explain_var = tk.StringVar(value="")
        self.norm_warning_var = tk.StringVar(value="")
        self.cmd_preview_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Listo para ejecutar")
        self.selected_report_var = tk.StringVar(value="")

    def _build_layout(self):
        # Encabezado superior
        header = tk.Frame(self.root, bg="#0f172a", padx=20, pady=12)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Banco de Pruebas A/B — Pooling Morfológico Adaptativo Vectorial (RGB)",
            style="Header.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Configuración interactiva de Arquitectura, Descriptor Estadístico (φ), Partición de Dominio del Paper (W0..W49) e Hiperparámetros",
            style="SubHeader.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        # Contenedor principal dividido en 2 columnas: Formulario (Izquierda) + Consola/Acciones (Derecha)
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=14, pady=12)

        left_outer = ttk.Frame(main_pane, style="Bg.TFrame")
        right_outer = ttk.Frame(main_pane, style="Bg.TFrame")
        main_pane.add(left_outer, weight=5)
        main_pane.add(right_outer, weight=6)

        self._build_form_panel(left_outer)
        self._build_console_panel(right_outer)

    def _build_form_panel(self, parent):
        # Tarjeta con scroll suave por si la pantalla tiene baja resolución vertical
        canvas = tk.Canvas(parent, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas, style="Card.TFrame", padding=(16, 14))

        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0

        # --- SECCIÓN 1: DATASET Y ARQUITECTURA ---
        ttk.Label(form, text="1. Dataset y Arquitectura de Red", style="SectionTitle.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        ttk.Label(form, text="Dataset (-d):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        ds_frame = ttk.Frame(form, style="Card.TFrame")
        ds_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ds_frame.columnconfigure(0, weight=1)

        self.dataset_combo = ttk.Combobox(ds_frame, textvariable=self.dataset_var)
        self.dataset_combo.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        self.dataset_combo.bind("<KeyRelease>", self._on_any_param_change)

        ttk.Button(ds_frame, text="Examinar...", width=11, command=self._browse_dataset).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(ds_frame, text="↻", width=3, command=self._refresh_datasets).grid(row=0, column=2)
        row += 1

        ttk.Label(form, textvariable=self.dataset_info_var, style="Hint.TLabel").grid(
            row=row, column=1, columnspan=2, sticky="w", pady=(0, 6)
        )
        row += 1

        ttk.Label(form, text="Arquitectura (--model):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.model_combo = ttk.Combobox(
            form, textvariable=self.model_var, state="readonly", values=[label for _, label in MODEL_OPTIONS]
        )
        self.model_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        row += 1

        ttk.Label(form, text="Pooling Baseline (--pool):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.pool_combo = ttk.Combobox(
            form, textvariable=self.pool_var, state="readonly", values=[label for _, label in POOL_OPTIONS]
        )
        self.pool_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.pool_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # --- SECCIÓN 2: MORPHPOOLING Y PARTICIÓN DE DOMINIO DEL PAPER ---
        ttk.Label(form, text="2. Configuración Morfológica y Partición de Dominio (Paper)", style="SectionTitle.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        ttk.Label(form, text="Descriptor φ (--metric):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.metric_combo = ttk.Combobox(
            form, textvariable=self.metric_var, state="readonly", values=[label for _, label in METRIC_OPTIONS]
        )
        self.metric_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.metric_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        row += 1

        ttk.Label(form, text="Partición Dominio (-p):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.partition_combo = ttk.Combobox(
            form, textvariable=self.partition_var, state="readonly", values=[label for _, label in PARTITION_OPTIONS]
        )
        self.partition_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.partition_combo.bind("<<ComboboxSelected>>", self._on_partition_changed)
        row += 1

        ttk.Label(form, text="Ventana B (--stat-window):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.window_combo = ttk.Combobox(
            form, textvariable=self.window_var, state="readonly", values=[label for _, label in WINDOW_OPTIONS]
        )
        self.window_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.window_combo.bind("<<ComboboxSelected>>", self._on_window_changed)
        row += 1

        # Recuadro explicativo en vivo de la nomenclatura del paper
        badge_box = tk.Frame(form, bg="#eef2ff", highlightbackground="#c7d2fe", highlightthickness=1, padx=10, pady=8)
        badge_box.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 4))
        ttk.Label(badge_box, textvariable=self.domain_summary_var, style="Badge.TLabel").pack(anchor="w")
        tk.Label(
            badge_box,
            textvariable=self.domain_explain_var,
            bg="#eef2ff",
            fg="#4338ca",
            font=("Segoe UI", 8),
            wraplength=430,
            justify="left"
        ).pack(anchor="w", pady=(2, 0))
        row += 1

        # Banner de advertencia de normalización ImageNet si aplica
        self.warn_frame = tk.Frame(form, bg="#fffbeb", highlightbackground="#fde68a", highlightthickness=1, padx=8, pady=6)
        self.warn_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 4))
        tk.Label(
            self.warn_frame,
            textvariable=self.norm_warning_var,
            bg="#fffbeb",
            fg="#92400e",
            font=("Segoe UI", 8),
            wraplength=430,
            justify="left"
        ).pack(anchor="w")
        self.warn_frame.grid_remove()
        row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # --- SECCIÓN 3: HIPERPARÁMETROS Y SEMILLAS ---
        ttk.Label(form, text="3. Hiperparámetros, Épocas y Semillas (Seeds)", style="SectionTitle.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1

        ttk.Label(form, text="Épocas (-e / --epochs-list):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.epochs_combo = ttk.Combobox(form, textvariable=self.epochs_var, values=EPOCHS_PRESETS)
        self.epochs_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.epochs_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        self.epochs_combo.bind("<KeyRelease>", self._on_any_param_change)
        row += 1

        ttk.Label(form, text="Batch Size (-b):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.batch_combo = ttk.Combobox(
            form, textvariable=self.batch_var, state="readonly", values=[label for _, label in BATCH_OPTIONS]
        )
        self.batch_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.batch_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        row += 1

        ttk.Label(form, text="Learning Rate (--lr):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.lr_combo = ttk.Combobox(form, textvariable=self.lr_var, values=LR_OPTIONS)
        self.lr_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.lr_combo.bind("<<ComboboxSelected>>", self._on_any_param_change)
        self.lr_combo.bind("<KeyRelease>", self._on_any_param_change)
        row += 1

        ttk.Label(form, text="Modo de Semillas:", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.seed_mode_combo = ttk.Combobox(
            form, textvariable=self.seed_mode_var, state="readonly", values=[label for _, label in SEED_MODES]
        )
        self.seed_mode_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.seed_mode_combo.bind("<<ComboboxSelected>>", self._on_seed_mode_changed)
        row += 1

        ttk.Label(form, text="Semilla(s) (-s / --seeds):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        seeds_frame = ttk.Frame(form, style="Card.TFrame")
        seeds_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        seeds_frame.columnconfigure(0, weight=1)

        self.seeds_entry = ttk.Entry(seeds_frame, textvariable=self.seeds_text_var)
        self.seeds_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.seeds_entry.bind("<KeyRelease>", self._on_any_param_change)

        self.seed_count_combo = ttk.Combobox(
            seeds_frame, textvariable=self.rand_seed_count_var, state="readonly", values=SEED_COUNT_PRESETS, width=4
        )
        self.seed_count_combo.grid(row=0, column=1, padx=(0, 4))
        self.gen_seeds_btn = ttk.Button(seeds_frame, text="Generar N", width=10, command=self._generate_random_seeds)
        self.gen_seeds_btn.grid(row=0, column=2)
        row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # --- SECCIÓN 4: ARCHIVO DE SALIDA ---
        ttk.Label(form, text="4. Informe HTML de Salida", style="SectionTitle.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        row += 1

        self.auto_out_chk = ttk.Checkbutton(
            form,
            text="Generar nombre descriptivo automáticamente según parámetros",
            variable=self.auto_output_var,
            command=self._on_auto_output_toggled
        )
        self.auto_out_chk.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        row += 1

        ttk.Label(form, text="Archivo (-o):", style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        self.output_entry = ttk.Entry(form, textvariable=self.output_var, state="readonly")
        self.output_entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self.output_entry.bind("<KeyRelease>", self._on_any_param_change)
        row += 1

        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=1)

    def _build_console_panel(self, parent):
        # Panel superior: Vista previa de comando + Botones de control
        ctrl_card = tk.Frame(parent, bg="#ffffff", highlightbackground="#cbd5e1", highlightthickness=1, padx=14, pady=12)
        ctrl_card.pack(fill="x", pady=(0, 10))

        ttk.Label(ctrl_card, text="Comando CLI Equivalente:", style="Field.TLabel").pack(anchor="w")

        cmd_row = tk.Frame(ctrl_card, bg="#ffffff")
        cmd_row.pack(fill="x", pady=(4, 10))
        cmd_entry = tk.Entry(
            cmd_row,
            textvariable=self.cmd_preview_var,
            font=("Consolas", 9),
            bg="#f8fafc",
            fg="#0f172a",
            readonlybackground="#f8fafc",
            relief="solid",
            bd=1,
            state="readonly"
        )
        cmd_entry.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 6))
        ttk.Button(cmd_row, text="Copiar CLI", command=self._copy_command).pack(side="right")

        btn_row = tk.Frame(ctrl_card, bg="#ffffff")
        btn_row.pack(fill="x")

        self.run_btn = tk.Button(
            btn_row,
            text="▶  Ejecutar Experimento A/B",
            command=self._start_experiment,
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=7,
            cursor="hand2"
        )
        self.run_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_row,
            text="⏹  Detener",
            command=self._stop_experiment,
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            padx=12,
            pady=7,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=(0, 8))

        ttk.Button(btn_row, text="Limpiar Consola", command=self._clear_console).pack(side="left")
        ttk.Button(btn_row, text="Abrir Carpeta Reportes", command=self._open_reports_folder).pack(side="right")

        # Selector y botón para abrir reportes HTML recién generados
        rep_row = tk.Frame(ctrl_card, bg="#ffffff")
        rep_row.pack(fill="x", pady=(10, 0))
        ttk.Label(rep_row, text="Informes generados:", style="Hint.TLabel").pack(side="left", padx=(0, 6))
        self.reports_combo = ttk.Combobox(rep_row, textvariable=self.selected_report_var, state="readonly")
        self.reports_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.open_report_btn = ttk.Button(
            rep_row, text="Abrir Informe en Navegador", command=self._open_selected_report, state="disabled"
        )
        self.open_report_btn.pack(side="right")

        # Panel inferior: Consola en vivo
        console_card = tk.Frame(parent, bg="#0f172a", highlightbackground="#334155", highlightthickness=1)
        console_card.pack(fill="both", expand=True)

        status_bar = tk.Frame(console_card, bg="#1e293b", padx=12, pady=6)
        status_bar.pack(fill="x")
        tk.Label(
            status_bar, text="Consola de Ejecución en Tiempo Real", bg="#1e293b", fg="#e2e8f0", font=("Segoe UI", 9, "bold")
        ).pack(side="left")
        tk.Label(
            status_bar, textvariable=self.status_var, bg="#1e293b", fg="#38bdf8", font=("Consolas", 9)
        ).pack(side="right")

        text_frame = tk.Frame(console_card, bg="#0f172a")
        text_frame.pack(fill="both", expand=True, padx=2, pady=2)

        self.console_text = tk.Text(
            text_frame,
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            wrap="word",
            relief="flat",
            padx=10,
            pady=8,
            state="disabled"
        )
        c_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=c_scroll.set)
        self.console_text.pack(side="left", fill="both", expand=True)
        c_scroll.pack(side="right", fill="y")

    # --- Helpers para leer claves seleccionadas ---
    @staticmethod
    def _extract_code(selected_label, options_list, fallback=""):
        for code, label in options_list:
            if selected_label == label or selected_label == code:
                return code
        return selected_label.split(" ")[0].strip() if selected_label else fallback

    def _get_current_params(self):
        dataset = self.dataset_var.get().strip() or "archive"
        model = self._extract_code(self.model_var.get(), MODEL_OPTIONS, "simple")
        pool = self._extract_code(self.pool_var.get(), POOL_OPTIONS, "max")
        metric = self._extract_code(self.metric_var.get(), METRIC_OPTIONS, "SMO")
        partition = self._extract_code(self.partition_var.get(), PARTITION_OPTIONS, "W0")
        window = self._extract_code(self.window_var.get(), WINDOW_OPTIONS, "2")
        batch = self._extract_code(self.batch_var.get(), BATCH_OPTIONS, "auto")
        seed_mode = self._extract_code(self.seed_mode_var.get(), SEED_MODES, "single")

        lr_raw = self.lr_var.get().strip()
        lr = None if lr_raw.lower().startswith("autom") or not lr_raw else lr_raw

        ep_tokens = [t for t in self.epochs_var.get().replace(",", " ").split() if t.strip()]
        if not ep_tokens:
            ep_tokens = ["5"]

        seed_tokens = [t for t in self.seeds_text_var.get().replace(",", " ").split() if t.strip()]
        if not seed_tokens:
            seed_tokens = ["42"]

        return {
            "dataset": dataset,
            "model": model,
            "pool": pool,
            "metric": metric,
            "partition": partition,
            "window": window,
            "batch": batch,
            "lr": lr,
            "epochs": ep_tokens,
            "seed_mode": seed_mode,
            "seeds": seed_tokens,
        }

    # --- Eventos dinámicos de sincronización ---
    def _on_partition_changed(self, _event=None):
        if self._updating_vars:
            return
        self._updating_vars = True
        try:
            part_code = self._extract_code(self.partition_var.get(), PARTITION_OPTIONS, "W0")
            win_code = self._extract_code(self.window_var.get(), WINDOW_OPTIONS, "2")
            if part_code == "W1":
                self.window_var.set(WINDOW_OPTIONS[-1][1])  # global
            elif win_code == "global":
                self.window_var.set(WINDOW_OPTIONS[0][1])   # volver a 2x2
        finally:
            self._updating_vars = False
        self._on_any_param_change()

    def _on_window_changed(self, _event=None):
        if self._updating_vars:
            return
        self._updating_vars = True
        try:
            part_code = self._extract_code(self.partition_var.get(), PARTITION_OPTIONS, "W0")
            win_code = self._extract_code(self.window_var.get(), WINDOW_OPTIONS, "2")
            if win_code == "global" and part_code != "W1":
                self.partition_var.set(PARTITION_OPTIONS[1][1])  # W1
            elif win_code != "global" and part_code == "W1":
                self.partition_var.set(PARTITION_OPTIONS[0][1])  # W0
        finally:
            self._updating_vars = False
        self._on_any_param_change()

    def _on_seed_mode_changed(self, _event=None):
        mode = self._extract_code(self.seed_mode_var.get(), SEED_MODES, "single")
        if mode == "single":
            tokens = [t for t in self.seeds_text_var.get().replace(",", " ").split() if t.strip()]
            self.seeds_text_var.set(tokens[0] if tokens else "42")
        else:
            tokens = [t for t in self.seeds_text_var.get().replace(",", " ").split() if t.strip()]
            if len(tokens) <= 1:
                self._generate_random_seeds()
                return
        self._on_any_param_change()

    def _generate_random_seeds(self):
        try:
            n = int(self.rand_seed_count_var.get())
        except ValueError:
            n = 9
        sampled = random.sample(range(10, 1000), k=min(n, 900))
        self.seed_mode_var.set(SEED_MODES[1][1])
        self.seeds_text_var.set(" ".join(str(s) for s in sampled))
        self._on_any_param_change()

    def _on_auto_output_toggled(self):
        if self.auto_output_var.get():
            self.output_entry.configure(state="readonly")
        else:
            self.output_entry.configure(state="normal")
        self._on_any_param_change()

    def _refresh_datasets(self):
        datasets_found = discover_datasets(PROJECT_DIR)
        self.dataset_combo["values"] = datasets_found
        if self.dataset_var.get() not in datasets_found and not os.path.isdir(self.dataset_var.get()):
            self.dataset_var.set(datasets_found[0])
        self._update_dataset_info()

    def _browse_dataset(self):
        chosen = filedialog.askdirectory(initialdir=PROJECT_DIR, title="Seleccionar carpeta de Dataset (con 'train' y 'test')")
        if chosen:
            rel = os.path.relpath(chosen, PROJECT_DIR)
            self.dataset_var.set(rel if not rel.startswith("..") else chosen)
            self._on_any_param_change()

    def _update_dataset_info(self):
        ds = self.dataset_var.get().strip()
        ds_path = ds if os.path.isabs(ds) else os.path.join(PROJECT_DIR, ds)
        train_p = os.path.join(ds_path, "train")
        test_p = os.path.join(ds_path, "test")
        if os.path.isdir(train_p) and os.path.isdir(test_p):
            n_tr = count_images_in_folder(train_p)
            n_te = count_images_in_folder(test_p)
            self.dataset_info_var.set(f"✔ Válido: {n_tr:,} train  |  {n_te:,} test  ({n_tr + n_te:,} imágenes totales)")
        else:
            self.dataset_info_var.set("⚠ No se detectan subcarpetas 'train/' y 'test/' en esta ruta.")

    def _on_any_param_change(self, _event=None):
        self._update_dataset_info()
        p = self._get_current_params()

        part_norm = normalize_partition(p["partition"], p["window"])
        label_paper = format_domain_label(part_norm, p["window"])
        short_code = label_paper.split(" ")[0]
        self.domain_summary_var.set(f"Configuración activa: {p['metric']} · {label_paper}")

        # Sincronizar estado visual del selector de ventana según si usa lupa (W0) o sub-regiones (W1..W64)
        if part_norm == "W0":
            self.window_combo.configure(state="readonly")
            b_sz = p["window"]
            self.domain_explain_var.set(
                f"Estrategia 1 (Vecindad): el vector de pesos ψ se extrae de la ventana local {b_sz}×{b_sz} alrededor de cada celda de pooling."
            )
        elif part_norm == "W1":
            self.window_combo.configure(state="disabled")
            self.domain_explain_var.set(
                "Estrategia 2 (Caso Global W1): la lupa queda desactivada porque se usa TODA la imagen completa (1×1) para calcular ψ."
            )
        else:
            self.window_var.set(WINDOW_OPTIONS[0][1])
            self.window_combo.configure(state="disabled")
            m_g, n_g = parse_partition_grid(part_norm)
            self.domain_explain_var.set(
                f"Estrategia 2 (Sub-regiones {part_norm}): la lupa 5×5/7×7 NO se usa. Se usa TODA la sub-región fija (de {m_g}×{n_g}) "
                f"para calcular ψ en el pooling 2×2 (uniendo sub-regiones vecinas si el 2×2 pisa el borde)."
            )

        # Advertencia metodológica ImageNet
        if p["model"] != "simple" and p["metric"] not in ("SMO", "VAR", "ENT"):
            self.norm_warning_var.set(
                f"⚠ Nota metodológica: {p['model'].upper()} normaliza con ImageNet (valores negativos). "
                f"Se recomienda priorizar SMO, VAR o ENT para invarianza entre arquitecturas."
            )
            self.warn_frame.grid()
        else:
            self.warn_frame.grid_remove()

        # Nombre automático del HTML
        if self.auto_output_var.get():
            ds_clean = os.path.basename(os.path.normpath(p["dataset"])).replace(" ", "_")
            ep_tag = "-".join(p["epochs"]) + "ep"
            pool_tag = "" if p["pool"] == "max" else f"_vs_{p['pool']}"
            auto_name = f"{p['model']}_{p['metric']}_{short_code}_{ds_clean}_{ep_tag}{pool_tag}.html"
            self.output_var.set(auto_name)

        # Construir comando CLI
        cmd_args = self._build_cli_args(p)
        quoted = [f'"{a}"' if " " in a else a for a in ["python", "train_eval.py"] + cmd_args]
        self.cmd_preview_var.set(" ".join(quoted))

    def _build_cli_args(self, p):
        args = [
            "--model", p["model"],
            "--pool", p["pool"],
            "--metric", p["metric"],
            "--partition", p["partition"],
            "--stat-window", p["window"],
            "-d", p["dataset"],
        ]
        if len(p["epochs"]) == 1:
            args += ["-e", p["epochs"][0]]
        else:
            args += ["--epochs-list"] + p["epochs"]

        if p["batch"] != "auto":
            args += ["-b", p["batch"]]

        if p["lr"] is not None:
            args += ["--lr", p["lr"]]

        if p["seed_mode"] == "single" and len(p["seeds"]) == 1:
            args += ["-s", p["seeds"][0]]
        else:
            args += ["--seeds"] + p["seeds"]

        out_name = self.output_var.get().strip() or "reporte_experimento_melanoma.html"
        args += ["-o", out_name]
        return args

    # --- Acciones de consola y subproceso ---
    def _copy_command(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.cmd_preview_var.get())
        self.status_var.set("Comando copiado al portapapeles")

    def _clear_console(self):
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")
        self._last_was_cr = False

    def _append_console_chunk(self, chunk):
        self.console_text.configure(state="normal")
        i = 0
        n = len(chunk)
        while i < n:
            ch = chunk[i]
            if ch == "\r":
                if i + 1 < n and chunk[i + 1] == "\n":
                    self.console_text.insert("end", "\n")
                    self._last_was_cr = False
                    i += 2
                    continue
                self._last_was_cr = True
                i += 1
            elif ch == "\n":
                self.console_text.insert("end", "\n")
                self._last_was_cr = False
                i += 1
            else:
                if self._last_was_cr:
                    # Sobrescribir la última línea de progreso
                    self.console_text.delete("end-1c linestart", "end-1c")
                    self._last_was_cr = False
                # Leer tramo continuo hasta el siguiente \r o \n
                j = i + 1
                while j < n and chunk[j] not in ("\r", "\n"):
                    j += 1
                seg = chunk[i:j]
                self.console_text.insert("end", seg)
                i = j

        self.console_text.see("end")
        self.console_text.configure(state="disabled")

        # Capturar rutas de reportes HTML generados
        for line in chunk.replace("\r", "\n").splitlines():
            for prefix in ("Archivo HTML Agregado:", "Archivo HTML:"):
                if prefix in line:
                    path = line.split(prefix, 1)[1].strip()
                    if path and os.path.isfile(path) and path not in self.generated_reports:
                        self.generated_reports.insert(0, path)
                        self.reports_combo["values"] = self.generated_reports
                        self.selected_report_var.set(path)
                        self.open_report_btn.configure(state="normal")

    def _validate_inputs(self, p):
        try:
            for e in p["epochs"]:
                if int(e) < 1:
                    raise ValueError
        except ValueError:
            messagebox.showerror("Épocas inválidas", "Ingrese números enteros positivos para las épocas (ej. 5, 50 o '5 10').")
            return False

        try:
            for s in p["seeds"]:
                int(s)
        except ValueError:
            messagebox.showerror("Semillas inválidas", "Las semillas deben ser números enteros separados por espacios (ej. '42' o '42 108 212').")
            return False

        if p["lr"] is not None:
            try:
                if float(p["lr"]) <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Learning Rate inválido", "La tasa de aprendizaje debe ser un número positivo (ej. 0.001 o 0.0001).")
                return False

        return True

    def _start_experiment(self):
        if self.process is not None and self.process.poll() is None:
            return

        p = self._get_current_params()
        if not self._validate_inputs(p):
            return

        cli_args = self._build_cli_args(p)
        cmd = [sys.executable, "-u", os.path.join(PROJECT_DIR, "train_eval.py")] + cli_args

        self._append_console_chunk(f"\n>>> Ejecutando: {' '.join(cmd)}\n")
        self.start_ts = time.time()
        self.status_var.set("Entrenando... (0.0s)")
        self.run_btn.configure(state="disabled", bg="#94a3b8")
        self.stop_btn.configure(state="normal")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        def worker():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    bufsize=0
                )
                while True:
                    raw = self.process.stdout.read(256)
                    if not raw:
                        break
                    text = raw.decode("utf-8", errors="replace")
                    self.log_queue.put(("out", text))
                rc = self.process.wait()
                self.log_queue.put(("done", rc))
            except Exception as exc:
                self.log_queue.put(("err", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_experiment(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._append_console_chunk("\n[!] Ejecución detenida manualmente por el usuario.\n")
            self.status_var.set("Ejecución cancelada")

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "out":
                    self._append_console_chunk(payload)
                elif kind == "err":
                    self._append_console_chunk(f"\n[ERROR] {payload}\n")
                    self._on_process_finished(-1)
                elif kind == "done":
                    self._on_process_finished(payload)
        except queue.Empty:
            pass

        if self.process is not None and self.process.poll() is None and self.start_ts is not None:
            elapsed = time.time() - self.start_ts
            self.status_var.set(f"Ejecutando... ({elapsed:.1f}s)")

        self.root.after(60, self._poll_log_queue)

    def _on_process_finished(self, return_code):
        elapsed = time.time() - self.start_ts if self.start_ts else 0.0
        self.process = None
        self.run_btn.configure(state="normal", bg="#16a34a")
        self.stop_btn.configure(state="disabled")
        if return_code == 0:
            self.status_var.set(f"✔ Completado en {elapsed:.1f}s")
        elif self.status_var.get() != "Ejecución cancelada":
            self.status_var.set(f"✖ Finalizado con código {return_code}")

    def _open_reports_folder(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(REPORTS_DIR)
        else:
            webbrowser.open(f"file://{REPORTS_DIR}")

    def _open_selected_report(self):
        path = self.selected_report_var.get().strip()
        if path and os.path.isfile(path):
            webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
        else:
            messagebox.showwarning("Archivo no encontrado", "Seleccione un informe HTML válido.")


def main():
    root = tk.Tk()
    MorphPoolExperimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
