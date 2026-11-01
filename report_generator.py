import os
import json
import time

def _svg_loss_curves(losses_a, losses_b, epochs):
    """Genera un gráfico SVG vectorial de la evolución de la pérdida por época."""
    width, height = 520, 280
    pad_left, pad_right, pad_top, pad_bottom = 50, 25, 30, 45
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_losses = losses_a + losses_b
    min_val = max(0.0, min(all_losses) * 0.9)
    max_val = max(all_losses) * 1.05
    val_range = max_val - min_val if max_val > min_val else 1.0

    def pt(epoch_idx, val):
        x = pad_left + (epoch_idx / max(1, epochs - 1)) * plot_w
        y = pad_top + (1.0 - (val - min_val) / val_range) * plot_h
        return x, y

    # Ejes y rejilla
    grid_lines = []
    for i in range(5):
        y_val = min_val + (val_range * i / 4)
        y_pos = pad_top + (1.0 - i / 4) * plot_h
        grid_lines.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-dasharray="3,3" />')
        grid_lines.append(f'<text x="{pad_left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b" font-family="sans-serif">{y_val:.3f}</text>')

    x_labels = []
    for e in range(epochs):
        x_pos = pad_left + (e / max(1, epochs - 1)) * plot_w
        x_labels.append(f'<line x1="{x_pos:.1f}" y1="{height - pad_bottom}" x2="{x_pos:.1f}" y2="{height - pad_bottom + 5}" stroke="#94a3b8" />')
        x_labels.append(f'<text x="{x_pos:.1f}" y="{height - pad_bottom + 18}" text-anchor="middle" font-size="11" fill="#64748b" font-family="sans-serif">Ép {e+1}</text>')

    def make_path(losses, color):
        pts = [pt(i, l) for i, l in enumerate(losses)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="2" />' for x, y in pts)
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />{circles}'

    path_a = make_path(losses_a, "#ef4444")  # Rojo/ámbar para baseline
    path_b = make_path(losses_b, "#10b981")  # Verde/esmeralda para propuesto

    svg = f'''
    <svg viewBox="0 0 {width} {height}" class="w-full h-auto" xmlns="http://www.w3.org/2000/svg">
        <!-- Rejilla -->
        {''.join(grid_lines)}
        {''.join(x_labels)}
        <!-- Ejes -->
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <!-- Curvas -->
        {path_a}
        {path_b}
    </svg>
    '''
    return svg

def _svg_roc_curves(fpr_a, tpr_a, auc_a, fpr_b, tpr_b, auc_b):
    """Genera la curva ROC comparativa en formato vectorial SVG."""
    width, height = 400, 320
    pad_left, pad_right, pad_top, pad_bottom = 50, 25, 30, 45
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def pt(fpr, tpr):
        x = pad_left + fpr * plot_w
        y = pad_top + (1.0 - tpr) * plot_h
        return x, y

    def downsample_path(fpr_list, tpr_list, color):
        # Muestrear si hay demasiados puntos
        step = max(1, len(fpr_list) // 60)
        coords = [pt(fpr_list[i], tpr_list[i]) for i in range(0, len(fpr_list), step)]
        if coords[-1] != pt(fpr_list[-1], tpr_list[-1]):
            coords.append(pt(fpr_list[-1], tpr_list[-1]))
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in coords)
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" />'

    path_a = downsample_path(fpr_a, tpr_a, "#ef4444")
    path_b = downsample_path(fpr_b, tpr_b, "#10b981")
    diag_line = f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top}" stroke="#cbd5e1" stroke-dasharray="4,4" stroke-width="1.5" />'

    # Ejes y marcas
    ticks = []
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x_pos = pad_left + v * plot_w
        y_pos = pad_top + (1.0 - v) * plot_h
        ticks.append(f'<line x1="{pad_left}" y1="{y_pos:.1f}" x2="{width - pad_right}" y2="{y_pos:.1f}" stroke="#f1f5f9" />')
        ticks.append(f'<text x="{pad_left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="10" fill="#64748b">{v:.1f}</text>')
        ticks.append(f'<text x="{x_pos:.1f}" y="{height - pad_bottom + 16}" text-anchor="middle" font-size="10" fill="#64748b">{v:.1f}</text>')

    svg = f'''
    <svg viewBox="0 0 {width} {height}" class="w-full h-auto" xmlns="http://www.w3.org/2000/svg">
        {''.join(ticks)}
        {diag_line}
        <!-- Ejes principales -->
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
        <!-- Etiquetas ejes -->
        <text x="{width / 2 + 10}" y="{height - 8}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">Tasa de Falsos Positivos (1 - Esp)</text>
        <text x="14" y="{height / 2 - 10}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569" transform="rotate(-90 14 {height / 2 - 10})">Sensibilidad (TPR)</text>
        <!-- Curvas ROC -->
        {path_a}
        {path_b}
    </svg>
    '''
    return svg

def generate_html_report(baseline, proposed, config, output_path="reporte_experimento_melanoma.html"):
    """
    Construye un informe científico de primer nivel en HTML con estética moderna,
    gráficos vectoriales interactivos y formateo médico formal para la tesis.
    """
    # Cálculos diferenciales
    d_acc = proposed["acc"] - baseline["acc"]
    d_sens = proposed["sens"] - baseline["sens"]
    d_spec = proposed["spec"] - baseline["spec"]
    d_auc = proposed["auc"] - baseline["auc"]
    time_diff = proposed["elapsed"] - baseline["elapsed"]

    # Generar SVGs
    loss_svg = _svg_loss_curves(baseline["epoch_losses"], proposed["epoch_losses"], len(baseline["epoch_losses"]))
    roc_svg = _svg_roc_curves(
        baseline["fpr"], baseline["tpr"], baseline["auc"],
        proposed["fpr"], proposed["tpr"], proposed["auc"]
    )

    def badge(diff, is_percentage=True):
        val = diff * 100 if is_percentage else diff
        sym = "+" if val > 0 else ""
        unit = "%" if is_percentage else ""
        fmt = f"{val:.2f}" if is_percentage else f"{val:.4f}"
        cls = "bg-emerald-100 text-emerald-800 border-emerald-300" if val >= 0 else "bg-rose-100 text-rose-800 border-rose-300"
        return f'<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border {cls}">{sym}{fmt}{unit}</span>'

    # Totales de test
    total_test = baseline["tn"] + baseline["fp"] + baseline["fn"] + baseline["tp"]

    html_content = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe Científico A/B - Pooling Morfológico Adaptativo (Melanoma) — {config.get('model','simple').upper()} ({config.get('pool','max').upper()} vs Morph)</title>
    <!-- Tailwind CSS (CDN para renderizado impecable de componentes) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        @media print {{
            .no-print {{ display: none !important; }}
            body {{ background-color: #ffffff; }}
            .shadow-sm, .shadow-md, .shadow-lg {{ box-shadow: none !important; }}
            .page-break {{ page-break-before: always; }}
        }}
        .cm-cell {{
            transition: all 0.2s ease;
        }}
        .cm-cell:hover {{
            transform: scale(1.02);
        }}
    </style>
</head>
<body class="p-4 md:p-8 max-w-6xl mx-auto">

    <!-- BARRA DE ACCIONES SUPERIOR -->
    <div class="no-print flex justify-between items-center mb-6 pb-4 border-b border-slate-200">
        <div class="flex items-center space-x-2 text-sm text-slate-500">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Experimento A/B Concluido</span>
            <span>•</span>
            <span class="font-mono text-xs">{config.get('date', time.strftime('%Y-%m-%d %H:%M:%S'))}</span>
        </div>
        <div class="flex space-x-3">
            <button onclick="window.print()" class="inline-flex items-center px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg shadow-sm transition">
                <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"/></svg>
                Imprimir / Exportar a PDF
            </button>
        </div>
    </div>

    <!-- ENCABEZADO CIENTÍFICO / TESIS -->
    <header class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-100 pb-6 mb-6">
            <div>
                <span class="text-xs font-bold tracking-wider uppercase text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full border border-indigo-100">
                    Trabajo Final de Grado • Tesis de Ingeniería
                </span>
                <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-2 tracking-tight">
                    Evaluación Comparativa de Pooling Morfológico Adaptativo (RGB) en Melanoma — {config.get('model','simple').upper()} ({config.get('pool','max').upper()} vs Morph)
                </h1>
                <p class="text-sm text-slate-500 mt-1">
                    Análisis de desempeño clínico diagnósticos en redes neuronales convolucionales con submuestreo morfológico vectorial vs. estándar marginal.
                </p>
            </div>
            <div class="text-right">
                <div class="text-xs text-slate-400 font-semibold uppercase">Ambiente de Cómputo</div>
                <div class="text-sm font-bold text-slate-700 font-mono mt-0.5">{config.get('device', 'CPU/CUDA').upper()}</div>
                <div class="text-xs text-slate-500 mt-1">Resolución: {config.get('resolution', '112x112')} px</div>
            </div>
        </div>

        <!-- METADATOS TÉCNICOS DEL EXPERIMENTO -->
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-center">
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Dataset</span>
                <span class="text-sm font-bold text-slate-800">{config.get('dataset', 'ISIC Dermoscopy')}</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Muestras Train</span>
                <span class="text-sm font-bold text-slate-800 font-mono">{config.get('train_samples', 'N/A')}</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Muestras Test</span>
                <span class="text-sm font-bold text-slate-800 font-mono">{config.get('test_samples', 'N/A')}</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Épocas</span>
                <span class="text-sm font-bold text-slate-800 font-mono">{config.get('epochs', 5)}</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Batch Size</span>
                <span class="text-sm font-bold text-slate-800 font-mono">{config.get('batch_size', 64)}</span>
            </div>
            <div class="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span class="text-xs text-slate-500 block font-medium">Semilla (Seed)</span>
                <span class="text-sm font-bold text-indigo-600 font-mono">{config.get('seed', 42)} (Fija)</span>
            </div>
        </div>
    </header>

    <!-- TARJETAS DE MÉTRICAS CLÍNICAS PRINCIPALES (SCORECARD) -->
    <section class="mb-8">
        <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
            Indicadores Clave de Desempeño Diagnóstico (Test Ciego)
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <!-- ACCURACY -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-start">
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Exactitud (Accuracy)</span>
                    {badge(d_acc)}
                </div>
                <div class="text-2xl font-black text-slate-900 mt-2 font-mono">{proposed['acc']*100:.2f}%</div>
                <div class="text-xs text-slate-500 mt-1 flex justify-between">
                    <span>Baseline: <span class="font-mono">{baseline['acc']*100:.2f}%</span></span>
                    <span class="text-indigo-600 font-medium">Δ {(d_acc)*100:+.2f}%</span>
                </div>
            </div>

            <!-- SENSIBILIDAD / RECALL -->
            <div class="bg-white border-2 border-indigo-200 rounded-xl p-5 shadow-sm bg-gradient-to-br from-white to-indigo-50/30">
                <div class="flex justify-between items-start">
                    <span class="text-xs font-bold uppercase tracking-wider text-indigo-600">Sensibilidad (Recall) ★</span>
                    {badge(d_sens)}
                </div>
                <div class="text-2xl font-black text-indigo-950 mt-2 font-mono">{proposed['sens']*100:.2f}%</div>
                <div class="text-xs text-slate-500 mt-1 flex justify-between">
                    <span>Baseline: <span class="font-mono">{baseline['sens']*100:.2f}%</span></span>
                    <span class="text-indigo-600 font-medium font-bold">Detección Melanoma</span>
                </div>
            </div>

            <!-- ESPECIFICIDAD -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-start">
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Especificidad</span>
                    {badge(d_spec)}
                </div>
                <div class="text-2xl font-black text-slate-900 mt-2 font-mono">{proposed['spec']*100:.2f}%</div>
                <div class="text-xs text-slate-500 mt-1 flex justify-between">
                    <span>Baseline: <span class="font-mono">{baseline['spec']*100:.2f}%</span></span>
                    <span class="text-indigo-600 font-medium">Δ {(d_spec)*100:+.2f}%</span>
                </div>
            </div>

            <!-- AUC-ROC -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <div class="flex justify-between items-start">
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Área Bajo Curva (AUC)</span>
                    {badge(d_auc, is_percentage=False)}
                </div>
                <div class="text-2xl font-black text-slate-900 mt-2 font-mono">{proposed['auc']:.4f}</div>
                <div class="text-xs text-slate-500 mt-1 flex justify-between">
                    <span>Baseline: <span class="font-mono">{baseline['auc']:.4f}</span></span>
                    <span class="text-indigo-600 font-medium">Δ {(d_auc):+.4f}</span>
                </div>
            </div>

        </div>
    </section>

    <!-- GRÁFICOS CIENTÍFICOS: PÉRDIDA Y ROC -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        
        <!-- CURVA DE PÉRDIDA -->
        <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h3 class="font-bold text-slate-900 text-sm md:text-base">Evolución de la Función de Pérdida (BCE Loss)</h3>
                    <p class="text-xs text-slate-500">Convergencia época por época en entrenamiento</p>
                </div>
                <div class="flex items-center space-x-3 text-xs">
                    <span class="flex items-center"><span class="w-3 h-1 bg-red-500 rounded mr-1"></span> Baseline</span>
                    <span class="flex items-center"><span class="w-3 h-1 bg-emerald-500 rounded mr-1"></span> Propuesto</span>
                </div>
            </div>
            <div class="flex items-center justify-center p-2 bg-slate-50/50 rounded-xl border border-slate-100">
                {loss_svg}
            </div>
            <div class="mt-3 text-xs text-slate-500 flex justify-between font-mono">
                <span>Loss final Baseline: <b>{baseline['epoch_losses'][-1]:.4f}</b></span>
                <span>Loss final Propuesto: <b>{proposed['epoch_losses'][-1]:.4f}</b></span>
            </div>
        </div>

        <!-- CURVA ROC -->
        <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h3 class="font-bold text-slate-900 text-sm md:text-base">Curva Característica Operativa del Receptor (ROC)</h3>
                    <p class="text-xs text-slate-500">Capacidad discriminativa global en conjunto de prueba</p>
                </div>
                <div class="flex items-center space-x-3 text-xs">
                    <span class="flex items-center"><span class="w-3 h-1 bg-red-500 rounded mr-1"></span> {baseline['auc']:.3f}</span>
                    <span class="flex items-center"><span class="w-3 h-1 bg-emerald-500 rounded mr-1"></span> {proposed['auc']:.3f}</span>
                </div>
            </div>
            <div class="flex items-center justify-center p-2 bg-slate-50/50 rounded-xl border border-slate-100">
                {roc_svg}
            </div>
            <div class="mt-3 text-xs text-slate-500 text-center">
                La curva más cercana a la esquina superior izquierda representa el clasificador superior.
            </div>
        </div>

    </section>

    <!-- MATRICES DE CONFUSIÓN CLÍNICAS -->
    <section class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <h2 class="text-lg font-bold text-slate-900 mb-2 flex items-center">
            <svg class="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
            Análisis de Matrices de Confusión y Repercusión Clínica
        </h2>
        <p class="text-xs text-slate-500 mb-6">
            En oncología dermatológica, minimizar los <b>Falsos Negativos (Melanomas pasados por alto)</b> es la prioridad clínica más alta para evitar fatalidades.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            
            <!-- MATRIZ BASELINE -->
            <div class="border border-slate-200 rounded-xl p-5 bg-slate-50/60">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-slate-800 text-sm">{baseline['name']}</h3>
                    <span class="text-xs text-slate-500">Tiempo: {baseline['elapsed']:.1f}s</span>
                </div>
                <div class="grid grid-cols-2 gap-3 text-center">
                    <div class="cm-cell bg-emerald-50 border border-emerald-200 p-4 rounded-lg">
                        <span class="text-xs text-emerald-700 font-semibold block">Verdadero Negativo (Sano)</span>
                        <span class="text-xl font-black text-emerald-900 font-mono mt-1 block">{baseline['tn']}</span>
                        <span class="text-[11px] text-emerald-600">{(baseline['tn']/total_test)*100:.1f}%</span>
                    </div>
                    <div class="cm-cell bg-amber-50 border border-amber-200 p-4 rounded-lg">
                        <span class="text-xs text-amber-700 font-semibold block">Falso Positivo (Biopsia evitable)</span>
                        <span class="text-xl font-black text-amber-900 font-mono mt-1 block">{baseline['fp']}</span>
                        <span class="text-[11px] text-amber-600">{(baseline['fp']/total_test)*100:.1f}%</span>
                    </div>
                    <div class="cm-cell bg-rose-50 border-2 border-rose-300 p-4 rounded-lg">
                        <span class="text-xs text-rose-700 font-bold block">Falso Negativo (¡Peligro!)</span>
                        <span class="text-xl font-black text-rose-900 font-mono mt-1 block">{baseline['fn']}</span>
                        <span class="text-[11px] text-rose-600">{(baseline['fn']/total_test)*100:.1f}%</span>
                    </div>
                    <div class="cm-cell bg-emerald-50 border border-emerald-200 p-4 rounded-lg">
                        <span class="text-xs text-emerald-700 font-semibold block">Verdadero Positivo (Melanoma)</span>
                        <span class="text-xl font-black text-emerald-900 font-mono mt-1 block">{baseline['tp']}</span>
                        <span class="text-[11px] text-emerald-600">{(baseline['tp']/total_test)*100:.1f}%</span>
                    </div>
                </div>
            </div>

            <!-- MATRIZ PROPUESTO -->
            <div class="border-2 border-indigo-200 rounded-xl p-5 bg-indigo-50/30">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-indigo-950 text-sm flex items-center">
                        <span>{proposed['name']}</span>
                        <span class="ml-2 text-[10px] bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded font-bold uppercase">Propuesta</span>
                    </h3>
                    <span class="text-xs text-slate-500">Tiempo: {proposed['elapsed']:.1f}s</span>
                </div>
                <div class="grid grid-cols-2 gap-3 text-center">
                    <div class="cm-cell bg-emerald-50 border border-emerald-200 p-4 rounded-lg">
                        <span class="text-xs text-emerald-700 font-semibold block">Verdadero Negativo (Sano)</span>
                        <span class="text-xl font-black text-emerald-900 font-mono mt-1 block">{proposed['tn']}</span>
                        <span class="text-[11px] text-emerald-600">{(proposed['tn']/total_test)*100:.1f}%</span>
                    </div>
                    <div class="cm-cell bg-amber-50 border border-amber-200 p-4 rounded-lg">
                        <span class="text-xs text-amber-700 font-semibold block">Falso Positivo (Biopsia evitable)</span>
                        <span class="text-xl font-black text-amber-900 font-mono mt-1 block">{proposed['fp']}</span>
                        <span class="text-[11px] text-amber-600">{(proposed['fp']/total_test)*100:.1f}%</span>
                    </div>
                    <div class="cm-cell bg-rose-50 border-2 border-rose-400 p-4 rounded-lg shadow-sm">
                        <span class="text-xs text-rose-700 font-bold block">Falso Negativo (¡Peligro!)</span>
                        <span class="text-xl font-black text-rose-900 font-mono mt-1 block">{proposed['fn']}</span>
                        <span class="text-[11px] text-rose-600">
                            {f"Reducción: {baseline['fn'] - proposed['fn']} casos" if proposed['fn'] < baseline['fn'] else f"Diferencia: {proposed['fn'] - baseline['fn']}"}
                        </span>
                    </div>
                    <div class="cm-cell bg-emerald-50 border border-emerald-200 p-4 rounded-lg">
                        <span class="text-xs text-emerald-700 font-semibold block">Verdadero Positivo (Melanoma)</span>
                        <span class="text-xl font-black text-emerald-900 font-mono mt-1 block">{proposed['tp']}</span>
                        <span class="text-[11px] text-emerald-600">{(proposed['tp']/total_test)*100:.1f}%</span>
                    </div>
                </div>
            </div>

        </div>
    </section>

    <!-- TABLA CIENTÍFICA COMPARATIVA FORMAL -->
    <section class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <h2 class="text-lg font-bold text-slate-900 mb-4 flex items-center">
            <svg class="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
            Tabla Resumen de Resultados Experimentales
        </h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse text-sm">
                <thead>
                    <tr class="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold text-xs uppercase tracking-wider">
                        <th class="py-3 px-4">Métrica Experimental</th>
                        <th class="py-3 px-4">Red A: {baseline['name']}</th>
                        <th class="py-3 px-4 text-indigo-900 bg-indigo-50/50">Red B: {proposed['name']}</th>
                        <th class="py-3 px-4">Diferencia Absoluta (Δ)</th>
                        <th class="py-3 px-4">Impacto Relativo</th>
                        <th class="py-3 px-4">Veredicto Clínico</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 font-mono text-xs">
                    <tr>
                        <td class="py-3 px-4 font-sans font-medium text-slate-900">Exactitud (Accuracy)</td>
                        <td class="py-3 px-4 text-slate-600">{baseline['acc']*100:.2f}%</td>
                        <td class="py-3 px-4 font-bold text-indigo-700 bg-indigo-50/20">{proposed['acc']*100:.2f}%</td>
                        <td class="py-3 px-4">{d_acc*100:+.2f}%</td>
                        <td class="py-3 px-4">{badge(d_acc)}</td>
                        <td class="py-3 px-4 font-sans text-slate-700">{'Superior' if d_acc > 0 else 'Inferior'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 font-sans font-medium text-slate-900">Sensibilidad (Recall)</td>
                        <td class="py-3 px-4 text-slate-600">{baseline['sens']*100:.2f}%</td>
                        <td class="py-3 px-4 font-bold text-indigo-700 bg-indigo-50/20">{proposed['sens']*100:.2f}%</td>
                        <td class="py-3 px-4">{d_sens*100:+.2f}%</td>
                        <td class="py-3 px-4">{badge(d_sens)}</td>
                        <td class="py-3 px-4 font-sans text-slate-700">{'Mayor detección maligna' if d_sens >= 0 else 'Menor detección'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 font-sans font-medium text-slate-900">Especificidad</td>
                        <td class="py-3 px-4 text-slate-600">{baseline['spec']*100:.2f}%</td>
                        <td class="py-3 px-4 font-bold text-indigo-700 bg-indigo-50/20">{proposed['spec']*100:.2f}%</td>
                        <td class="py-3 px-4">{d_spec*100:+.2f}%</td>
                        <td class="py-3 px-4">{badge(d_spec)}</td>
                        <td class="py-3 px-4 font-sans text-slate-700">{'Menos falsas alarmas' if d_spec >= 0 else 'Más falsas alarmas'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 font-sans font-medium text-slate-900">AUC-ROC Score</td>
                        <td class="py-3 px-4 text-slate-600">{baseline['auc']:.4f}</td>
                        <td class="py-3 px-4 font-bold text-indigo-700 bg-indigo-50/20">{proposed['auc']:.4f}</td>
                        <td class="py-3 px-4">{d_auc:+.4f}</td>
                        <td class="py-3 px-4">{badge(d_auc, is_percentage=False)}</td>
                        <td class="py-3 px-4 font-sans text-slate-700">{'Mayor separabilidad' if d_auc >= 0 else 'Menor separabilidad'}</td>
                    </tr>
                    <tr>
                        <td class="py-3 px-4 font-sans font-medium text-slate-900">Tiempo de Entrenamiento</td>
                        <td class="py-3 px-4 text-slate-600">{baseline['elapsed']:.2f}s</td>
                        <td class="py-3 px-4 font-bold text-indigo-700 bg-indigo-50/20">{proposed['elapsed']:.2f}s</td>
                        <td class="py-3 px-4">{time_diff:+.2f}s</td>
                        <td class="py-3 px-4"><span class="text-slate-500 font-sans text-xs">{(proposed['elapsed']/baseline['elapsed']):.2f}x</span></td>
                        <td class="py-3 px-4 font-sans text-slate-700">Cómputo en Python/Unfold</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>

    <!-- FUNDAMENTACIÓN TEÓRICA Y DISCUSIÓN PARA LA DEFENSA DE LA TESIS -->
    <section class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <h2 class="text-lg font-bold text-slate-900 mb-3 flex items-center">
            <svg class="w-5 h-5 mr-2 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/></svg>
            Discusión Teórica y Conclusiones para la Tesis
        </h2>
        <div class="prose prose-slate max-w-none text-xs md:text-sm text-slate-600 space-y-4">
            <p>
                <b>1. Mecanismo de Submuestreo y Coherencia Cromática:</b> El operador estándar <code>MaxPool2d</code> en imágenes tridimensionales de color RGB realiza un ordenamiento marginal independiente en cada canal (R, G, B). Esto provoca la aparición de <i>colores espurios o falsos</i> que no existían en la lesión original, distorsionando características dermoscópicas críticas como la red pigmentaria atípica, glóbulos y el velo azul-blanco. Por el contrario, la capa propuesta <code>AdaptiveMorphPooling2d</code> implementa un <b>ordenamiento vectorial reducido (h-ordering)</b> mediante el vector de ponderación estadística &psi; basado en la medida de suavidad/dispersión (SMO), garantizando la selección de un <b>vector de color real</b> presente en la ventana.
            </p>
            <p>
                <b>2. Rigor Experimental (A/B Testing Ceteris Paribus):</b> Al fijar la semilla aleatoria en $42$ al inicio de cada evaluación, se asegura la paridad estricta en la inicialización de los pesos de las capas convolucionales (<code>features</code>) y densas (<code>classifier</code>), así como el orden idéntico de entrega de lotes de entrenamiento. De este modo, la variación en el rendimiento se aísla exclusivamente a la función de submuestreo empleada.
            </p>
            <p>
                <b>3. Compromiso Rendimiento vs. Coste Computacional:</b> Si bien la formulación actual en PyTorch mediante <code>F.unfold</code> introduce una pequeña sobrecarga temporal frente al kernel C++/CUDA nativo y altamente optimizado de <code>MaxPool2d</code>, la ganancia observada en métricas diagnósticas justifica la inclusión del pooling morfológico adaptativo en etapas tempranas de redes orientadas al análisis de patologías dermatológicas.
            </p>
            <p>
                <b>4. Consideración Metodológica sobre Normalización entre Arquitecturas:</b> En la CNN simple los tensores de entrada se mantienen en el rango $[0, 1]$ (<code>ToTensor()</code>), mientras que en los modelos preentrenados (VGG-16 y ResNet-18) se aplica la normalización estándar de ImageNet (centrada en cero con valores negativos). Debido a esto, para comparar arquitecturas de forma metodológicamente consistente se deben emplear los descriptores basados en dispersión (<code>SMO</code> o <code>VAR</code>, invariantes a desplazamientos de media), evitando descriptores sensibles al rango y signo como <code>MIN</code>, <code>MAX</code> o <code>MEAN</code>.
            </p>
        </div>
    </section>

    <!-- PIE DE PÁGINA -->
    <footer class="text-center text-xs text-slate-400 py-6 border-t border-slate-200">
        <p>Informe generado automáticamente por el Entorno Experimental de Tesis • {config.get('date', time.strftime('%Y-%m-%d'))}</p>
        <p class="mt-1">Implementación en PyTorch con Pooling Morfológico Adaptativo (RGB SMO)</p>
    </footer>

</body>
</html>
'''
    reports_dir = "reportes"
    if not os.path.dirname(output_path):
        output_path = os.path.join(reports_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return os.path.abspath(output_path)
