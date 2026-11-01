import os
import json
import time
import math
import statistics as st
import numpy as np

try:
    from scipy.stats import ttest_rel as _scipy_ttest_rel
except ImportError:
    _scipy_ttest_rel = None


def _paired_ttest(prop_vals, base_vals):
    """
    Calcula el t-test pareado (prop_vals vs base_vals).
    Usa scipy.stats.ttest_rel si está disponible; de lo contrario, aplica el cálculo exacto.
    """
    n = len(prop_vals)
    if n < 2:
        return 0.0, 1.0

    diffs = np.asarray(prop_vals, dtype=np.float64) - np.asarray(base_vals, dtype=np.float64)
    if np.allclose(diffs, diffs[0]):
        if abs(diffs[0]) < 1e-12:
            return 0.0, 1.0
        return float("inf") if diffs[0] > 0 else float("-inf"), 0.0

    if _scipy_ttest_rel is not None:
        res = _scipy_ttest_rel(prop_vals, base_vals)
        t_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
        return t_stat, p_val

    # Fallback analítico si scipy no está instalado
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    if std_d < 1e-15:
        return 0.0, 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    # Aproximación normal bilateral para p-value en caso de no contar con scipy
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return t_stat, p_val


def aggregate_report(results, output_path="reporte_agregado_semillas.html", config=None):
    """
    Genera un informe HTML científico agregado a partir de múltiples semillas (seeds),
    calculando media, desviación estándar y p-value mediante t-test pareado.

    Parámetros:
        results: lista de diccionarios [{"seed": 42, "epochs": 10, "baseline": {...}, "proposed": {...}}, ...]
        output_path: nombre o ruta del archivo HTML de salida (se guarda en 'reportes/')
        config: diccionario opcional con metadatos del experimento
    """
    config = config or {}
    reports_dir = "reportes"
    if not os.path.dirname(output_path):
        output_path = os.path.join(reports_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar también respaldo en JSON
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"config": config, "results": results}, jf, indent=2)

    n_seeds = len(results)
    seeds_str = ", ".join(str(r.get("seed", "?")) for r in results)

    metrics_spec = [
        ("AUC-ROC", "auc", False),
        ("Accuracy (Exactitud)", "acc", True),
        ("Sensibilidad (Recall)", "sens", True),
        ("Especificidad", "spec", True),
    ]

    summary_rows = []
    for label, key, is_pct in metrics_spec:
        scale = 100.0 if is_pct else 1.0
        unit = "%" if is_pct else ""
        fmt = ".2f" if is_pct else ".4f"

        base_vals = [r["baseline"][key] * scale for r in results]
        prop_vals = [r["proposed"][key] * scale for r in results]
        deltas = [p - b for p, b in zip(prop_vals, base_vals)]

        mean_base = st.mean(base_vals)
        std_base = st.stdev(base_vals) if n_seeds > 1 else 0.0
        mean_prop = st.mean(prop_vals)
        std_prop = st.stdev(prop_vals) if n_seeds > 1 else 0.0
        mean_delta = st.mean(deltas)
        std_delta = st.stdev(deltas) if n_seeds > 1 else 0.0

        t_stat, p_value = _paired_ttest(prop_vals, base_vals)

        if n_seeds < 2:
            verdict = '<span class="px-2.5 py-1 rounded text-xs font-semibold bg-slate-100 text-slate-600">Se requieren ≥2 seeds</span>'
        elif p_value < 0.05 and mean_delta > 0:
            verdict = '<span class="px-2.5 py-1 rounded text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">✅ Significativo (Ganancia)</span>'
        elif p_value < 0.05 and mean_delta < 0:
            verdict = '<span class="px-2.5 py-1 rounded text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-300">❌ Significativo (Inferior)</span>'
        else:
            verdict = '<span class="px-2.5 py-1 rounded text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-300">⚠️ No significativo</span>'

        delta_cls = "text-emerald-700 font-bold" if mean_delta >= 0 else "text-rose-700 font-bold"
        p_cls = "text-emerald-700 font-bold" if (n_seeds > 1 and p_value < 0.05) else "text-slate-600"

        summary_rows.append(f"""
        <tr class="hover:bg-slate-50/80">
            <td class="py-3.5 px-4 font-sans font-semibold text-slate-900">{label}</td>
            <td class="py-3.5 px-4 text-slate-600">{mean_base:{fmt}}{unit} ± {std_base:{fmt}}{unit}</td>
            <td class="py-3.5 px-4 font-semibold text-indigo-700 bg-indigo-50/30">{mean_prop:{fmt}}{unit} ± {std_prop:{fmt}}{unit}</td>
            <td class="py-3.5 px-4 {delta_cls}">{mean_delta:+{fmt}}{unit}</td>
            <td class="py-3.5 px-4 text-slate-600">±{std_delta:{fmt}}{unit}</td>
            <td class="py-3.5 px-4 {p_cls}">{p_value:.4f}</td>
            <td class="py-3.5 px-4 font-sans">{verdict}</td>
        </tr>
        """)

    seed_rows = []
    for r in results:
        s = r.get("seed", "N/A")
        b = r["baseline"]
        p = r["proposed"]
        d_auc = p["auc"] - b["auc"]
        d_acc = (p["acc"] - b["acc"]) * 100.0
        d_sens = (p["sens"] - b["sens"]) * 100.0
        d_spec = (p["spec"] - b["spec"]) * 100.0
        seed_rows.append(f"""
        <tr class="hover:bg-slate-50/80">
            <td class="py-2.5 px-4 font-bold text-indigo-600">{s}</td>
            <td class="py-2.5 px-4">{b['auc']:.4f} → <b>{p['auc']:.4f}</b> ({d_auc:+.4f})</td>
            <td class="py-2.5 px-4">{b['acc']*100:.2f}% → <b>{p['acc']*100:.2f}%</b> ({d_acc:+.2f}%)</td>
            <td class="py-2.5 px-4">{b['sens']*100:.2f}% → <b>{p['sens']*100:.2f}%</b> ({d_sens:+.2f}%)</td>
            <td class="py-2.5 px-4">{b['spec']*100:.2f}% → <b>{p['spec']*100:.2f}%</b> ({d_spec:+.2f}%)</td>
        </tr>
        """)

    model_title = config.get("model", "simple").upper()
    pool_title = f"{config.get('pool', 'max').upper()} vs Morph"
    metric_name = config.get("metric", "SMO")
    stat_win = config.get("stat_window", "2")
    epochs_val = config.get("epochs", "N/A")
    dataset_name = config.get("dataset", "ISIC Dermoscopy")
    df_val = max(0, n_seeds - 1)
    engine_label = "scipy.stats.ttest_rel (Distribución t de Student exacta)" if _scipy_ttest_rel is not None else "Fallback analítico (ADVERTENCIA: instalar scipy para distribución t exacta)"

    power_warning_html = ""
    if n_seeds < 10:
        power_warning_html = f"""
    <section class="bg-amber-50 border border-amber-200 rounded-2xl p-5 md:p-6 mb-8 shadow-sm">
        <h3 class="text-sm font-bold text-amber-900 mb-1.5">
            ⚠️ Nota Metodológica sobre Potencia Estadística (n = {n_seeds} &lt; 10, gl = {df_val})
        </h3>
        <p class="text-xs md:text-sm text-amber-800 leading-relaxed">
            El test <i>t</i> de Student pareado se calculó con <b>n = {n_seeds} semillas</b> (grados de libertad <b>gl = {df_val}</b>) empleando <code>{engine_label}</code>.
            Con tamaños muestrales pequeños (<b>n &lt; 10</b>), la distribución <i>t</i> posee colas pesadas y una <b>potencia estadística reducida (1 − β)</b>, lo que incrementa la probabilidad de Error Tipo II (no detectar significancia estadística cuando la mejora es moderada).
            En consecuencia: (1) un resultado <b>p &lt; 0.05</b> con pocas semillas constituye evidencia sólida de un tamaño de efecto altamente consistente, mientras que (2) un resultado <b>p ≥ 0.05</b> con Δ medio positivo sugiere tendencia favorable que requiere ampliar a <b>n ≥ 10 semillas</b> para confirmar significancia.
        </p>
    </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe Estadístico Agregado (Multi-Seed) — {model_title} ({pool_title})</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #1e293b; }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        @media print {{
            .no-print {{ display: none !important; }}
            body {{ background-color: #ffffff; }}
        }}
    </style>
</head>
<body class="p-4 md:p-8 max-w-6xl mx-auto">

    <div class="no-print flex justify-between items-center mb-6 pb-4 border-b border-slate-200">
        <div class="flex items-center space-x-2 text-sm text-slate-500">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-indigo-600"></span>
            <span>Análisis Estadístico Multi-Semilla ({n_seeds} ejecuciones pareadas)</span>
            <span>•</span>
            <span class="font-mono text-xs">{time.strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        <button onclick="window.print()" class="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg shadow-sm transition">
            Imprimir / Exportar a PDF
        </button>
    </div>

    <header class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <span class="text-xs font-bold tracking-wider uppercase text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full border border-indigo-100">
            Validación Estadística • t-test Pareado (α = 0.05)
        </span>
        <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900 mt-3 tracking-tight">
            Tabla Agregada de Evidencia Estadística — {model_title} ({pool_title})
        </h1>
        <p class="text-sm text-slate-500 mt-1">
            Descriptor: <b>{metric_name}</b> (Ventana: <b>{stat_win}</b>) • Dataset: <b>{dataset_name}</b> • Épocas: <b>{epochs_val}</b> • Semillas evaluadas: <span class="font-mono">{seeds_str}</span>
        </p>
    </header>

    <section class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <h2 class="text-lg font-bold text-slate-900 mb-4">Resumen Estadístico y Significancia (Paired Student's t-test)</h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse text-sm">
                <thead>
                    <tr class="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold text-xs uppercase tracking-wider">
                        <th class="py-3 px-4">Métrica</th>
                        <th class="py-3 px-4">Baseline (Media ± Std)</th>
                        <th class="py-3 px-4 text-indigo-900 bg-indigo-50/50">Propuesto MorphPool (Media ± Std)</th>
                        <th class="py-3 px-4">Δ Medio</th>
                        <th class="py-3 px-4">Δ Std</th>
                        <th class="py-3 px-4">p-value</th>
                        <th class="py-3 px-4">Veredicto (α = 0.05)</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 font-mono text-xs">
                    {''.join(summary_rows)}
                </tbody>
            </table>
        </div>
    </section>

    <section class="bg-white border border-slate-200 rounded-2xl p-6 md:p-8 mb-8 shadow-sm">
        <h2 class="text-lg font-bold text-slate-900 mb-4">Desglose Individual por Semilla (Seed)</h2>
        <div class="overflow-x-auto">
            <table class="w-full text-left border-collapse text-sm">
                <thead>
                    <tr class="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold text-xs uppercase tracking-wider">
                        <th class="py-3 px-4">Seed</th>
                        <th class="py-3 px-4">AUC (Base → Prop)</th>
                        <th class="py-3 px-4">Accuracy (Base → Prop)</th>
                        <th class="py-3 px-4">Sensibilidad (Base → Prop)</th>
                        <th class="py-3 px-4">Especificidad (Base → Prop)</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 font-mono text-xs">
                    {''.join(seed_rows)}
                </tbody>
            </table>
        </div>
    </section>

    {power_warning_html}

</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_path)

