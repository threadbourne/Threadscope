import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Ingestion + defaults
# ----------------------------

def safe_read_csv(file):
    df = pd.read_csv(file.name)
    if df.empty:
        raise ValueError("CSV loaded but contains no rows.")
    return df

def infer_defaults(df: pd.DataFrame):
    cols = list(df.columns)

    # turn candidate
    turn_default = "turn" if "turn" in cols else None

    # speaker candidate
    speaker_default = "speaker" if "speaker" in cols else None

    # magnitude candidate
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    mag_default = None
    for c in ["tokens_est", "tokens", "words", "chars", "length"]:
        if c in numeric_cols:
            mag_default = c
            break
    if mag_default is None and numeric_cols:
        mag_default = numeric_cols[0]

    return turn_default, speaker_default, mag_default, cols, numeric_cols


# ----------------------------
# Stability (Drift & Hold)
# ----------------------------

def compute_stability(df: pd.DataFrame, turn_col: str, mag_col: str,
                      rolling_window: int, band_width: float,
                      stability_thresh: float, persistence: int):
    d = df.copy()

    if mag_col not in d.columns:
        raise ValueError(f"Selected magnitude column '{mag_col}' not found.")
    if not pd.api.types.is_numeric_dtype(d[mag_col]):
        raise ValueError(f"Selected magnitude column '{mag_col}' is not numeric.")

    # Sort by turn if possible, else use row order
    if turn_col and turn_col in d.columns and pd.api.types.is_numeric_dtype(d[turn_col]):
        d = d.sort_values(turn_col).reset_index(drop=True)
        x = d[turn_col].to_numpy()
    else:
        x = np.arange(len(d)) + 1
        turn_col = None

    y = d[mag_col].astype(float).to_numpy()

    w = int(rolling_window)
    w = max(3, min(w, len(d)))

    s = pd.Series(y)
    roll_mean = s.rolling(w, min_periods=max(3, w // 3)).mean().to_numpy()
    roll_std = s.rolling(w, min_periods=max(3, w // 3)).std(ddof=0).to_numpy()

    eps = 1e-9
    z = (y - roll_mean) / (roll_std + eps)

    stable = np.abs(z) <= float(stability_thresh)

    p = int(persistence)
    p = max(1, p)

    stable_persist = np.zeros_like(stable, dtype=bool)
    run = 0
    for i, ok in enumerate(stable):
        if ok and not np.isnan(roll_mean[i]) and not np.isnan(roll_std[i]):
            run += 1
        else:
            run = 0
        if run >= p:
            stable_persist[i] = True

    bw = float(band_width)
    upper = roll_mean + bw * roll_std
    lower = roll_mean - bw * roll_std

    d["_x"] = x
    d["_y"] = y
    d["_roll_mean"] = roll_mean
    d["_roll_std"] = roll_std
    d["_band_upper"] = upper
    d["_band_lower"] = lower
    d["_z"] = z
    d["_stable"] = stable_persist

    # Stable segments
    segments = []
    in_seg = False
    seg_start = None
    for i, ok in enumerate(stable_persist):
        if ok and not in_seg:
            in_seg = True
            seg_start = i
        if in_seg and (not ok or i == len(stable_persist) - 1):
            seg_end = i if ok else i - 1
            in_seg = False
            start_turn = d.loc[seg_start, "_x"]
            end_turn = d.loc[seg_end, "_x"]
            length = seg_end - seg_start + 1
            segments.append((start_turn, end_turn, length))

    return d, segments

def plot_drift_hold(d: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)

    ax.scatter(d["_x"], d["_y"], s=8, alpha=0.6, label="Turns")
    ax.plot(d["_x"], d["_roll_mean"], label="Rolling mean")
    ax.plot(d["_x"], d["_band_upper"], label="Band upper")
    ax.plot(d["_x"], d["_band_lower"], label="Band lower")

    stable_idx = d["_stable"].fillna(False).to_numpy(dtype=bool)
    if stable_idx.any():
        ax.scatter(d.loc[stable_idx, "_x"], d.loc[stable_idx, "_y"], s=14, alpha=0.9, label="Stable (persist)")

    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Magnitude")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------
# Perturbations
# ----------------------------

def temporal_scramble(df: pd.DataFrame, strength: float, seed: int, turn_col: str):
    rng = np.random.default_rng(int(seed))
    d = df.copy()

    if turn_col and turn_col in d.columns and pd.api.types.is_numeric_dtype(d[turn_col]):
        d = d.sort_values(turn_col).reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    n = len(d)
    if n < 2 or strength <= 0:
        return d

    window = int(1 + float(strength) * (n - 1))
    window = max(1, min(window, n))

    idx = np.arange(n)
    out = idx.copy()
    for start in range(0, n, window):
        end = min(start + window, n)
        chunk = out[start:end].copy()
        rng.shuffle(chunk)
        out[start:end] = chunk

    return d.iloc[out].reset_index(drop=True)

def metric_noise(df: pd.DataFrame, strength: float, seed: int, col: str):
    rng = np.random.default_rng(int(seed))
    d = df.copy()

    if col not in d.columns:
        raise ValueError(f"Noise column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(d[col]):
        raise ValueError(f"Noise column '{col}' is not numeric.")

    x = d[col].astype(float).to_numpy()
    if len(x) < 2 or strength <= 0:
        return d

    std = float(np.std(x))
    if std == 0:
        return d

    noise = rng.normal(0, float(strength) * std, size=len(x))
    d[col] = x + noise
    return d


# ----------------------------
# Callbacks
# ----------------------------

def on_upload(file):
    if file is None:
        return (gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=["None"], value="None"),
                "Upload a CSV to begin.",
                None)

    df = safe_read_csv(file)
    turn_default, speaker_default, mag_default, cols, numeric_cols = infer_defaults(df)

    status = f"Loaded {len(df)} rows, {len(df.columns)} columns."
    preview = df.head(15)

    return (gr.update(choices=cols, value=turn_default),
            gr.update(choices=cols, value=speaker_default),
            gr.update(choices=numeric_cols, value=mag_default),
            gr.update(choices=numeric_cols, value=mag_default),
            gr.update(choices=["None"] + cols, value=(turn_default if turn_default else "None")),
            status,
            preview)

def on_upload_all(file):
    tc, sc, mc, nc, tcs, st, pv = on_upload(file)
    return tc, sc, mc, nc, tcs, st, pv

def run_quick_view(file, turn_col, mag_col, rolling_window):
    if file is None:
        return None, "Upload a CSV first."

    df = safe_read_csv(file)

    use_turn = (turn_col not in [None, "None"] and turn_col in df.columns and pd.api.types.is_numeric_dtype(df[turn_col]))
    if use_turn:
        d = df.sort_values(turn_col).reset_index(drop=True)
        x = d[turn_col].to_numpy()
    else:
        d = df.reset_index(drop=True)
        x = np.arange(len(d)) + 1

    if mag_col not in d.columns or not pd.api.types.is_numeric_dtype(d[mag_col]):
        return None, f"'{mag_col}' is not a numeric magnitude column."

    y = d[mag_col].astype(float).to_numpy()
    w = int(rolling_window)
    w = max(3, min(w, len(d)))
    roll = pd.Series(y).rolling(w, min_periods=max(3, w // 3)).mean().to_numpy()

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.6, label="Turns")
    ax.plot(x, roll, label=f"Rolling mean (w={w})")
    ax.set_title(f"Quick View — {mag_col}")
    ax.set_xlabel(turn_col if use_turn else "Turn (row order)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    fig.tight_layout()

    return fig, "Quick View: magnitude over time. Use Advanced tabs for stability bands + perturbations."

def run_drift_hold(file, turn_col, mag_col, rolling_window, band_width, stability_thresh, persistence):
    if file is None:
        return None, "Upload a CSV first.", None

    df = safe_read_csv(file)
    turn_col = None if (turn_col in [None, "None"] or turn_col not in df.columns) else turn_col

    d, segments = compute_stability(
        df=df,
        turn_col=turn_col,
        mag_col=mag_col,
        rolling_window=int(rolling_window),
        band_width=float(band_width),
        stability_thresh=float(stability_thresh),
        persistence=int(persistence),
    )

    fig = plot_drift_hold(d, title=f"Drift & Hold — {mag_col}")

    if segments:
        top = "\n".join([f"• Stable: {s:.0f} → {e:.0f} (len={L})" for s, e, L in segments[:8]])
        msg = f"Detected {len(segments)} stable segment(s).\n{top}"
    else:
        msg = "No stable segments detected with current settings."

    preview_cols = ["_x", "_y", "_roll_mean", "_roll_std", "_stable"]
    return fig, msg, d[preview_cols].head(15)

def run_perturb(file, perturb_type, strength, seed, turn_col_for_scramble, noise_col):
    if file is None:
        return None, "Upload a CSV first.", None

    df = safe_read_csv(file)

    if turn_col_for_scramble in [None, "None"] or turn_col_for_scramble not in df.columns:
        turn_col = None
    else:
        turn_col = turn_col_for_scramble

    if perturb_type == "Temporal scramble":
        df2 = temporal_scramble(df, strength=float(strength), seed=int(seed), turn_col=turn_col)
        msg = f"Applied temporal scramble (strength={strength})."
    else:
        df2 = metric_noise(df, strength=float(strength), seed=int(seed), col=noise_col)
        msg = f"Applied metric noise to '{noise_col}' (strength={strength})."

    # pick a default numeric column to plot
    _, _, mag_default, _, numeric_cols = infer_defaults(df2)
    if not mag_default:
        return None, "No numeric columns available to plot.", df2.head(15)

    if turn_col and pd.api.types.is_numeric_dtype(df2[turn_col]):
        x = df2[turn_col].to_numpy()
        xlabel = turn_col
    else:
        x = np.arange(len(df2)) + 1
        xlabel = "Turn (row order)"

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x, df2[mag_default].astype(float).to_numpy())
    ax.set_title(f"Perturbed — {mag_default}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(mag_default)
    fig.tight_layout()

    return fig, msg, df2.head(15)


# ----------------------------
# UI
# ----------------------------

with gr.Blocks(title="Threadscope: Drift & Hold") as demo:
    gr.Markdown(
        "## Threadscope: Drift & Hold  \n"
        "*Bring Your Own Thread*\n\n"
        "Upload a CSV to visualize long-form interaction dynamics. **Processed in-session only (no storage).**\n\n"
        "**Quick start:** upload → Quick View → adjust rolling window."
    )

    file = gr.File(label="Upload CSV", file_types=[".csv"])
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Accordion("Data mapping (expand if needed)", open=False):
        with gr.Row():
            turn_col = gr.Dropdown(label="Turn column (optional)", choices=[], value=None)
            speaker_col = gr.Dropdown(label="Speaker column (optional)", choices=[], value=None)
            mag_col = gr.Dropdown(label="Magnitude column (numeric)", choices=[], value=None)
        preview = gr.Dataframe(label="Preview (first 15 rows)", interactive=False, wrap=True)

    with gr.Accordion("Advanced inputs (perturbations)", open=False):
        noise_col = gr.Dropdown(label="Noise column (numeric)", choices=[], value=None)
        turn_col_for_scramble = gr.Dropdown(label="Turn column for scramble (optional)", choices=[], value="None")

    file.change(
        fn=on_upload_all,
        inputs=[file],
        outputs=[turn_col, speaker_col, mag_col, noise_col, turn_col_for_scramble, status, preview],
    )

    with gr.Tabs():
        with gr.Tab("Quick View"):
            gr.Markdown(
                "**What this shows:** response magnitude over time + rolling mean.\n\n"
                "For stability bands + persistence, use **Drift & Hold (Advanced)**."
            )
            rolling_window_q = gr.Slider(3, 200, value=25, step=1, label="Rolling window (turns)")
            run_quick = gr.Button("Run Quick View")
            quick_plot = gr.Plot(label="Quick plot")
            quick_msg = gr.Textbox(label="Notes", interactive=False)

            run_quick.click(
                fn=run_quick_view,
                inputs=[file, turn_col, mag_col, rolling_window_q],
                outputs=[quick_plot, quick_msg],
            )

        with gr.Tab("Drift & Hold (Advanced)"):
            gr.Markdown(
                "**Start with defaults** and adjust one slider at a time.\n\n"
                "- Rolling window = 25\n"
                "- How wide is “normal”? = 2.0\n"
                "- How strict is “stable”? = 1.0\n"
                "- How long must it stay stable? = 10"
            )

            with gr.Row():
                rolling_window = gr.Slider(3, 200, value=25, step=1, label="Rolling window (turns)")
                band_width = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="How wide is “normal”? (band width)")

            with gr.Row():
                stability_thresh = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="How strict is “stable”? (threshold)")
                persistence = gr.Slider(1, 100, value=10, step=1, label="How long must it stay stable? (persistence)")

            run_btn = gr.Button("Run Drift & Hold")
            out_plot = gr.Plot(label="Drift & Hold plot")
            out_msg = gr.Textbox(label="Summary", interactive=False)

            with gr.Accordion("Details (computed preview)", open=False):
                out_table = gr.Dataframe(label="Computed preview", interactive=False, wrap=True)

            run_btn.click(
                fn=run_drift_hold,
                inputs=[file, turn_col, mag_col, rolling_window, band_width, stability_thresh, persistence],
                outputs=[out_plot, out_msg, out_table],
            )

        with gr.Tab("Perturbations (Advanced)"):
            gr.Markdown(
                "**Use this to test robustness:**\n"
                "- **Temporal scramble** breaks order but keeps values.\n"
                "- **Metric noise** perturbs values but keeps order."
            )

            perturb_type = gr.Dropdown(
                label="Perturbation type",
                choices=["Temporal scramble", "Metric noise injection"],
                value="Temporal scramble",
            )

            with gr.Row():
                strength = gr.Slider(0, 1, value=0.35, step=0.01, label="Strength")
                seed = gr.Number(value=7, precision=0, label="Seed")

            run_perturb_btn = gr.Button("Apply perturbation")
            pert_plot = gr.Plot(label="Perturbed plot")
            pert_msg = gr.Textbox(label="Notes", interactive=False)

            with gr.Accordion("Perturbed preview (first 15 rows)", open=False):
                pert_preview = gr.Dataframe(interactive=False, wrap=True)

            run_perturb_btn.click(
                fn=run_perturb,
                inputs=[file, perturb_type, strength, seed, turn_col_for_scramble, noise_col],
                outputs=[pert_plot, pert_msg, pert_preview],
            )

demo.launch()
