"""Report-width, headless figure exports for the evidence renderer."""

from .evidence_inputs import EvidenceError, MISSING, REQUESTS, SCREEN_RATES

PLOT_VERSION = "3.11.2"
SCOPE_NAMES = {
    "greece_wildfires_2018": "Greece wildfires",
    "hurricane_harvey_2017": "Hurricane Harvey",
    "synthetic_alpha": "Synthetic Alpha",
    "synthetic_beta": "Synthetic Beta",
    "pooled": "Pooled",
}
MODEL_NAMES = ("Majority", "Selected TF-IDF + LR", "DistilBERT seed 42", "DistilBERT seed 43", "DistilBERT seed 44")
COLORS = ("#6b7788", "#28648f", "#087d75", "#087d75", "#087d75", "#18354e")
MARKERS = ("s", "D", "o", "o", "o", "o")
INK = "#192b3b"
MUTED = "#48596a"
GRID = "#dce4eb"


def _metrics(report, scope):
    return report["pooled"] if scope == "pooled" else report["events"][scope]


def _heading(fig, title, mode, caption):
    fig.text(.025, .958, title, color=INK, fontsize=14, fontweight="bold", va="top")
    fig.text(.025, .88, mode, color=MUTED, fontsize=9, va="top")
    fig.text(.025, .035, caption, color=MUTED, fontsize=8.5, va="bottom", linespacing=1.4)


def _comparison(fig, evidence, mode):
    _heading(fig, "1. Model-selection performance by event", mode,
             "Macro-F1 uses all ten labels. Pooled scores come from the combined selection rows.\n"
             "The interval marks ±1 sample SD across three seeds evaluated on the same rows.")
    grid = fig.add_gridspec(1, 3, left=.25, right=.973, bottom=.255, top=.73, wspace=.24)
    for column, scope in enumerate(("pooled", *evidence.events)):
        ax = fig.add_subplot(grid[0, column])
        scores = [_metrics(c.report, scope)["macro_f1"] for c in evidence.comparison]
        stats = _metrics(evidence.cohort, scope)
        for i, score in enumerate([*scores, stats["mean"]]):
            ax.errorbar(score, i, xerr=stats["sample_sd"] if i == 5 else None, fmt=MARKERS[i],
                        color=COLORS[i], markersize=4.5, capsize=3, elinewidth=1.25, zorder=3)
        rows = _metrics(evidence.comparison[0].report, scope)["n_rows"]
        ax.set_title(f"{SCOPE_NAMES[scope]}\n$n$ = {rows:,}", fontsize=9.5, linespacing=1.5)
        ax.set_xlim(-.045, 1.045)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
        ax.set_ylim(5.55, -.55)
        ax.set_yticks(range(6), (*MODEL_NAMES, "DistilBERT mean ± SD") if column == 0 else [""] * 6)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.grid(axis="y", color=GRID, linewidth=.45)
    fig.text(.61, .145, "Fixed-ten-label macro-F1", ha="center", color=INK, fontsize=9.5)


def _screen(fig, evidence, mode):
    _heading(fig, "2. DistilBERT learning-rate screening", mode,
             "Both curves use seed 42 and the same selection rows. The ring identifies the accepted recipe.\n"
             "The vertical axis is scaled to the range of the recorded scores.")
    ax = fig.add_axes((.125, .265, .82, .49))
    for (key, rate), color, marker, line in zip(
        SCREEN_RATES.items(), ("#087d75", "#925525"), ("o", "s"), ("-", "--"), strict=True,
    ):
        reports = [c for c in evidence.candidates if c.key == key]
        ax.plot([c.epoch for c in reports], [c.report["pooled"]["macro_f1"] for c in reports],
                marker=marker, label=f"Learning rate {rate:.0g}", color=color,
                linestyle=line, linewidth=1.8, markersize=5)
    chosen = evidence.comparison[2]
    ax.scatter(chosen.epoch, chosen.report["pooled"]["macro_f1"], s=115, facecolors="none",
               edgecolors=INK, linewidths=1.3, zorder=4, label="Selected recipe")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(.68, 3.32)
    ax.margins(y=.20)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pooled macro-F1")
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, borderaxespad=0,
              frameon=False, handlelength=2.3, fontsize=8.5)


def _recall(fig, evidence, mode):
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    _heading(fig, "3. Recall for requests and missing/found people", mode,
             "Cells show correctly predicted rows / true rows. Darker cells indicate higher recall.\n"
             "N/A indicates zero true rows for the label in that event.")
    grid = fig.add_gridspec(1, 2, left=.25, right=.975, bottom=.185, top=.705, wspace=.09)
    for column, (label, title) in enumerate(((REQUESTS, "Requests / urgent needs"),
                                             (MISSING, "Missing / found people"))):
        ax = fig.add_subplot(grid[0, column])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=25)
        for i, c in enumerate(evidence.comparison):
            for j, scope in enumerate(("pooled", *evidence.events)):
                cell = _metrics(c.report, scope)["classes"][label]
                support, recall = cell["support"], cell["recall"]
                color = colormaps["Blues"](.12 + .71 * recall) if support else "#eceff2"
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
                text = f"{round(recall * support)}/{support}" if support else "N/A (n=0)"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.7,
                        color="white" if support and recall >= .69 else INK)
        ax.set_ylim(4.5, -.5)
        ax.set_xlim(-.5, 2.5)
        ax.xaxis.tick_top()
        headings = {"greece_wildfires_2018": "Greece", "hurricane_harvey_2017": "Harvey",
                    "synthetic_alpha": "Alpha", "synthetic_beta": "Beta"}
        ax.set_xticks(range(3), ["Pooled", *(headings[event] for event in evidence.events)])
        ax.set_yticks(range(5), MODEL_NAMES if column == 0 else [""] * 5)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.spines[:].set_visible(False)


def render_figures(directory, evidence, mode):
    try:
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        raise EvidenceError("Install the package's evidence extra for Matplotlib 3.11.2.") from None
    if mpl.__version__ != PLOT_VERSION:
        raise EvidenceError("Evidence rendering requires Matplotlib 3.11.2.")

    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
            "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": GRID, "xtick.color": INK, "ytick.color": INK,
            "pdf.fonttype": 42, "savefig.dpi": 240,
        })
        for name, height, draw in (
            ("figure-1-model-comparison", 3.35, _comparison),
            ("figure-2-learning-rate-screening", 3.25, _screen),
            ("figure-3-rare-class-recall", 3.35, _recall),
        ):
            fig = Figure(figsize=(7.1, height), facecolor="white")
            FigureCanvasAgg(fig)
            draw(fig, evidence, mode)
            fig.savefig(directory / f"{name}.png", metadata={"Software": f"Matplotlib {PLOT_VERSION}"})
            fig.savefig(directory / f"{name}.pdf", metadata={
                "Creator": f"Disaster Triage evidence renderer (Matplotlib {PLOT_VERSION})",
                "CreationDate": None, "ModDate": None,
            })
    return PLOT_VERSION
