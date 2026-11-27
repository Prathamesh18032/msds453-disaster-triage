"""Report-width figures for verified held-out aggregates."""

from matplotlib import colormaps
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from .constants import LABELS
from .evidence_figures import GRID, INK, MUTED, _heading
from .evidence_inputs import MISSING, REQUESTS, _correct
from .final_evidence import EVENT_ORDER, TRAINED, metrics

SCOPES = ("pooled", *EVENT_ORDER)
EVENT_NAMES = ("Pooled", "Idai", "Puebla / Mexico", "Matthew")
MODEL_NAMES = ("Majority", "TF-IDF + LR", "DistilBERT 42", "DistilBERT 43", "DistilBERT 44", "DistilBERT mean")
COLORS = ("#657688", "#2c628c", "#037970", "#8e592c", "#595395", INK)
MARKERS = ("s", "D", "o", "^", "v", "o")
CLASS_DISPLAY = (
    "Caution / advice", "Displaced / evacuation", "Infrastructure / utilities", "Injured / dead",
    "Missing / found", "Not humanitarian", "Other relevant info", "Requests / urgent needs",
    "Rescue / volunteering / donations", "Sympathy / support",
)


def _comparison(fig, evidence, mode):
    _heading(fig, "4. Performance across held-out disasters", mode,
             "The interval shows ±1 sample SD across three seeds on the same held-out rows.")
    grid = fig.add_gridspec(1, 4, left=.25, right=.98, bottom=.234, top=.72, wspace=.23)
    for column, (scope, name) in enumerate(zip(SCOPES, EVENT_NAMES, strict=True)):
        ax = fig.add_subplot(grid[0, column])
        reports = [metrics(report, scope) for report in evidence.reports.values()]
        stats = evidence.scores(scope)
        scores = [item["macro_f1"] for item in reports] + [stats["mean"]]
        for i, score in enumerate(scores):
            ax.errorbar(score, i, xerr=stats["sample_sd"] if i == 5 else None, fmt=MARKERS[i],
                        color=COLORS[i], markersize=3.6, capsize=2.5, zorder=3)
            ax.annotate(f"{score:.3f}", (score, i), xytext=(3.7, 0), textcoords="offset points",
                        va="center", fontsize=6.8, color=INK, fontweight="bold" if i == 5 else "normal")
        ax.set_title(f"{name}\n$n$ = {reports[0]['n_rows']:,}", fontsize=9, pad=10)
        ax.set(xlim=(-.04, 1.04), ylim=(5.55, -.55))
        ax.set_xticks([0, .5, 1], ["0", ".5", "1"])
        ax.set_yticks(range(6), MODEL_NAMES if column == 0 else [""] * 6)
        ax.grid(color=GRID, linewidth=.65)
        ax.tick_params(length=0)
    fig.text(.62, .132, "Fixed-ten-label macro-F1", ha="center", color=INK)


def _support(fig, evidence, mode):
    common = evidence.common_labels
    fixed_order = sorted(EVENT_ORDER, key=lambda event: evidence.scores(event)["mean"])
    common_order = sorted(EVENT_ORDER, key=lambda event: evidence.common_score("cohort_mean", event))
    ranking = "changes" if fixed_order != common_order else "preserves"
    _heading(fig, "5. Event scores and class support", mode,
             "Zero cells mark absent true classes. The exploratory comparison excludes any label absent in\n"
             f"one or more events. Using this common set {ranking} the event ordering.")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.13), left=.29, right=.96, bottom=.25, top=.70, wspace=.40)
    ax = fig.add_subplot(grid[0, 0])
    reference = evidence.reports["majority"]["events"]
    for i, label in enumerate(LABELS):
        for j, event in enumerate(EVENT_ORDER):
            count = reference[event]["classes"][label]["support"]
            ax.text(j, i, f"{count:,}", va="center", ha="center", fontsize=8.2,
                    color=INK if count else "#912e37", fontweight="normal" if count else "bold")
            if not count:
                ax.add_patch(Rectangle((j - .48, i - .47), .96, .94, color="#fae8ea", ec="none"))
    ax.set(xlim=(-.5, 2.5), ylim=(len(LABELS) - .5, -.5))
    ax.set_yticks(range(len(LABELS)), CLASS_DISPLAY, fontsize=8.1)
    ax.xaxis.tick_top()
    ax.set_xticks(range(3), ("Idai", "Puebla", "Matthew"), fontsize=8.2)
    ax.set_title("True class counts", fontsize=9, pad=22)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax = fig.add_subplot(grid[0, 1])
    for i, event in enumerate(EVENT_ORDER):
        fixed = evidence.scores(event)["mean"]
        shared = evidence.common_score("cohort_mean", event)
        ax.plot((fixed, shared), (i, i), color=GRID, linewidth=2)
        ax.scatter(fixed, i, color=COLORS[1], marker="s", s=27,
                   label="Fixed ten" if i == 0 else None, zorder=3)
        ax.scatter(shared, i, color=COLORS[2], marker="o", s=27,
                   label=f"Common {len(common)}" if i == 0 else None, zorder=3)
        for value, offset in ((fixed, -13), (shared, 8)):
            ax.annotate(f"{value:.3f}", (value, i), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
    ax.set(xlim=(0, 1), ylim=(2.58, -.6))
    ax.set_yticks(range(3), ("Idai", "Puebla", "Matthew"), fontsize=8.2)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("DistilBERT mean macro-F1")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.08), frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="x", color=GRID)
    ax.tick_params(length=0)


def _rare_recall(fig, evidence, mode):
    reference = evidence.reports["majority"]
    counts = [metrics(reference, event)["classes"][MISSING]["support"] for event in EVENT_ORDER]
    concentration = ""
    if sum(n > 0 for n in counts) == 1:
        concentration = f" All {sum(counts)} missing/found cases occur in {EVENT_NAMES[counts.index(sum(counts)) + 1]}."
    _heading(fig, "6. Recall for requests and missing/found people", mode,
             f"Cells show correct predictions / true cases.{concentration}\n"
             "N/A indicates zero true cases. The class metrics table includes false positives.")
    grid = fig.add_gridspec(1, 2, left=.23, right=.983, bottom=.21, top=.69, wspace=.16)
    for i, (label, name) in enumerate(((REQUESTS, "Requests / urgent needs"), (MISSING, "Missing / found people"))):
        ax = fig.add_subplot(grid[0, i])
        rows = [[metrics(evidence.reports[key], scope)["classes"][label] for scope in SCOPES]
                for key in TRAINED]
        color = [[cell["recall"] if cell["recall"] is not None else float("nan") for cell in row] for row in rows]
        ax.imshow(color, cmap=colormaps["GnBu"].with_extremes(bad="#edf0f3"), vmin=0, vmax=1, aspect="auto")
        for y, row in enumerate(rows):
            for x, cell in enumerate(row):
                support = cell["support"]
                label_text = f"{_correct(cell)}/{support}" if support else "N/A"
                ax.text(x, y, label_text, ha="center", va="center", fontsize=8.5,
                        color="white" if support and cell["recall"] >= .69 else INK)
        ax.xaxis.tick_top()
        ax.set_title(name, fontsize=9, pad=20)
        ax.set_xticks(range(4), ("Pooled", "Idai", "Puebla", "Matthew"), fontsize=8)
        ax.set_yticks(range(4), MODEL_NAMES[1:5] if i == 0 else [""] * 4)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)


def _review(fig, evidence, mode):
    _heading(fig, "7. Simulated review workload and accepted errors", mode,
             "Points follow frozen nominal scenarios (0%, 10%, 20%, 30%); the x-axis shows realized workload.\n"
             "Selective error is the fraction of automatically accepted predictions that were wrong.")
    grid = fig.add_gridspec(1, 2, left=.104, right=.965, bottom=.285, top=.685, wspace=.36)
    for i, (title, field) in enumerate((("Selective error", "selective_error"),
                                        ("Accepted request misses", "accepted_false_negatives"))):
        ax = fig.add_subplot(grid[0, i])
        for idx, key in enumerate(TRAINED, 1):
            points = [row["pooled"] for row in evidence.reports[key]["review"]]
            values = [p[field] if field == "selective_error" else p["requests"][field] for p in points]
            ax.plot([p["review_fraction"] for p in points], values, color=COLORS[idx], marker=MARKERS[idx],
                    markersize=4, linestyle=("-", "--", "-.", ":")[idx - 1], label=MODEL_NAMES[idx])
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if i == 0:
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(color=GRID, linewidth=.65)
    fig.text(.54, .176, "Actual fraction of messages deferred to review", ha="center", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.53, .826),
               fontsize=8.4, frameon=False, ncol=4)


FIGURES = (
    ("figure-4-held-out-model-comparison", 3.55, _comparison),
    ("figure-5-event-support", 4.3, _support),
    ("figure-6-held-out-rare-class-recall", 3.55, _rare_recall),
    ("figure-7-simulated-review", 3.75, _review),
)
