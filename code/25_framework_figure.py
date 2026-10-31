"""Figure 0: four-phase framework diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "analysis", "figures")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(FIG, exist_ok=True)

C_HDR = "#1F3864"
C_BOX = "#EDF2FA"
C_EDGE = "#2A6F97"
C_OUT = "#E8F1E4"
C_OUTE = "#4C8C3F"

PHASES = [
    ("Phase 1\nEvaluation Protocol",
     ["Participant-separated splits", "Permutation nulls",
      "External positive control"],
     "Control 0.775 balanced, 100 participants\nDEED affect 0.322 to 0.377"),
    ("Phase 2\nMechanism Analysis",
     ["Participant-identity decoding", "Identity-only oracle",
      "Pipeline-ordering ladder"],
     "Identity 0.804; oracle 0.547\nProtocol alone reaches 0.906"),
    ("Phase 3\nFoundation-Model Check",
     ["CBraMod, 9,000 h pretraining", "Same protocol throughout",
      "Within-participant test"],
     "Control 0.791; affect 0.315\nWithin-participant p = 0.71"),
    ("Phase 4\nProposed Method (PCD)",
     ["Null decoders on permuted labels", "Margin p-value D(x)",
      "Abstain when D(x) < 1 - alpha"],
     "Separation 9.02x\nFalse-positive rate 0.039"),
]

# Height tracks the y-range: extending ylim to -1.5 without this would
# compress the points-per-data-unit scale and clip the two-line headers.
fig, ax = plt.subplots(figsize=(7.2, 9.3))
ax.set_xlim(0, 10)
ax.set_ylim(-2.6, 21.5)   # the final result box sits below zero
ax.axis("off")

y = 19.6
for i, (title, bullets, result) in enumerate(PHASES):
    ax.add_patch(FancyBboxPatch((0.5, y - 1.32), 9.0, 1.32,
                                boxstyle="round,pad=0.06",
                                fc=C_HDR, ec=C_HDR))
    ax.text(5.0, y - 0.66, title, ha="center", va="center", color="white",
            fontsize=10, fontweight="bold", linespacing=1.35)

    h = 0.62 * len(bullets) + 0.42
    ax.add_patch(FancyBboxPatch((0.5, y - 1.32 - h), 9.0, h,
                                boxstyle="round,pad=0.06",
                                fc=C_BOX, ec=C_EDGE, lw=1.1))
    for j, b in enumerate(bullets):
        ax.text(1.0, y - 1.82 - 0.62 * j, "•  " + b, ha="left",
                va="center", fontsize=8.6)

    yr = y - 1.32 - h - 0.72
    ax.add_patch(FancyBboxPatch((1.6, yr - 0.16), 6.8, 0.88,
                                boxstyle="round,pad=0.05",
                                fc=C_OUT, ec=C_OUTE, lw=1.0))
    ax.text(5.0, yr + 0.28, result, ha="center", va="center", fontsize=8.2,
            color="#1D4D14", linespacing=1.3)

    y = yr - 0.62
    if i < len(PHASES) - 1:
        ax.add_patch(FancyArrowPatch((5.0, y + 0.06), (5.0, y - 0.52),
                                     arrowstyle="-|>", mutation_scale=14,
                                     lw=1.5, color=C_HDR))
        y -= 0.78

ax.text(5.0, 21.0, "Proposed Four-Phase Framework", ha="center",
        va="center", fontsize=11.5, fontweight="bold", color=C_HDR)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig0_framework.{ext}"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, f"fig0_framework.{ext}"), dpi=300,
                bbox_inches="tight")
plt.close(fig)
print("wrote fig0_framework")
