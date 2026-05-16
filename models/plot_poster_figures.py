"""
Poster figures — focused on encoder design rationale and agent architecture.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

OUT = "/home/ohsejun/Capstone_Design/figures"
NAVY  = "#1a237e"
TEAL  = "#00838f"
AMBER = "#e65100"
RED   = "#b71c1c"
GRAY  = "#546e7a"
GREEN = "#2e7d32"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────
# Figure 1 — Drug Encoder comparison: why ChemBERTa ft?
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

encoders = [
    "Morgan FP\n(no params,\nfixed bits)",
    "GNN\n(from scratch,\nDAVIS only)",
    "ChemBERTa\n(frozen,\nDAVIS only)",
    "ChemBERTa\n(frozen,\nBindingDB→DAVIS)",
    "ChemBERTa\n(fine-tuned,\nBindingDB→DAVIS)",
]
r_vals  = [0.8082, 0.5795, 0.7915, 0.8166, 0.8677]
colors  = [GRAY, RED, AMBER, TEAL, NAVY]
hatches = ["", "//", "//", "", ""]

bars = ax.bar(encoders, r_vals, color=colors, width=0.55, zorder=3,
              edgecolor="white", linewidth=0.8)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

for bar, val in zip(bars, r_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

# problem annotations
ax.annotate("Data scarcity:\n68 unique drugs", xy=(1, 0.5795), xytext=(1.55, 0.62),
            fontsize=8.5, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate("Pretraining\nmismatch\n(MLM ≠ pKd)", xy=(2, 0.7915), xytext=(2.0, 0.72),
            fontsize=8.5, color=AMBER, ha="center",
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))

# baseline reference
ax.axhline(0.8082, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(4.35, 0.8082 + 0.003, "Morgan FP\nbaseline", color=GRAY,
        fontsize=8, ha="right")

ax.set_ylim(0.50, 0.905)
ax.set_ylabel("Pearson r on DAVIS (Test)", fontsize=12)
ax.set_title("Drug Encoder Design Rationale\n"
             "— Why ChemBERTa Fine-tuning on BindingDB?",
             fontsize=13, fontweight="bold", pad=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.tick_params(axis="x", labelsize=9)

legend_items = [
    mpatches.Patch(color=GRAY,  label="Fixed descriptor (no learning)"),
    mpatches.Patch(color=RED,   label="Insufficient training data (68 drugs)", hatch="//"),
    mpatches.Patch(color=AMBER, label="Pretraining objective mismatch", hatch="//"),
    mpatches.Patch(color=TEAL,  label="BindingDB transfer (32K drugs, frozen)"),
    mpatches.Patch(color=NAVY,  label="BindingDB transfer + ChemBERTa fine-tuning ← Ours"),
]
ax.legend(handles=legend_items, fontsize=8.5, loc="lower right", framealpha=0.9)

fig.tight_layout()
fig.savefig(f"{OUT}/fig1_drug_encoder.png", bbox_inches="tight")
plt.close(fig)
print("fig1 saved")


# ─────────────────────────────────────────────────────────────
# Figure 2 — Protein Encoder: Placeholder '#' vs FoldSeek 3Di
# ─────────────────────────────────────────────────────────────
models    = ["SaProt-35M", "SaProt-650M\n8-bit", "SaProt-650M\n4-bit", "SaProt-650M\nFP16"]
r_placeholder = [0.7832, 0.7812, 0.7914, 0.7855]
r_3di         = [0.7996, 0.8027, 0.7977, 0.8082]

x     = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

b1 = ax.bar(x - width/2, r_placeholder, width, color=GRAY,  label='Placeholder "#" token\n(no structure info)', zorder=3, alpha=0.85)
b2 = ax.bar(x + width/2, r_3di,         width, color=TEAL, label="FoldSeek 3Di token\n(AlphaFold structure)", zorder=3)

for bar, val in zip(b1, r_placeholder):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)
for bar, val in zip(b2, r_3di):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# delta annotations
deltas = [r2 - r1 for r1, r2 in zip(r_placeholder, r_3di)]
for i, (xpos, d) in enumerate(zip(x, deltas)):
    ax.text(xpos, max(r_3di[i], r_placeholder[i]) + 0.006,
            f"+{d:.3f}", ha="center", fontsize=9, color=GREEN, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0.765, 0.830)
ax.set_ylabel("Pearson r on DAVIS (Test)", fontsize=12)
ax.set_title("Protein Encoder Design Rationale\n"
             "— Effect of FoldSeek 3Di Structural Tokens",
             fontsize=13, fontweight="bold", pad=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

# note: 4-bit degrades with 3Di
ax.annotate("4-bit quantization\ndegrades 3Di signal",
            xy=(2 + width/2, 0.7977), xytext=(2.7, 0.790),
            fontsize=8, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_protein_encoder.png", bbox_inches="tight")
plt.close(fig)
print("fig2 saved")


# ─────────────────────────────────────────────────────────────
# Figure 3 — Agent pipeline diagram (matplotlib)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

def box(ax, x, y, w, h, label, sublabel="", color=NAVY, fontsize=10, textcolor="white"):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor)
        ax.text(x + w/2, y + h*0.25, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=textcolor, alpha=0.88)
    else:
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor)

def arrow(ax, x1, y1, x2, y2, label="", color="#333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, fontsize=7.5, color=color,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

# User query
box(ax, 4.5, 6.1, 3.0, 0.7, '"Does Imatinib bind to BCR-ABL?"',
    color="#37474f", fontsize=9)

# Agent
box(ax, 3.8, 4.7, 4.4, 1.1,
    "LLM-based Agent",
    "smolagents · ReAct paradigm\n(plan → act → observe → repeat)",
    color=NAVY, fontsize=10.5)

# arrows: user → agent, agent → answer
arrow(ax, 6.0, 6.1, 6.0, 5.8, color="#555")

# Tool boxes (bottom row)
tool_data = [
    (0.2,  "Tool 4\nDrug Name\nResolver", "(PubChem API)\nname → SMILES",   "#5c6bc0"),
    (2.5,  "Tool 5\nProtein Name\nResolver","(UniProt API)\nname → seq",    "#5c6bc0"),
    (4.8,  "Tool 1\nDTI Prediction",       "SaProt + ChemBERTa ft\nSMILES + seq → pKd", TEAL),
    (7.1,  "Tool 2\nProtein\nStructure",   "(AlphaFold DB)\nUniProt → PDB", "#00695c"),
    (9.4,  "Tool 3\nLigand\nStructure",    "(RDKit)\nSMILES → 3D SDF",     "#00695c"),
]

for tx, name, sub, col in tool_data:
    box(ax, tx, 1.6, 2.1, 2.7, name, sub, color=col, fontsize=8.5)

# arrows agent → tools
tool_centers = [tx + 1.05 for tx, *_ in tool_data]
agent_bottom = 4.7
for tc in tool_centers:
    arrow(ax, 6.0, agent_bottom, tc, 1.6 + 2.7, color="#888")

# DTI tool highlight ring
ring = mpatches.FancyBboxPatch((4.65, 1.45), 2.4, 3.05,
                                boxstyle="round,pad=0.1",
                                facecolor="none", edgecolor=AMBER, linewidth=2.5,
                                linestyle="--")
ax.add_patch(ring)
ax.text(5.85, 1.3, "Core Module", ha="center", fontsize=8,
        color=AMBER, fontweight="bold")

# output box
box(ax, 3.5, 0.15, 5.0, 1.0,
    "pKd = 8.7  →  Strong binding  |  3D PDB  |  Ligand SDF",
    color="#263238", fontsize=9)
arrow(ax, 5.85, 1.6, 5.85, 1.15, color="#555")

# input labels on arrows (SMILES / seq)
ax.text(1.4, 3.3, "SMILES", fontsize=7.5, color="#5c6bc0", ha="center", fontstyle="italic")
ax.text(3.6, 3.3, "AA seq", fontsize=7.5, color="#5c6bc0", ha="center", fontstyle="italic")
ax.text(5.85, 3.3, "SMILES\n+ seq", fontsize=7.5, color=TEAL, ha="center", fontstyle="italic")
ax.text(8.15, 3.3, "UniProt ID", fontsize=7.5, color="#00695c", ha="center", fontstyle="italic")
ax.text(10.45, 3.3, "SMILES", fontsize=7.5, color="#00695c", ha="center", fontstyle="italic")

ax.set_title("Bio-AI Agent Pipeline — End-to-End Architecture",
             fontsize=14, fontweight="bold", pad=8)

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_agent_pipeline.png", bbox_inches="tight")
plt.close(fig)
print("fig3 saved")

print("\nAll figures saved to", OUT)
