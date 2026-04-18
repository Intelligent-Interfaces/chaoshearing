#!/usr/bin/env python3
"""
Shape Grammar Visualizer

Renders the three-layer shape grammar as diagrams:
  - Layer architecture (backend / interface / frontend)
  - Production rules and cross-layer isomorphisms
  - The meta-rule pipeline

Requires: matplotlib, pyyaml
"""

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

GRAMMAR_DIR = Path(__file__).parent

# Colors for each layer
COLORS = {
    "backend": "#4A90D9",    # blue
    "interface": "#D94A4A",  # red
    "frontend": "#4AD97A",   # green
    "cross": "#D9A84A",      # amber
}


def load_yaml(name: str) -> dict:
    with open(GRAMMAR_DIR / name) as f:
        return yaml.safe_load(f)


def draw_layer_architecture(ax, primitives: dict):
    """Draw the three-layer stack with primitives."""
    layers = [
        ("Frontend\n(McDermott Demos)", COLORS["frontend"], primitives.get("frontend", {})),
        ("Interface\n(Chaos Hearing)", COLORS["interface"], primitives.get("interface", {})),
        ("Backend\n(TileGym)", COLORS["backend"], primitives.get("backend", {})),
    ]

    y_positions = [0.7, 0.4, 0.1]
    box_height = 0.2
    box_width = 0.85

    for (label, color, layer_data), y in zip(layers, y_positions):
        rect = mpatches.FancyBboxPatch(
            (0.075, y), box_width, box_height,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.25, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        # Layer label
        ax.text(0.15, y + box_height / 2, label,
                fontsize=10, fontweight="bold", va="center", ha="left",
                color=color)

        # Primitive symbols
        prims = layer_data.get("primitives", [])
        symbols = [p["symbol"] for p in prims[:4]]
        if symbols:
            ax.text(0.55, y + box_height / 2, "  ".join(symbols),
                    fontsize=7, va="center", ha="left",
                    fontfamily="monospace", color="#333")

    # Arrows between layers
    for y_top, y_bot in [(0.7, 0.4 + box_height), (0.4, 0.1 + box_height)]:
        ax.annotate("", xy=(0.5, y_bot), xytext=(0.5, y_top),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["cross"],
                                    lw=2, connectionstyle="arc3,rad=0"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Shape Grammar: Three-Layer Architecture", fontsize=12, fontweight="bold")


def draw_meta_rule(ax):
    """Draw the universal meta-rule pipeline."""
    stages = [
        ("INPUT\n(complex)", "#888"),
        ("DECOMPOSE\ntiles / oscillators\n/ streams", COLORS["backend"]),
        ("DISPATCH\nprocessors / channels\n/ frames", COLORS["interface"]),
        ("TRANSFORM\nfused kernels /\nspectral inference", COLORS["cross"]),
        ("REDUCE\nsplit-K / scene\n/ binding", COLORS["frontend"]),
        ("OUTPUT\n(structured)", "#888"),
    ]

    n = len(stages)
    box_w = 0.12
    box_h = 0.6
    gap = (1.0 - n * box_w) / (n + 1)

    for i, (label, color) in enumerate(stages):
        x = gap + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, 0.2), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, 0.2 + box_h / 2, label,
                fontsize=6, va="center", ha="center", fontweight="bold",
                color="#222")

        # Arrow to next
        if i < n - 1:
            x_end = gap + (i + 1) * (box_w + gap)
            ax.annotate("", xy=(x_end, 0.5), xytext=(x + box_w, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Meta-Rule: Selective Decomposition & Recomposition", fontsize=12, fontweight="bold")


def draw_isomorphism_table(ax, isos: dict):
    """Draw the cross-layer isomorphism table."""
    table_data = isos.get("isomorphism_table", [])
    if not table_data:
        ax.text(0.5, 0.5, "No isomorphism data", ha="center", va="center")
        return

    headers = ["Concept", "Backend", "Interface", "Frontend"]
    col_widths = [0.15, 0.28, 0.28, 0.28]

    n_rows = len(table_data)
    row_height = 0.8 / (n_rows + 1)
    y_start = 0.9

    # Header
    x = 0.01
    for header, w in zip(headers, col_widths):
        ax.text(x + w / 2, y_start, header,
                fontsize=7, fontweight="bold", ha="center", va="center",
                color="#222")
        x += w

    # Separator
    ax.plot([0.01, 0.99], [y_start - row_height * 0.4] * 2, color="#ccc", lw=0.5)

    # Rows
    for i, row in enumerate(table_data):
        y = y_start - (i + 1) * row_height
        x = 0.01
        values = [row.get("concept", ""), row.get("backend", ""),
                  row.get("interface", ""), row.get("frontend", "")]
        colors_row = ["#222", COLORS["backend"], COLORS["interface"], COLORS["frontend"]]

        for val, w, c in zip(values, col_widths, colors_row):
            ax.text(x + w / 2, y, val,
                    fontsize=5.5, ha="center", va="center", color=c,
                    fontfamily="monospace")
            x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Cross-Layer Isomorphisms", fontsize=12, fontweight="bold")


def main():
    primitives = load_yaml("primitives.yaml")
    isomorphisms = load_yaml("isomorphisms.yaml")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    draw_layer_architecture(axes[0], primitives)
    draw_meta_rule(axes[1])
    draw_isomorphism_table(axes[2], isomorphisms)

    fig.suptitle(
        "Shape Grammar for Audio-Visual Computational Perception",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = GRAMMAR_DIR / "shape_grammar_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
