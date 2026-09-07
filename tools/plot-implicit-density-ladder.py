"""Plot independently evaluated reference/candidate sections from the CPU receipt.

Requires numpy and matplotlib. This reads measured values; it does not resample
or modify the representation. Usage: python tools/plot-implicit-density-ladder.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/implicit-density/ladder.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/implicit-density/ladder.png"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    rows = report["rows"]
    fig, panels = plt.subplots(2, 5, figsize=(17, 7.4), constrained_layout=True)
    for panel, row in zip(panels.flat, rows):
        section = row["section"]
        exact, represented = np.array(section["exact"]), np.array(section["represented"])
        panel.contour(section["x"], section["y"], exact, levels=[0.5], colors=["#182b49"], linewidths=2.6)
        panel.contour(section["x"], section["y"], represented, levels=[0.5], colors=["#ed8b23"], linestyles="dashed", linewidths=1.7)
        panel.set_title(row["id"], fontsize=10)
        panel.set_xlabel("xyz"[section["axes"][0]])
        panel.set_ylabel("xyz"[section["axes"][1]])
        panel.set_aspect("equal")
        panel.set_xlim(section["x"][0], section["x"][-1])
        panel.set_ylim(section["y"][0], section["y"][-1])
        panel.grid(alpha=0.2)
        panel.text(0.02, 0.03, f'max surface density residual\n{row["maximumSurfaceResidual"]:.2e}',
                   transform=panel.transAxes, fontsize=8, color="#44516a")
    fig.suptitle("Implicit density ladder: reference and retained surface after 100 split/merge cycles\n"
                 "Exact-family CPU algebra; feature inference, global assembly and dynamics remain unvalidated",
                 fontsize=13)
    fig.legend(handles=[Line2D([], [], color="#182b49", lw=2.6, label="Independent analytic reference"),
                        Line2D([], [], color="#ed8b23", lw=1.7, ls="--", label="Retained field (overlaps reference)")],
               loc="outside lower center", ncols=2, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
