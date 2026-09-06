"""Compare published height changes with accepted column-volume changes.

Run with the capture directory made by probe-surface-rung-continuity-dawn.ts.
No smoothing is applied to the captured fields.
"""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("capture", type=Path)
args = parser.parse_args()
previous = None
receipts = []
for directory in sorted(args.capture.glob("step-*"), key=lambda p: int(p.name.split("-")[-1])):
    plan = json.loads((directory / "source.json").read_text())
    nx, ny, nz = plan["sampleDimensions"]
    phi = np.fromfile(directory / "phi.bin", dtype=np.float32).reshape(nz, ny, nx)
    density = np.fromfile(directory / "density.bin", dtype=np.float32).reshape(nz, ny, nx)
    height = np.full((nz, nx), np.nan)
    for y in range(ny - 1):
        lo, hi = phi[:, y, :], phi[:, y + 1, :]
        valid = (lo <= 0) & (hi > 0)
        height[valid] = y + .5 - lo[valid] / (hi[valid] - lo[valid])
    volume = density.sum(axis=1, dtype=np.float64)
    receipt = {
        "step": int(directory.name.split("-")[-1]),
        "heightMinimum": float(np.nanmin(height)),
        "heightMaximum": float(np.nanmax(height)),
        "heightMean": float(np.nanmean(height)),
        "volume": float(volume.sum()),
        "maximumHeightVolumeDisagreement": float(np.nanmax(np.abs(height - volume))),
    }
    if previous is not None:
        height_change = height - previous[0]
        volume_change = volume - previous[1]
        worst = np.unravel_index(np.nanargmax(np.abs(height_change)), height.shape)
        receipt.update({
            "maximumHeightChange": float(np.nanmax(np.abs(height_change))),
            "maximumColumnVolumeChange": float(np.max(np.abs(volume_change))),
            "maximumChangeDisagreement": float(np.nanmax(np.abs(height_change - volume_change))),
            "worstHeightColumnXZ": [int(worst[1]), int(worst[0])],
            "worstColumnHeightChange": float(height_change[worst]),
            "worstColumnVolumeChange": float(volume_change[worst]),
        })
    receipts.append(receipt)
    previous = height, volume
print(json.dumps(receipts, indent=2))
