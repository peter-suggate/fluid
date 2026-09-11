#!/usr/bin/env python3
"""CPU-only rollup of the September 2026 long-dam historical GPU receipts."""
import argparse
import json
import statistics
from pathlib import Path


def read_receipt(directory, name):
    receipt = json.loads((directory / f"{name}.json").read_text())
    assert not receipt["validationErrors"], name
    if "diagnostic" in receipt:
        assert receipt["diagnostic"]["passed"], name
        assert receipt["pressureCutoverReceiptGate"]["passed"], name
        assert receipt["closure"]["maximumAbsoluteError_ms"] < 1e-6, name
    return receipt


def summarize_window(receipt, first, last):
    start = first - receipt["warmupSamples"] - 1
    stop = last - receipt["warmupSamples"]
    assert 0 <= start < stop <= receipt["samples"]
    stages = {
        stage["stage"]: statistics.mean(stage["samples_ms"][start:stop])
        for stage in receipt["stages"] if "|" not in stage["stage"]
    }
    timelines = receipt["candidateTransferTimelines"][start:stop]
    assert [t["advance"] for t in timelines] == list(range(first, last + 1))
    total = statistics.mean(t["gpuTotal_ms"] for t in timelines)
    # Per-stage samples in the JSON are rounded to four decimal places.
    assert abs(sum(stages.values()) - total) < 0.002
    work = receipt["pressureTopologyWork"][start:stop]
    result = {
        "firstAdvance": first, "lastAdvance": last,
        "meanGPU_ms": total,
        "medianGPU_ms": statistics.median(t["gpuTotal_ms"] for t in timelines),
        "stageMeans_ms": stages,
        "workMeans": {key: statistics.mean(w[key] for w in work) for key in [
            "acceptedCells", "acceptedRows", "pressureCells", "pressureRows",
            "endFrameCommittedBricks", "pcmCellDirtyLeaves", "pcmRowPublishedWords",
        ]},
    }
    frames = [f for f in receipt.get("abFrameWork", [])
              if first <= f["advance"] <= last]
    if frames:
        assert len(frames) == last - first + 1
        result["workMeans"]["pressureIterationsExecuted"] = statistics.mean(
            f["pressureIterationsExecuted"] for f in frames)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    old = read_receipt(args.directory, "sep05-pristine-56")
    current = read_receipt(args.directory, "current-pristine-56")
    for key in ["finestGrid", "dt_s", "brickFineResolution", "presentationPageResolution"]:
        assert old["configuration"][key] == current["configuration"][key], key
    result = {"windows": {}}
    for first, last in [(9, 24), (41, 56)]:
        before, after = [summarize_window(r, first, last) for r in [old, current]]
        delta = after["meanGPU_ms"] - before["meanGPU_ms"]
        transport_delta = sum(after["stageMeans_ms"].get(s, 0)
                              - before["stageMeans_ms"].get(s, 0)
                              for s in ["face-preparation", "conservative-transport"])
        result["windows"][f"{first}-{last}"] = {
            "before": before, "current": after,
            "ratio": after["meanGPU_ms"] / before["meanGPU_ms"],
            "delta_ms": delta, "faceAndTransportDelta_ms": transport_delta,
            "faceAndTransportFractionOfDelta": transport_delta / delta,
        }
    result["production"] = {
        name: read_receipt(args.directory, name)["frame_ms"]
        for name in ["sep05-production-repeat", "current-production-pristine"]
    }
    replay = read_receipt(args.directory, "current-face-replay-combined")["abFaceReplay"]
    assert replay[0]["name"] == replay[-1]["name"]
    assert replay[-1]["changedFaceWords"] == 0
    result["sameFrameFaceReplay"] = replay
    result["census"] = {
        name: read_receipt(args.directory, name)["transportProfile"]["face"]
        for name in ["sep06-face-census", "current-face-census"]
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
