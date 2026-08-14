import { readFileSync } from "node:fs";
import type { GPUDataFlowManifest } from "../lib/harness/webgpu-data-flow-manifest";

export interface GPUCriticalPathSegment {
  readonly ordinal: number;
  readonly label: string;
  readonly duration_ms: number;
}

export interface GPUCriticalPathReport {
  readonly measuredAdvances: number;
  readonly wall_msPerAdvance: number;
  readonly work_msPerAdvance: number;
  readonly criticalPath_msPerAdvance: number;
  readonly criticalPathToWall: number;
  readonly impliedParallelism: number;
  readonly topChainSegments: readonly GPUCriticalPathSegment[];
  readonly overlap: { readonly observable: false; readonly reason: string };
  readonly decision: "fund-critical-path" | "kill-critical-path" | "inconclusive";
}

/** Longest dependency path through the captured command-order stream. An edge
 * exists for RAW and WAW hazards. Read-only buffers deliberately add no edge. */
export function analyzeGPUCriticalPath(
  manifest: GPUDataFlowManifest,
  wall_msPerAdvance: number,
): GPUCriticalPathReport {
  if (!Number.isFinite(wall_msPerAdvance) || wall_msPerAdvance <= 0) {
    throw new RangeError("wall_msPerAdvance must be positive and finite");
  }
  const advances = Math.max(1, manifest.measuredAdvances);
  const passByLabel = new Map(manifest.passes.map((pass) => [pass.label, pass]));
  const durationByLabel = new Map(manifest.passes.map((pass) => [pass.label,
    (pass.total_ms ?? 0) / Math.max(1, pass.dispatches)]));
  const lastWriter = new Map<number, number>();
  const distance = new Float64Array(manifest.sequence.length);
  const parent = new Int32Array(manifest.sequence.length).fill(-1);
  let terminal = -1;
  for (const node of manifest.sequence) {
    const predecessors = new Set<number>();
    for (const id of [...node.readBufferIds, ...node.writtenBufferIds]) {
      const writer = lastWriter.get(id);
      if (writer !== undefined) predecessors.add(writer);
    }
    let best = 0, bestParent = -1;
    for (const predecessor of predecessors) {
      if (distance[predecessor]! > best) {
        best = distance[predecessor]!;
        bestParent = predecessor;
      }
    }
    const duration = durationByLabel.get(node.label) ?? 0;
    distance[node.ordinal] = best + duration;
    parent[node.ordinal] = bestParent;
    for (const id of node.writtenBufferIds) lastWriter.set(id, node.ordinal);
    if (terminal < 0 || distance[node.ordinal]! > distance[terminal]!) terminal = node.ordinal;
  }
  const chain: GPUCriticalPathSegment[] = [];
  for (let at = terminal; at >= 0; at = parent[at]!) {
    const node = manifest.sequence[at]!;
    chain.push({ ordinal: at, label: node.label,
      duration_ms: durationByLabel.get(node.label) ?? 0 });
  }
  chain.reverse();
  const workTotal = manifest.passes.reduce((sum, pass) => sum + (pass.total_ms ?? 0), 0);
  const critical = terminal < 0 ? 0 : distance[terminal]! / advances;
  const ratio = critical / wall_msPerAdvance;
  return {
    measuredAdvances: advances,
    wall_msPerAdvance,
    work_msPerAdvance: workTotal / advances,
    criticalPath_msPerAdvance: critical,
    criticalPathToWall: ratio,
    impliedParallelism: critical > 0 ? workTotal / advances / critical : 0,
    topChainSegments: [...chain].sort((a, b) => b.duration_ms - a.duration_ms).slice(0, 10),
    overlap: { observable: false,
      reason: "aggregate WebGPU pass timestamps do not retain absolute Metal interval endpoints" },
    decision: ratio > 0.7 ? "fund-critical-path" : ratio < 0.4
      ? "kill-critical-path" : "inconclusive",
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const artifact = process.argv[2];
  if (!artifact) throw new Error("usage: power-liquids-critical-path.ts ARTIFACT.json [wall-ms]");
  const parsed = JSON.parse(readFileSync(artifact, "utf8")) as {
    dataFlow?: GPUDataFlowManifest; summary?: { dataFlow?: GPUDataFlowManifest; advanceWall_ms?: number };
    advanceWall_ms?: number;
  };
  const manifest = parsed.dataFlow ?? parsed.summary?.dataFlow;
  const wall = Number(process.argv[3] ?? parsed.advanceWall_ms ?? parsed.summary?.advanceWall_ms);
  if (!manifest) throw new Error("artifact has no dataFlow manifest");
  console.log(JSON.stringify(analyzeGPUCriticalPath(manifest, wall), null, 2));
}
