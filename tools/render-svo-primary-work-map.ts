#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import sharp from "sharp";

const reportPath = process.argv[2];
if (!reportPath) throw new Error("usage: render-svo-primary-work-map.ts REPORT.json [OUT-DIR]");
const report = JSON.parse(readFileSync(reportPath, "utf8")) as {
  resolution: { width: number; height: number };
  primaryWorkMap?: { rawPath: string };
};
if (!report.primaryWorkMap) throw new Error("report has no primaryWorkMap");
const { width, height } = report.resolution;
const raw = readFileSync(report.primaryWorkMap.rawPath);
const words = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const out = path.resolve(process.argv[3] ?? path.dirname(reportPath));
mkdirSync(out, { recursive: true });

const metrics = {
  "node-visits": (p: number) => words[p * 4] & 0xffff,
  "leaf-visits": (p: number) => words[p * 4 + 1] & 0xffff,
  "voxel-cells": (p: number) => words[p * 4 + 2],
  "candidate-nodes": (p: number) => (words[p * 4 + 3] >>> 8) & 0xfff,
  "primitive-tests": (p: number) => words[p * 4 + 3] >>> 20,
  "work-items": (p: number) => (words[p * 4] & 0xffff) + words[p * 4 + 2]
    + ((words[p * 4 + 3] >>> 8) & 0xfff) + 4 * (words[p * 4 + 3] >>> 20),
} as const;

const ramp = [
  [8, 5, 31], [36, 71, 255], [0, 217, 255], [0, 255, 133],
  [234, 255, 0], [255, 133, 0], [255, 23, 77], [255, 255, 255],
] as const;
const colour = (tIn: number): readonly [number, number, number] => {
  const t = Math.max(0, Math.min(1, tIn)) * (ramp.length - 1);
  const lo = Math.floor(t), hi = Math.min(ramp.length - 1, lo + 1), f = t - lo;
  return [0, 1, 2].map((c) => Math.round(ramp[lo][c] * (1 - f) + ramp[hi][c] * f)) as [number, number, number];
};

const metadata: Record<string, unknown> = {};
for (const [name, read] of Object.entries(metrics)) {
  const values = Array.from({ length: width * height }, (_, pixel) => read(pixel));
  const sorted = [...values].sort((a, b) => a - b);
  const p99 = sorted[Math.floor(0.99 * (sorted.length - 1))] || 1;
  const maximum = sorted[sorted.length - 1];
  const rgb = Buffer.alloc(width * height * 3);
  for (let pixel = 0; pixel < values.length; pixel += 1) {
    const [r, g, b] = colour(values[pixel] / p99);
    rgb[pixel * 3] = r; rgb[pixel * 3 + 1] = g; rgb[pixel * 3 + 2] = b;
  }
  await sharp(rgb, { raw: { width, height, channels: 3 } }).png().toFile(path.join(out, `${name}.png`));
  metadata[name] = { scaleMaximum: p99, maximum, clippedPixels: values.filter((value) => value > p99).length };
}

// Categorical companion: seed outcome explains the broad cheap regions, while
// the white mask identifies the top 1% by hierarchy + DDA work.
const work = Array.from({ length: width * height }, (_, pixel) => metrics["work-items"](pixel));
const workSorted = [...work].sort((a, b) => a - b);
const tail = workSorted[Math.floor(0.99 * (workSorted.length - 1))];
const percentile = (values: number[], fraction: number) => values[Math.floor(fraction * (values.length - 1))];
const totalWork = work.reduce((sum, value) => sum + value, 0);
const meanWork = totalWork / work.length;
const workVariance = work.reduce((sum, value) => sum + (value - meanWork) ** 2, 0) / work.length;
const descendingWork = [...workSorted].reverse();
const topShare = (fraction: number) => {
  const count = Math.max(1, Math.ceil(work.length * fraction));
  return descendingWork.slice(0, count).reduce((sum, value) => sum + value, 0) / totalWork;
};
const correlation = (a: number[], b: number[]) => {
  const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
  const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
  let covariance = 0, varianceA = 0, varianceB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const da = a[i] - meanA, db = b[i] - meanB;
    covariance += da * db; varianceA += da * da; varianceB += db * db;
  }
  return covariance / Math.sqrt(varianceA * varianceB);
};
const quadEfficiencies: number[] = [];
for (let y = 0; y < height; y += 2) for (let x = 0; x < width; x += 2) {
  const values: number[] = [];
  for (let qy = y; qy < Math.min(height, y + 2); qy += 1) {
    for (let qx = x; qx < Math.min(width, x + 2); qx += 1) values.push(work[qy * width + qx]);
  }
  const maximum = Math.max(...values);
  if (maximum > 0) quadEfficiencies.push(values.reduce((sum, value) => sum + value, 0) / (values.length * maximum));
}
quadEfficiencies.sort((a, b) => a - b);
const tileSize = 32;
const tileTotals: { x: number; y: number; total: number; pixels: number }[] = [];
for (let y = 0; y < height; y += tileSize) for (let x = 0; x < width; x += tileSize) {
  let total = 0, pixels = 0;
  for (let ty = y; ty < Math.min(height, y + tileSize); ty += 1) {
    for (let tx = x; tx < Math.min(width, x + tileSize); tx += 1) {
      total += work[ty * width + tx]; pixels += 1;
    }
  }
  tileTotals.push({ x, y, total, pixels });
}
const rasterOrderedTileTotals = [...tileTotals];
tileTotals.sort((a, b) => b.total - a.total);
const meanTileTotal = totalWork / tileTotals.length;
const nodeValues = Array.from({ length: width * height }, (_, pixel) => metrics["node-visits"](pixel));
const leafValues = Array.from({ length: width * height }, (_, pixel) => metrics["leaf-visits"](pixel));
const voxelValues = Array.from({ length: width * height }, (_, pixel) => metrics["voxel-cells"](pixel));
const totalNodes = nodeValues.reduce((sum, value) => sum + value, 0);
const totalVoxels = voxelValues.reduce((sum, value) => sum + value, 0);
const groupSchedule = (values: number[], groupWidth: number, groupHeight: number) => {
  let scheduled = 0, clipped = 0, tailDeleted = 0, groups = 0, tailGroups = 0, tailLanes = 0;
  for (let y = 0; y < height; y += groupHeight) for (let x = 0; x < width; x += groupWidth) {
    const group: number[] = [];
    for (let gy = y; gy < Math.min(height, y + groupHeight); gy += 1) {
      for (let gx = x; gx < Math.min(width, x + groupWidth); gx += 1) group.push(values[gy * width + gx]);
    }
    const maximum = Math.max(...group), lanes = group.length;
    const inTail = group.filter((value) => value >= tail).length;
    scheduled += maximum * lanes;
    clipped += Math.min(maximum, tail) * lanes;
    tailDeleted += Math.max(...group.map((value) => value >= tail ? 0 : value)) * lanes;
    groups += 1;
    if (inTail > 0) { tailGroups += 1; tailLanes += inTail; }
  }
  return {
    groupWidth, groupHeight, groups,
    efficiency: totalWork / scheduled,
    clippedP99ScheduledReduction: (scheduled - clipped) / scheduled,
    deleteTailScheduledReduction: (scheduled - tailDeleted) / scheduled,
    tailGroupShare: tailGroups / groups,
    meanTailLanesPerTouchedGroup: tailLanes / tailGroups,
  };
};
const shuffledWork = [...work];
let randomState = 0x9e3779b9;
const random = () => {
  randomState ^= randomState << 13; randomState ^= randomState >>> 17; randomState ^= randomState << 5;
  return (randomState >>> 0) / 0x100000000;
};
for (let index = shuffledWork.length - 1; index > 0; index -= 1) {
  const other = Math.floor(random() * (index + 1));
  [shuffledWork[index], shuffledWork[other]] = [shuffledWork[other], shuffledWork[index]];
}
const workerCount = 32;
const idealWorkerLoad = totalWork / workerCount;
const scheduleTiles = (tiles: readonly { total: number }[], dynamic: boolean) => {
  const loads = Array.from({ length: workerCount }, () => 0);
  const ordered = dynamic ? [...tiles].sort((a, b) => b.total - a.total) : tiles;
  ordered.forEach((tile, index) => {
    const worker = dynamic
      ? loads.reduce((best, value, candidate) => value < loads[best] ? candidate : best, 0)
      : index % workerCount;
    loads[worker] += tile.total;
  });
  return { maximumWorkerLoad: Math.max(...loads), excessOverIdeal: Math.max(...loads) / idealWorkerLoad - 1 };
};
const groupModels = [[8, 4], [4, 8], [16, 2], [32, 1]]
  .map(([groupWidth, groupHeight]) => groupSchedule(work, groupWidth, groupHeight));
const analysis = {
  workDistribution: {
    total: totalWork,
    mean: meanWork,
    standardDeviation: Math.sqrt(workVariance),
    coefficientOfVariation: Math.sqrt(workVariance) / meanWork,
    p50: percentile(workSorted, 0.5),
    p90: percentile(workSorted, 0.9),
    p95: percentile(workSorted, 0.95),
    p99: percentile(workSorted, 0.99),
    maximum: workSorted[workSorted.length - 1],
    topShares: {
      "0.1%": topShare(0.001), "1%": topShare(0.01), "5%": topShare(0.05),
      "10%": topShare(0.1), "20%": topShare(0.2),
    },
  },
  countedOperationShares: {
    hierarchyNodes: totalNodes / (totalNodes + totalVoxels),
    voxelCells: totalVoxels / (totalNodes + totalVoxels),
  },
  correlations: {
    nodeVisitsVsVoxelCells: correlation(nodeValues, voxelValues),
    leafVisitsVsVoxelCells: correlation(leafValues, voxelValues),
  },
  quadEfficiency: {
    mean: quadEfficiencies.reduce((sum, value) => sum + value, 0) / quadEfficiencies.length,
    p10: percentile(quadEfficiencies, 0.1),
    p50: percentile(quadEfficiencies, 0.5),
    p95: percentile(quadEfficiencies, 0.95),
  },
  hottestTile: {
    tileSize,
    ...tileTotals[0],
    meanWorkPerPixel: tileTotals[0].total / tileTotals[0].pixels,
    relativeToMeanTileTotal: tileTotals[0].total / meanTileTotal,
  },
  slowTailThroughputModel: {
    model: "spatial SIMT footprint; each group pays its slowest lane's node-plus-cell work",
    p99Threshold: tail,
    usefulWorkExcessAboveP99: work.reduce((sum, value) => sum + Math.max(0, value - tail), 0) / totalWork,
    groupModels,
    shuffled8x4Efficiency: groupSchedule(shuffledWork, 8, 4).efficiency,
    tileScheduling: {
      tileSize, workerCount, idealWorkerLoad,
      greedyDynamic: scheduleTiles(rasterOrderedTileTotals, true),
      staticRasterRoundRobin: scheduleTiles(rasterOrderedTileTotals, false),
    },
  },
};
const categorical = Buffer.alloc(width * height * 3);
for (let pixel = 0; pixel < width * height; pixel += 1) {
  const seed = (words[pixel * 4] >>> 16) & 3;
  const base = seed === 1 ? [20, 42, 74] : seed === 2 ? [0, 188, 212] : [92, 72, 130];
  const rgb = work[pixel] >= tail ? [255, 255, 255] : base;
  categorical.set(rgb, pixel * 3);
}
await sharp(categorical, { raw: { width, height, channels: 3 } }).png().toFile(path.join(out, "entry-seed-and-p99-tail.png"));
writeFileSync(path.join(out, "work-map-scales.json"), `${JSON.stringify({ ...metadata, p99TailThreshold: tail, analysis }, null, 2)}\n`);
console.log(JSON.stringify({ reportPath, out, width, height, ...metadata, p99TailThreshold: tail, analysis }, null, 2));
