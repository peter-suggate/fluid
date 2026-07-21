import { pathToFileURL } from "node:url";

import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { planSvoWideFanout, traverseSvoWideFanout, type SvoWideRay } from "../lib/svo-wide-fanout";
import { packSparseBrickPlan, type SparseBrickCoordinate } from "../lib/sparse-brick-octree";
import { traversePackedSvo, type SvoWorldMapping } from "../lib/webgpu-svo-traversal";
import { planWebgpuSvoWideFanoutAllocation } from "../lib/webgpu-svo-wide-fanout";

export const SVO_DERIVED_LOOKUP_REDUCTION_GATE = 0.35;

export interface SvoDerivedBenchmarkOptions {
  rays: number;
  seed: number;
  gate: boolean;
  json: boolean;
}

export interface SvoDerivedBenchmarkReport {
  seed: number;
  rayCount: number;
  canonical: { nodeVisits: number; averageNodeVisits: number; hits: number; misses: number; failures: number };
  wide: { pageVisits: number; averagePageVisits: number; descriptorTests: number; hits: number; misses: number; failures: number };
  estimatedLookupReduction: number;
  estimatedLookupReductionPercent: number;
  gateThresholdPercent: number;
  gatePassed: boolean;
  topology: { canonicalNodes: number; canonicalLeaves: number; widePages: number; wideDescriptors: number; mixedTerminalLevels: readonly number[] };
  memory: {
    wideBytes: number;
    mipPages: number;
    mipPayloadBytes: number;
    mipAtlasBytes: number;
    mipDirectoryBytes: number;
    mipAllocatedBytes: number;
  };
}

function positiveInteger(raw: string | undefined, label: string): number {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function uint32(raw: string | undefined, label: string): number {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`${label} must fit uint32`);
  return value >>> 0;
}

export function parseSvoDerivedBenchmarkArgs(args: readonly string[]): SvoDerivedBenchmarkOptions {
  const result: SvoDerivedBenchmarkOptions = { rays: 4_096, seed: 0x5eeda11, gate: false, json: false };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--gate") result.gate = true;
    else if (argument === "--json") result.json = true;
    else if (argument === "--rays") result.rays = positiveInteger(args[++index], "Ray count");
    else if (argument.startsWith("--rays=")) result.rays = positiveInteger(argument.slice(7), "Ray count");
    else if (argument === "--seed") result.seed = uint32(args[++index], "Seed");
    else if (argument.startsWith("--seed=")) result.seed = uint32(argument.slice(7), "Seed");
    else throw new RangeError(`Unknown benchmark argument: ${argument}`);
  }
  return result;
}

function nextRandom(state: { value: number }): number {
  let value = state.value >>> 0;
  value ^= value << 13; value ^= value >>> 17; value ^= value << 5;
  state.value = value >>> 0;
  return state.value / 0x1_0000_0000;
}

function coordinateKey(value: SparseBrickCoordinate): string { return `${value.x},${value.y},${value.z}`; }

function deterministicCoordinates(seed: number): { solver: SparseBrickCoordinate[]; proxy: SparseBrickCoordinate[] } {
  const random = { value: seed || 1 };
  const unique = (count: number, offset: number) => {
    const result = new Map<string, SparseBrickCoordinate>();
    while (result.size < count) {
      const value = {
        x: (Math.floor(nextRandom(random) * 240) + offset) & 255,
        y: Math.floor(nextRandom(random) * 160) + 16,
        z: (Math.floor(nextRandom(random) * 240) + offset * 3) & 255,
      };
      result.set(coordinateKey(value), value);
    }
    return [...result.values()];
  };
  const solver = unique(384, 0);
  const solverKeys = new Set(solver.map(coordinateKey));
  const proxy = unique(256, 7).filter((value) => !solverKeys.has(coordinateKey(value)));
  return { solver, proxy };
}

function leafCenterRay(
  coordinate: SparseBrickCoordinate,
  level: number,
  maximumDepth: number,
  brickSize: number,
  sample: number,
): SvoWideRay {
  const scale = 2 ** (maximumDepth - level) * brickSize;
  const jitter = ((sample * 0.61803398875) % 1 - 0.5) * 0.2;
  return {
    origin: [
      (coordinate.x + 0.5 + jitter) * scale,
      (coordinate.y + 0.5 - jitter * 0.5) * scale,
      (coordinate.z + 0.5 + jitter * 0.25) * scale,
    ],
    direction: [0.9622504486, 0.1924500897, 0.1924500897],
    tMin: 0,
    tMax: scale * 4,
  };
}

export function runSvoDerivedHierarchyBenchmark(options: Pick<SvoDerivedBenchmarkOptions, "rays" | "seed">): SvoDerivedBenchmarkReport {
  if (!Number.isSafeInteger(options.rays) || options.rays < 1) throw new RangeError("Ray count must be a positive integer");
  if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 0xffff_ffff) throw new RangeError("Seed must fit uint32");
  const maximumDepth = 8, brickSize = 8;
  const coordinates = deterministicCoordinates(options.seed);
  const canonicalPlan = planAdaptiveSparseBrickOctree({
    brickSize,
    solverBricks: coordinates.solver,
    proxyBricks: coordinates.proxy,
    maximumDepth,
    maximumEnvironmentCoarseningPower: 2,
  });
  const widePlan = planSvoWideFanout({
    sourceGeneration: 1,
    generation: 1,
    maximumDepth,
    terminals: canonicalPlan.leaves.map((leaf) => {
      const node = canonicalPlan.nodes[leaf.nodeIndex];
      return { sourceNodeIndex: node.index, sourceLeafIndex: leaf.index, level: node.level,
        coordinate: [node.coordinate.x, node.coordinate.y, node.coordinate.z] as const };
    }),
  });
  const packed = packSparseBrickPlan(canonicalPlan, 1);
  const mapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize, maximumDepth };
  let canonicalNodeVisits = 0, canonicalHits = 0, canonicalMisses = 0, canonicalFailures = 0;
  let widePageVisits = 0, wideDescriptorTests = 0, wideHits = 0, wideMisses = 0, wideFailures = 0;
  for (let sample = 0; sample < options.rays; sample += 1) {
    const leaf = canonicalPlan.leaves[sample % canonicalPlan.leaves.length];
    const node = canonicalPlan.nodes[leaf.nodeIndex];
    const ray = leafCenterRay(node.coordinate, node.level, maximumDepth, brickSize, sample);
    const canonical = traversePackedSvo(ray, { nodes: packed.nodes, leaves: packed.leaves,
      publishedNodeCount: canonicalPlan.nodes.length, publishedLeafCount: canonicalPlan.leaves.length }, mapping);
    canonicalNodeVisits += canonical.visits;
    if (canonical.status === "hit") canonicalHits += 1;
    else if (canonical.status === "miss") canonicalMisses += 1;
    else canonicalFailures += 1;
    const wide = traverseSvoWideFanout(ray, widePlan, mapping);
    widePageVisits += wide.pageVisits;
    wideDescriptorTests += wide.descriptorTests;
    if (wide.status === "hit") wideHits += 1;
    else if (wide.status === "miss") wideMisses += 1;
    else wideFailures += 1;
  }
  const reduction = canonicalNodeVisits > 0 ? 1 - widePageVisits / canonicalNodeVisits : 0;
  const finestPageCoordinates = new Map<string, readonly [number, number, number]>();
  for (const leaf of canonicalPlan.leaves) {
    const node = canonicalPlan.nodes[leaf.nodeIndex];
    const scale = 2 ** (maximumDepth - node.level);
    const coordinate = [Math.floor(node.coordinate.x * scale / 8), Math.floor(node.coordinate.y * scale / 8),
      Math.floor(node.coordinate.z * scale / 8)] as const;
    finestPageCoordinates.set(coordinate.join(","), coordinate);
  }
  const maximumPageCoordinate = Math.max(0, ...[...finestPageCoordinates.values()].flat());
  const levelCount = Math.max(1, Math.ceil(Math.log2(maximumPageCoordinate + 1)) + 1);
  const mip = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [...finestPageCoordinates.values()], levelCount });
  const wideAllocation = planWebgpuSvoWideFanoutAllocation({ maximumPages: Math.max(1, widePlan.pages.length),
    maximumDescriptors: Math.max(1, widePlan.descriptorCount) });
  return {
    seed: options.seed >>> 0,
    rayCount: options.rays,
    canonical: { nodeVisits: canonicalNodeVisits, averageNodeVisits: canonicalNodeVisits / options.rays,
      hits: canonicalHits, misses: canonicalMisses, failures: canonicalFailures },
    wide: { pageVisits: widePageVisits, averagePageVisits: widePageVisits / options.rays,
      descriptorTests: wideDescriptorTests, hits: wideHits, misses: wideMisses, failures: wideFailures },
    estimatedLookupReduction: reduction,
    estimatedLookupReductionPercent: reduction * 100,
    gateThresholdPercent: SVO_DERIVED_LOOKUP_REDUCTION_GATE * 100,
    gatePassed: canonicalFailures === 0 && wideFailures === 0 && reduction >= SVO_DERIVED_LOOKUP_REDUCTION_GATE,
    topology: { canonicalNodes: canonicalPlan.nodes.length, canonicalLeaves: canonicalPlan.leaves.length,
      widePages: widePlan.pages.length, wideDescriptors: widePlan.descriptorCount,
      mixedTerminalLevels: [...new Set(canonicalPlan.leaves.map((leaf) => canonicalPlan.nodes[leaf.nodeIndex].level))].sort((a, b) => a - b) },
    memory: { wideBytes: wideAllocation.allocatedBytes, mipPages: mip.residentPageCount,
      mipPayloadBytes: mip.pagePayloadBytes, mipAtlasBytes: mip.atlasBytes,
      mipDirectoryBytes: mip.directoryBytes, mipAllocatedBytes: mip.allocatedBytes },
  };
}

export function formatSvoDerivedBenchmarkReport(report: SvoDerivedBenchmarkReport): string {
  return [
    "SVO derived hierarchy CPU report",
    `  rays/seed: ${report.rayCount} / ${report.seed}`,
    `  canonical: ${report.canonical.nodeVisits} node visits (${report.canonical.averageNodeVisits.toFixed(2)}/ray), hits=${report.canonical.hits}, misses=${report.canonical.misses}, failures=${report.canonical.failures}`,
    `  wide 4^3: ${report.wide.pageVisits} page visits (${report.wide.averagePageVisits.toFixed(2)}/ray), descriptor tests=${report.wide.descriptorTests}, hits=${report.wide.hits}, misses=${report.wide.misses}, failures=${report.wide.failures}`,
    `  estimated lookup reduction: ${report.estimatedLookupReductionPercent.toFixed(2)}% (gate >= ${report.gateThresholdPercent.toFixed(0)}%: ${report.gatePassed ? "PASS" : "MISS"})`,
    `  topology: canonical nodes=${report.topology.canonicalNodes}, leaves=${report.topology.canonicalLeaves}, wide pages=${report.topology.widePages}, descriptors=${report.topology.wideDescriptors}, terminal levels=${report.topology.mixedTerminalLevels.join(",")}`,
    `  memory: wide=${report.memory.wideBytes} B, mip pages=${report.memory.mipPages}, payload=${report.memory.mipPayloadBytes} B, atlas=${report.memory.mipAtlasBytes} B, directory=${report.memory.mipDirectoryBytes} B, total=${report.memory.mipAllocatedBytes} B`,
  ].join("\n");
}

function main(args: readonly string[]): void {
  const options = parseSvoDerivedBenchmarkArgs(args);
  const report = runSvoDerivedHierarchyBenchmark(options);
  process.stdout.write(`${options.json ? JSON.stringify(report, null, 2) : formatSvoDerivedBenchmarkReport(report)}\n`);
  if (options.gate && !report.gatePassed) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv.slice(2));
