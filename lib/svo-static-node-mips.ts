import type { SceneDescription } from "./model";
import type { SparseSceneDomainPlan } from "./sparse-scene-domain";
import {
  SVO_NODE_MIP_LAYOUT,
  encodeSvoNodeMipMorton,
  planSvoNodeMipPyramid,
  reduceSvoNodeMipChildren,
  svoNodeMipPageKey,
  type SvoNodeMipCoordinate,
  type SvoNodeMipPageKey,
  type SvoNodeMipPyramidPlan,
  type SvoNodeMipRgba8,
} from "./svo-node-mip-pyramid";
import { terrainHeightAt } from "./terrain";
import type { EnvironmentProxyPrimitive } from "./voxel-environments";

/** WebGPU guarantees at least 8192 texels on a 2D axis; directory height is one row per page. */
export const SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY = 8_192;

export interface SvoStaticNodeMipOptions {
  generation: number;
  capacity?: number;
  /** Defaults to the number of levels needed to cover the complete scene domain. */
  levelCount?: number;
  /** Defaults to two, or eight occupancy evaluations per base texel. */
  samplesPerAxis?: 1 | 2 | 4;
  /** Optional final opacity policy; glass and the open presentation wall are excluded by default. */
  includeProxy?: (proxy: EnvironmentProxyPrimitive) => boolean;
}

export interface SvoStaticNodeMipInterior {
  key: SvoNodeMipPageKey;
  /** 8^3 RGBA8: solid mean/max followed by zeroed fluid mean/max. */
  interior: Uint8Array;
}

export interface SvoStaticNodeMipPublication {
  generation: number;
  plan: SvoNodeMipPyramidPlan;
  interiors: readonly SvoStaticNodeMipInterior[];
  worldOrigin_m: readonly [number, number, number];
  baseVoxelSize_m: readonly [number, number, number];
  basePageSize_m: readonly [number, number, number];
  candidateBasePageCount: number;
  selectedBasePageCount: number;
  omittedBasePageCount: number;
  proxyCandidatePageCount: number;
  terrainCandidatePageCount: number;
}

type Triple = [number, number, number];

function defaultProxyOpacity(proxy: EnvironmentProxyPrimitive): boolean {
  if (proxy.key.endsWith("/shell/wall-front") && proxy.tags.includes("shell")) return false;
  return !proxy.tags.includes("glass") && !proxy.group.toLowerCase().includes("glass");
}

function coordinateKey(value: SvoNodeMipCoordinate): string { return `${value[0]},${value[1]},${value[2]}`; }
function interiorKey(level: number, value: SvoNodeMipCoordinate): string { return `${level}:${coordinateKey(value)}`; }

function cellBounds(domain: SparseSceneDomainPlan, globalCell: readonly [number, number, number]) {
  const minimum = globalCell.map((cell, axis) => domain.worldOrigin_m[(["x", "y", "z"] as const)[axis]] + cell * domain.cellSize_m[axis]) as Triple;
  return { minimum, maximum: minimum.map((value, axis) => value + domain.cellSize_m[axis]) as Triple };
}

function pointInsideProxy(proxy: EnvironmentProxyPrimitive, point: readonly [number, number, number]): boolean {
  const dx = point[0] - proxy.center_m.x, dy = point[1] - proxy.center_m.y, dz = point[2] - proxy.center_m.z;
  if (proxy.kind === "box") return Math.abs(dx) <= proxy.halfSize_m.x && Math.abs(dy) <= proxy.halfSize_m.y && Math.abs(dz) <= proxy.halfSize_m.z;
  if (proxy.kind === "cylinder") return dx * dx + dz * dz <= proxy.radius_m * proxy.radius_m && Math.abs(dy) <= proxy.halfHeight_m;
  return (dx / proxy.radius_m.x) ** 2 + (dy / proxy.radius_m.y) ** 2 + (dz / proxy.radius_m.z) ** 2 <= 1;
}

function aabbOverlapsCell(proxy: EnvironmentProxyPrimitive, minimum: Triple, maximum: Triple): boolean {
  const axes = ["x", "y", "z"] as const;
  return axes.every((axis, index) => proxy.aabb_m.max[axis] >= minimum[index] && proxy.aabb_m.min[axis] <= maximum[index]);
}

function pageRangeForProxy(
  proxy: EnvironmentProxyPrimitive,
  domain: SparseSceneDomainPlan,
  basePageDimensions: Triple,
): SvoNodeMipCoordinate[] {
  const axes = ["x", "y", "z"] as const;
  const pageSize = domain.cellSize_m.map((value) => value * SVO_NODE_MIP_LAYOUT.interiorSize);
  const first = axes.map((axis, index) => Math.max(0, Math.floor((proxy.aabb_m.min[axis] - domain.worldOrigin_m[axis]) / pageSize[index]))) as Triple;
  const last = axes.map((axis, index) => Math.min(basePageDimensions[index] - 1, Math.ceil((proxy.aabb_m.max[axis] - domain.worldOrigin_m[axis]) / pageSize[index]) - 1)) as Triple;
  if (first.some((value, axis) => value > last[axis])) return [];
  const result: SvoNodeMipCoordinate[] = [];
  for (let z = first[2]; z <= last[2]; z += 1) for (let y = first[1]; y <= last[1]; y += 1) for (let x = first[0]; x <= last[0]; x += 1) result.push([x, y, z]);
  return result;
}

function terrainSurfacePages(scene: SceneDescription, domain: SparseSceneDomainPlan, basePageDimensions: Triple): SvoNodeMipCoordinate[] {
  if (!scene.terrain) return [];
  const result = new Map<string, SvoNodeMipCoordinate>();
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const solverFirstX = domain.solverGridOriginCells[0], solverFirstZ = domain.solverGridOriginCells[2];
  const solverLastX = solverFirstX + domain.solverDimensionsCells[0], solverLastZ = solverFirstZ + domain.solverDimensionsCells[2];
  for (let pageZ = Math.floor(solverFirstZ / n); pageZ <= Math.floor((solverLastZ - 1) / n); pageZ += 1) {
    for (let pageX = Math.floor(solverFirstX / n); pageX <= Math.floor((solverLastX - 1) / n); pageX += 1) {
      const firstX = Math.max(solverFirstX, pageX * n), lastX = Math.min(solverLastX, (pageX + 1) * n);
      const firstZ = Math.max(solverFirstZ, pageZ * n), lastZ = Math.min(solverLastZ, (pageZ + 1) * n);
      for (let cellZ = firstZ; cellZ < lastZ; cellZ += 1) for (let cellX = firstX; cellX < lastX; cellX += 1) {
        const x = domain.worldOrigin_m.x + (cellX + .5) * domain.cellSize_m[0];
        const z = domain.worldOrigin_m.z + (cellZ + .5) * domain.cellSize_m[2];
        const height = terrainHeightAt(scene.terrain, x, z);
        const surfaceCell = Math.floor((height - domain.worldOrigin_m.y) / domain.cellSize_m[1]);
        // Include the cell containing the surface and its lower neighbour. This
        // retains exact-boundary surfaces without allocating the deep solid volume.
        for (const cellY of [surfaceCell - 1, surfaceCell]) {
          const pageY = Math.floor(cellY / n);
          if (pageX < 0 || pageZ < 0 || pageX >= basePageDimensions[0] || pageY < 0 || pageY >= basePageDimensions[1] || pageZ >= basePageDimensions[2]) continue;
          const coordinate: SvoNodeMipCoordinate = [pageX, pageY, pageZ];
          result.set(coordinateKey(coordinate), coordinate);
        }
      }
    }
  }
  return [...result.values()];
}

function selectedWithinCapacity(candidates: readonly SvoNodeMipCoordinate[], levelCount: number, capacity: number): SvoNodeMipCoordinate[] {
  const selected: SvoNodeMipCoordinate[] = [];
  const planned = new Set<string>();
  for (const candidate of candidates) {
    const additions: string[] = [];
    let coordinate = [...candidate] as Triple;
    for (let level = 0; level < levelCount; level += 1) {
      const key = interiorKey(level, coordinate);
      if (!planned.has(key)) additions.push(key);
      coordinate = coordinate.map((value) => Math.floor(value / 2)) as Triple;
    }
    if (planned.size + additions.length > capacity) continue;
    selected.push(candidate);
    additions.forEach((key) => planned.add(key));
  }
  return selected;
}

function buildBaseInterior(
  page: SvoNodeMipCoordinate,
  scene: SceneDescription,
  domain: SparseSceneDomainPlan,
  proxies: readonly EnvironmentProxyPrimitive[],
  samplesPerAxis: number,
): Uint8Array {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize, channels = SVO_NODE_MIP_LAYOUT.channelCount;
  const result = new Uint8Array(n ** 3 * channels);
  const hasTerrain = !!scene.terrain;
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const globalCell = [page[0] * n + x, page[1] * n + y, page[2] * n + z] as const;
    if (globalCell.some((value, axis) => value < 0 || value >= domain.sceneDimensionsCells[axis])) continue;
    const bounds = cellBounds(domain, globalCell), candidates = proxies.filter((proxy) => aabbOverlapsCell(proxy, bounds.minimum, bounds.maximum));
    let occupied = 0;
    const sampleCount = samplesPerAxis ** 3;
    for (let sampleZ = 0; sampleZ < samplesPerAxis; sampleZ += 1) for (let sampleY = 0; sampleY < samplesPerAxis; sampleY += 1) for (let sampleX = 0; sampleX < samplesPerAxis; sampleX += 1) {
      const point = bounds.minimum.map((minimum, axis) => minimum + ([sampleX, sampleY, sampleZ][axis] + .5) * domain.cellSize_m[axis] / samplesPerAxis) as Triple;
      const insideTerrain = hasTerrain
        && point[0] >= domain.solverBounds_m.min.x && point[0] <= domain.solverBounds_m.max.x
        && point[2] >= domain.solverBounds_m.min.z && point[2] <= domain.solverBounds_m.max.z
        && point[1] <= terrainHeightAt(scene.terrain, point[0], point[2]);
      if (insideTerrain || candidates.some((proxy) => pointInsideProxy(proxy, point))) occupied += 1;
    }
    const terrainCouldIntersect = hasTerrain && bounds.minimum[0] <= domain.solverBounds_m.max.x && bounds.maximum[0] >= domain.solverBounds_m.min.x
      && bounds.minimum[2] <= domain.solverBounds_m.max.z && bounds.maximum[2] >= domain.solverBounds_m.min.z
      && Math.max(...[
        [bounds.minimum[0], bounds.minimum[2]], [bounds.maximum[0], bounds.minimum[2]],
        [bounds.minimum[0], bounds.maximum[2]], [bounds.maximum[0], bounds.maximum[2]],
        [.5 * (bounds.minimum[0] + bounds.maximum[0]), .5 * (bounds.minimum[2] + bounds.maximum[2])],
      ].map(([x, z]) => terrainHeightAt(scene.terrain, x, z))) >= bounds.minimum[1];
    const offset = ((z * n + y) * n + x) * channels;
    result[offset] = Math.round(255 * occupied / sampleCount);
    result[offset + 1] = occupied > 0 || candidates.length > 0 || terrainCouldIntersect ? 255 : 0;
    // Fluid lanes are deliberately zero: dynamic unified-octree fluid remains authoritative.
    result[offset + 2] = 0; result[offset + 3] = 0;
  }
  return result;
}

function texel(interior: Uint8Array | undefined, coordinate: Triple): SvoNodeMipRgba8 {
  if (!interior) return [0, 0, 0, 0];
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const offset = ((coordinate[2] * n + coordinate[1]) * n + coordinate[0]) * 4;
  return [interior[offset], interior[offset + 1], interior[offset + 2], interior[offset + 3]];
}

/**
 * Builds an immutable static-opacity view derived from the same world lattice
 * as the unified octree. It owns no simulation state and never populates fluid lanes.
 */
export function buildSvoStaticNodeMipPublication(
  scene: SceneDescription,
  domain: SparseSceneDomainPlan,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[],
  options: SvoStaticNodeMipOptions,
): SvoStaticNodeMipPublication {
  const capacity = options.capacity ?? SVO_STATIC_NODE_MIP_DEFAULT_CAPACITY;
  if (!Number.isSafeInteger(capacity) || capacity < 0) throw new RangeError("Static SVO node-mip capacity must be a non-negative safe integer");
  const samplesPerAxis = options.samplesPerAxis ?? 2;
  if (![1, 2, 4].includes(samplesPerAxis)) throw new RangeError("Static SVO node-mip samples per axis must be 1, 2, or 4");
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const basePageDimensions = domain.sceneDimensionsCells.map((value) => Math.ceil(value / n)) as Triple;
  const defaultLevelCount = Math.max(1, Math.ceil(Math.log2(Math.max(...basePageDimensions))) + 1);
  const levelCount = options.levelCount ?? defaultLevelCount;
  if (!Number.isSafeInteger(levelCount) || levelCount < 1 || levelCount > 32) throw new RangeError("Static SVO node-mip level count must be in [1, 32]");
  const includeProxy = options.includeProxy ?? defaultProxyOpacity;
  const proxies = environmentPrimitives.filter(includeProxy);
  const proxyPages = new Map<string, SvoNodeMipCoordinate>();
  for (const proxy of proxies) for (const page of pageRangeForProxy(proxy, domain, basePageDimensions)) proxyPages.set(coordinateKey(page), page);
  const terrainPages = new Map(terrainSurfacePages(scene, domain, basePageDimensions).map((page) => [coordinateKey(page), page]));
  const candidates = new Map<string, SvoNodeMipCoordinate>([...proxyPages, ...terrainPages]);
  const orderedCandidates = [...candidates.values()].sort((a, b) => {
    const ma = encodeSvoNodeMipMorton(a), mb = encodeSvoNodeMipMorton(b);
    return ma < mb ? -1 : ma > mb ? 1 : 0;
  });
  const selected = selectedWithinCapacity(orderedCandidates, levelCount, capacity);
  const plan = planSvoNodeMipPyramid({ generation: options.generation, occupiedPages: selected, levelCount, capacity });
  if (!plan.complete) throw new Error("Static SVO node-mip capacity selection produced an incomplete plan");

  const values = new Map<string, Uint8Array>();
  const selectedSet = new Set(selected.map(coordinateKey));
  for (const page of plan.pages.filter(({ key }) => key.level === 0 && selectedSet.has(coordinateKey(key.coordinate)))) {
    values.set(interiorKey(0, page.key.coordinate), buildBaseInterior(page.key.coordinate, scene, domain, proxies, samplesPerAxis));
  }
  for (let level = 1; level < levelCount; level += 1) {
    for (const page of plan.pages.filter(({ key }) => key.level === level)) {
      const parent = new Uint8Array(n ** 3 * 4);
      for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
        const children: SvoNodeMipRgba8[] = [];
        for (let childZ = 0; childZ < 2; childZ += 1) for (let childY = 0; childY < 2; childY += 1) for (let childX = 0; childX < 2; childX += 1) {
          const globalFine = [(page.key.coordinate[0] * n + x) * 2 + childX, (page.key.coordinate[1] * n + y) * 2 + childY, (page.key.coordinate[2] * n + z) * 2 + childZ] as Triple;
          const childPage = globalFine.map((value) => Math.floor(value / n)) as Triple;
          const childTexel = globalFine.map((value) => value % n) as Triple;
          children.push(texel(values.get(interiorKey(level - 1, childPage)), childTexel));
        }
        parent.set(reduceSvoNodeMipChildren(children), ((z * n + y) * n + x) * 4);
      }
      values.set(interiorKey(level, page.key.coordinate), parent);
    }
  }
  const interiors = plan.pages.map(({ key }) => {
    const interior = values.get(interiorKey(key.level, key.coordinate));
    if (!interior) throw new Error(`Missing static SVO node-mip interior ${svoNodeMipPageKey(key)}`);
    return { key, interior };
  });
  return {
    generation: options.generation,
    plan,
    interiors,
    worldOrigin_m: [domain.worldOrigin_m.x, domain.worldOrigin_m.y, domain.worldOrigin_m.z],
    baseVoxelSize_m: [...domain.cellSize_m],
    basePageSize_m: domain.cellSize_m.map((value) => value * n) as Triple,
    candidateBasePageCount: candidates.size,
    selectedBasePageCount: selected.length,
    omittedBasePageCount: candidates.size - selected.length,
    proxyCandidatePageCount: proxyPages.size,
    terrainCandidatePageCount: terrainPages.size,
  };
}
