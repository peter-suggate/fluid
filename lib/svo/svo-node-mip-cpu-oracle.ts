import type { SceneDescription } from "../core/model";
import type { SparseSceneDomainPlan } from "../core/sparse-scene-domain";
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
import type { EnvironmentProxyPrimitive } from "../core/voxel-environments";
import {
  SOLID_WORLD_BRICK_CELLS,
  sampleSolidWorld,
  solidWorldForScene,
  type SolidWorld,
} from "../core/solid-world";
import { sceneCellSizes_m } from "../core/scene-lattice";
import { VOXEL_MATERIAL_IDS } from "../core/voxel-scene";

/** WebGPU guarantees at least 8192 texels on a 2D axis; directory height is one row per page. */
export const SVO_NODE_MIP_CPU_ORACLE_DEFAULT_CAPACITY = 8_192;

export interface SvoNodeMipCpuOracleOptions {
  generation: number;
  capacity?: number;
  /** Defaults to the number of levels needed to cover the complete scene domain. */
  levelCount?: number;
  /** Defaults to two, or eight occupancy evaluations per base texel. */
  samplesPerAxis?: 1 | 2 | 4;
  /** Optional final opacity policy; glass and the open presentation wall are excluded by default. */
  includeProxy?: (proxy: EnvironmentProxyPrimitive) => boolean;
}

export interface SvoNodeMipCpuOracleInterior {
  key: SvoNodeMipPageKey;
  /** 8^3 RGBA8: solid mean/max followed by zeroed fluid mean/max. */
  interior: Uint8Array;
}

export interface SvoNodeMipCpuOraclePublication {
  generation: number;
  plan: SvoNodeMipPyramidPlan;
  interiors: readonly SvoNodeMipCpuOracleInterior[];
  worldOrigin_m: readonly [number, number, number];
  baseVoxelSize_m: readonly [number, number, number];
  basePageSize_m: readonly [number, number, number];
  candidateBasePageCount: number;
  selectedBasePageCount: number;
  omittedBasePageCount: number;
  proxyCandidatePageCount: number;
  solidWorldCandidatePageCount: number;
  /**
   * Pages the pyramid would need to represent every candidate. Capacity
   * selection drops base pages in Morton order once the budget is spent, and a
   * dropped page is indistinguishable from empty space to every consumer: the
   * marcher samples a non-resident page as zero occupancy, so its geometry
   * stops casting shadows and stops occluding GI rather than falling back to an
   * exact trace. A caller that finds `omittedBasePageCount` non-zero is holding
   * a publication that renders the wrong picture, and must either raise
   * capacity to this number or decline to publish.
   */
  requiredPageCount: number;
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

/** Proxy-local sample point: the world offset, brought back through any authored rotation. */
function localProxyPoint(proxy: EnvironmentProxyPrimitive, point: readonly [number, number, number]): Triple {
  const offset: Triple = [point[0] - proxy.center_m.x, point[1] - proxy.center_m.y, point[2] - proxy.center_m.z];
  const q = proxy.orientation;
  if (!q) return offset;
  const [x, y, z] = offset;
  const ux = -q.x, uy = -q.y, uz = -q.z;
  const tx = 2 * (uy * z - uz * y), ty = 2 * (uz * x - ux * z), tz = 2 * (ux * y - uy * x);
  return [
    x + q.w * tx + (uy * tz - uz * ty),
    y + q.w * ty + (uz * tx - ux * tz),
    z + q.w * tz + (ux * ty - uy * tx),
  ];
}

function pointInsideProxy(proxy: EnvironmentProxyPrimitive, point: readonly [number, number, number]): boolean {
  const [dx, dy, dz] = localProxyPoint(proxy, point);
  if (proxy.kind === "box") return Math.abs(dx) <= proxy.halfSize_m.x && Math.abs(dy) <= proxy.halfSize_m.y && Math.abs(dz) <= proxy.halfSize_m.z;
  if (proxy.kind === "cylinder") return dx * dx + dz * dz <= proxy.radius_m * proxy.radius_m && Math.abs(dy) <= proxy.halfHeight_m;
  if (proxy.kind === "capsule") {
    const segmentY = Math.max(-proxy.halfLength_m, Math.min(proxy.halfLength_m, dy));
    return dx * dx + (dy - segmentY) ** 2 + dz * dz <= proxy.radius_m * proxy.radius_m;
  }
  if (proxy.kind === "torus") {
    const ring = Math.hypot(dx, dz) - proxy.majorRadius_m;
    return ring * ring + dy * dy <= proxy.minorRadius_m * proxy.minorRadius_m;
  }
  if (proxy.kind === "cone") {
    if (Math.abs(dy) > proxy.halfHeight_m) return false;
    const height = (dy + proxy.halfHeight_m) / (2 * proxy.halfHeight_m);
    const radius_m = proxy.baseRadius_m + (proxy.topRadius_m - proxy.baseRadius_m) * height;
    return dx * dx + dz * dz <= radius_m * radius_m;
  }
  // A tape is its conservative box here, and that is the same trade one level
  // up: the box is what every non-render consumer sees, so the pyramid
  // over-occludes a pitted interior rather than under-populating a page.
  if (proxy.kind === "field-program") {
    return Math.abs(dx) <= proxy.halfExtent_m.x && Math.abs(dy) <= proxy.halfExtent_m.y
      && Math.abs(dz) <= proxy.halfExtent_m.z;
  }
  // A cluster and an ellipsoid share their `radius_m`, and that is deliberate:
  // this oracle mirrors what the voxelizer writes, and the voxelizer sees an
  // aggregate as its solid lobe. Over-occluding a fissured interior is the safe
  // direction — an under-populated page is indistinguishable from empty space
  // to every consumer of the pyramid.
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

function solidWorldDestinationPages(scene: SceneDescription, world: SolidWorld,
  domain: SparseSceneDomainPlan,
  basePageDimensions: Triple): SvoNodeMipCoordinate[] {
  const result = new Map<string, SvoNodeMipCoordinate>();
  const cell = sceneCellSizes_m(scene);
  const sourceOrigin: Triple = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m];
  const destinationPageSize = domain.cellSize_m.map((value) => value
    * SVO_NODE_MIP_LAYOUT.interiorSize) as Triple;
  for (const page of world.pages) {
    if (!page.solidFraction.some((fraction, voxel) => fraction > 0
      && page.materialId[voxel] !== VOXEL_MATERIAL_IDS.containerGlass)) continue;
    const minimum = page.coordinate.map((value, axis) => sourceOrigin[axis]!
      + value * SOLID_WORLD_BRICK_CELLS * cell[axis]!) as Triple;
    const maximum = minimum.map((value, axis) => value
      + SOLID_WORLD_BRICK_CELLS * cell[axis]!) as Triple;
    const first = minimum.map((value, axis) => Math.max(0, Math.floor((value
      - domain.worldOrigin_m[(['x', 'y', 'z'] as const)[axis]])
      / destinationPageSize[axis]!))) as Triple;
    const last = maximum.map((value, axis) => Math.min(basePageDimensions[axis]! - 1,
      Math.ceil((value - domain.worldOrigin_m[(['x', 'y', 'z'] as const)[axis]])
        / destinationPageSize[axis]!) - 1)) as Triple;
    if (first.some((value, axis) => value > last[axis]!)) continue;
    for (let z = first[2]; z <= last[2]; z += 1)
      for (let y = first[1]; y <= last[1]; y += 1)
        for (let x = first[0]; x <= last[0]; x += 1) {
          const coordinate: SvoNodeMipCoordinate = [x, y, z];
          result.set(coordinateKey(coordinate), coordinate);
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
  domain: SparseSceneDomainPlan,
  proxies: readonly EnvironmentProxyPrimitive[],
  solidWorld: SolidWorld,
  solidWorldOrigin: Triple,
  solidWorldCell: Triple,
  samplesPerAxis: number,
): Uint8Array {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize, channels = SVO_NODE_MIP_LAYOUT.channelCount;
  const result = new Uint8Array(n ** 3 * channels);
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const globalCell = [page[0] * n + x, page[1] * n + y, page[2] * n + z] as const;
    if (globalCell.some((value, axis) => value < 0 || value >= domain.sceneDimensionsCells[axis])) continue;
    const bounds = cellBounds(domain, globalCell), candidates = proxies.filter((proxy) => aabbOverlapsCell(proxy, bounds.minimum, bounds.maximum));
    let occupied = 0, solidCouldIntersect = false;
    const sampleCount = samplesPerAxis ** 3;
    for (let sampleZ = 0; sampleZ < samplesPerAxis; sampleZ += 1) for (let sampleY = 0; sampleY < samplesPerAxis; sampleY += 1) for (let sampleX = 0; sampleX < samplesPerAxis; sampleX += 1) {
      const point = bounds.minimum.map((minimum, axis) => minimum + ([sampleX, sampleY, sampleZ][axis] + .5) * domain.cellSize_m[axis] / samplesPerAxis) as Triple;
      const solidCoordinate = point.map((value, axis) => Math.floor((value
        - solidWorldOrigin[axis]!) / solidWorldCell[axis]!)) as Triple;
      const solid = sampleSolidWorld(solidWorld, solidCoordinate);
      const opaqueSolidFraction = solid.materialId === VOXEL_MATERIAL_IDS.containerGlass
        ? 0 : solid.solidFraction;
      solidCouldIntersect ||= opaqueSolidFraction > 0;
      occupied += Math.max(opaqueSolidFraction,
        Number(candidates.some((proxy) => pointInsideProxy(proxy, point))));
    }
    const offset = ((z * n + y) * n + x) * channels;
    result[offset] = Math.round(255 * occupied / sampleCount);
    result[offset + 1] = occupied > 0 || candidates.length > 0
      || solidCouldIntersect ? 255 : 0;
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
 * CPU reference construction of an opacity view over a supplied scene snapshot.
 * It owns no runtime state and never populates fluid lanes.
 */
export function buildSvoNodeMipCpuOraclePublication(
  scene: SceneDescription,
  domain: SparseSceneDomainPlan,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[],
  options: SvoNodeMipCpuOracleOptions,
): SvoNodeMipCpuOraclePublication {
  const capacity = options.capacity ?? SVO_NODE_MIP_CPU_ORACLE_DEFAULT_CAPACITY;
  if (!Number.isSafeInteger(capacity) || capacity < 0) throw new RangeError("Node-mip CPU oracle capacity must be a non-negative safe integer");
  const samplesPerAxis = options.samplesPerAxis ?? 2;
  if (![1, 2, 4].includes(samplesPerAxis)) throw new RangeError("Node-mip CPU oracle samples per axis must be 1, 2, or 4");
  const n = SVO_NODE_MIP_LAYOUT.interiorSize;
  const basePageDimensions = domain.sceneDimensionsCells.map((value) => Math.ceil(value / n)) as Triple;
  const defaultLevelCount = Math.max(1, Math.ceil(Math.log2(Math.max(...basePageDimensions))) + 1);
  const levelCount = options.levelCount ?? defaultLevelCount;
  if (!Number.isSafeInteger(levelCount) || levelCount < 1 || levelCount > 32) throw new RangeError("Node-mip CPU oracle level count must be in [1, 32]");
  const includeProxy = options.includeProxy ?? defaultProxyOpacity;
  const proxies = environmentPrimitives.filter(includeProxy);
  const solidWorld = solidWorldForScene(scene);
  const solidWorldOrigin: Triple = [-0.5 * scene.container.width_m, 0,
    -0.5 * scene.container.depth_m];
  const solidWorldCell = [...sceneCellSizes_m(scene)] as Triple;
  const proxyPages = new Map<string, SvoNodeMipCoordinate>();
  for (const proxy of proxies) for (const page of pageRangeForProxy(proxy, domain, basePageDimensions)) proxyPages.set(coordinateKey(page), page);
  const solidWorldPages = new Map(solidWorldDestinationPages(scene, solidWorld,
    domain, basePageDimensions).map((page) => [coordinateKey(page), page]));
  const candidates = new Map<string, SvoNodeMipCoordinate>([...proxyPages,
    ...solidWorldPages]);
  const orderedCandidates = [...candidates.values()].sort((a, b) => {
    const ma = encodeSvoNodeMipMorton(a), mb = encodeSvoNodeMipMorton(b);
    return ma < mb ? -1 : ma > mb ? 1 : 0;
  });
  const selected = selectedWithinCapacity(orderedCandidates, levelCount, capacity);
  // What a complete pyramid over every candidate would have cost. Planning is
  // pure bookkeeping, so asking is cheap, and it turns "some geometry vanished"
  // into a number the caller can act on.
  const requiredPageCount = selected.length === orderedCandidates.length
    ? undefined
    : planSvoNodeMipPyramid({
      generation: options.generation, occupiedPages: orderedCandidates, levelCount,
      capacity: Number.MAX_SAFE_INTEGER,
    }).requestedPageCount;
  const plan = planSvoNodeMipPyramid({ generation: options.generation, occupiedPages: selected, levelCount, capacity });
  if (!plan.complete) throw new Error("Node-mip CPU oracle capacity selection produced an incomplete plan");

  const values = new Map<string, Uint8Array>();
  const selectedSet = new Set(selected.map(coordinateKey));
  for (const page of plan.pages.filter(({ key }) => key.level === 0 && selectedSet.has(coordinateKey(key.coordinate)))) {
    values.set(interiorKey(0, page.key.coordinate), buildBaseInterior(
      page.key.coordinate, domain, proxies, solidWorld, solidWorldOrigin,
      solidWorldCell, samplesPerAxis));
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
    if (!interior) throw new Error(`Missing node-mip CPU oracle interior ${svoNodeMipPageKey(key)}`);
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
    solidWorldCandidatePageCount: solidWorldPages.size,
    requiredPageCount: requiredPageCount ?? plan.requestedPageCount,
  };
}
