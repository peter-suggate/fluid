import { damBreakFractions } from "./initial-fluid";
import type { HierarchySettings, RigidBodyDescription, SceneDescription, Vec3 } from "./model";

export interface Int3 { x: number; y: number; z: number }

export interface GridBounds {
  min: Vec3;
  max: Vec3;
}

export interface HierarchyBrick {
  id: number;
  level: number;
  coord: Int3;
  parentId: number;
  childIds: number[];
}

export interface LeafPageTable {
  dims: Int3;
  brickIds: Uint32Array;
  levels: Uint8Array;
}

export interface HierarchyTopology {
  readonly settings: HierarchySettings;
  readonly finestCellLength_m: number;
  readonly paddedFinestCellDims: Int3;
  readonly finestBrickDims: Int3;
  readonly baseBrickDims: Int3;
  readonly origin_m: Vec3;
  readonly physicalBounds: GridBounds;
  readonly bricks: HierarchyBrick[];
  leaves: HierarchyBrick[];
  pageTable: LeafPageTable;
  saturated: boolean;
}

export type BrickRefinementOracle = (brick: Readonly<HierarchyBrick>, bounds: Readonly<GridBounds>, cellLength_m: number) => boolean;

const ceilDiv = (value: number, divisor: number) => Math.ceil(value / divisor);
const key = (level: number, coord: Int3) => `${level}:${coord.x}:${coord.y}:${coord.z}`;
const product = (v: Int3) => v.x * v.y * v.z;

function topologyDimensions(scene: SceneDescription): Pick<HierarchyTopology, "paddedFinestCellDims" | "finestBrickDims" | "baseBrickDims"> {
  const { levels, brickSize } = scene.hierarchy;
  const levelScale = 2 ** (levels - 1);
  const alignment = brickSize * levelScale;
  const requested = {
    x: Math.ceil(scene.container.width_m / scene.nominalResolution.length_m),
    y: Math.ceil(scene.container.height_m / scene.nominalResolution.length_m),
    z: Math.ceil(scene.container.depth_m / scene.nominalResolution.length_m)
  };
  const paddedFinestCellDims = {
    x: ceilDiv(requested.x, alignment) * alignment,
    y: ceilDiv(requested.y, alignment) * alignment,
    z: ceilDiv(requested.z, alignment) * alignment
  };
  const finestBrickDims = {
    x: paddedFinestCellDims.x / brickSize,
    y: paddedFinestCellDims.y / brickSize,
    z: paddedFinestCellDims.z / brickSize
  };
  const baseBrickDims = {
    x: finestBrickDims.x / levelScale,
    y: finestBrickDims.y / levelScale,
    z: finestBrickDims.z / levelScale
  };
  return { paddedFinestCellDims, finestBrickDims, baseBrickDims };
}

export function brickCellLength(topology: Pick<HierarchyTopology, "settings" | "finestCellLength_m">, level: number): number {
  return topology.finestCellLength_m * 2 ** (topology.settings.levels - 1 - level);
}

export function brickBounds(topology: Pick<HierarchyTopology, "settings" | "finestCellLength_m" | "origin_m">, brick: Pick<HierarchyBrick, "level" | "coord">): GridBounds {
  const extent = brickCellLength(topology, brick.level) * topology.settings.brickSize;
  const min = {
    x: topology.origin_m.x + brick.coord.x * extent,
    y: topology.origin_m.y + brick.coord.y * extent,
    z: topology.origin_m.z + brick.coord.z * extent
  };
  return { min, max: { x: min.x + extent, y: min.y + extent, z: min.z + extent } };
}

function splitBrick(topology: HierarchyTopology, brick: HierarchyBrick, nodes: Map<string, HierarchyBrick>): boolean {
  if (brick.childIds.length > 0 || brick.level + 1 >= topology.settings.levels) return false;
  if (topology.leaves.length + 7 > topology.settings.maxActiveBricks) {
    topology.saturated = true;
    return false;
  }
  const children: HierarchyBrick[] = [];
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    const child: HierarchyBrick = {
      id: topology.bricks.length,
      level: brick.level + 1,
      coord: { x: brick.coord.x * 2 + x, y: brick.coord.y * 2 + y, z: brick.coord.z * 2 + z },
      parentId: brick.id,
      childIds: []
    };
    topology.bricks.push(child);
    nodes.set(key(child.level, child.coord), child);
    children.push(child);
  }
  brick.childIds = children.map((child) => child.id);
  const leafIndex = topology.leaves.findIndex((leaf) => leaf.id === brick.id);
  topology.leaves.splice(leafIndex, 1, ...children);
  return true;
}

function leafAtFinestBrick(topology: HierarchyTopology, nodes: ReadonlyMap<string, HierarchyBrick>, x: number, y: number, z: number): HierarchyBrick | undefined {
  if (x < 0 || y < 0 || z < 0 || x >= topology.finestBrickDims.x || y >= topology.finestBrickDims.y || z >= topology.finestBrickDims.z) return undefined;
  for (let level = 0; level < topology.settings.levels; level += 1) {
    const scale = 2 ** (topology.settings.levels - 1 - level);
    const brick = nodes.get(key(level, { x: Math.floor(x / scale), y: Math.floor(y / scale), z: Math.floor(z / scale) }));
    if (!brick) return undefined;
    if (brick.childIds.length === 0) return brick;
  }
  return undefined;
}

function enforceBalance(topology: HierarchyTopology, nodes: Map<string, HierarchyBrick>): void {
  let changed = true;
  while (changed && !topology.saturated) {
    changed = false;
    const refine = new Set<number>();
    const d = topology.finestBrickDims;
    for (let z = 0; z < d.z; z += 1) for (let y = 0; y < d.y; y += 1) for (let x = 0; x < d.x; x += 1) {
      const a = leafAtFinestBrick(topology, nodes, x, y, z);
      if (!a) continue;
      for (const offset of [{ x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }]) {
        const b = leafAtFinestBrick(topology, nodes, x + offset.x, y + offset.y, z + offset.z);
        if (!b || Math.abs(a.level - b.level) <= 1) continue;
        refine.add(a.level < b.level ? a.id : b.id);
      }
    }
    for (const id of refine) changed = splitBrick(topology, topology.bricks[id], nodes) || changed;
  }
}

function makePageTable(topology: HierarchyTopology, nodes: ReadonlyMap<string, HierarchyBrick>): LeafPageTable {
  const dims = topology.finestBrickDims;
  const brickIds = new Uint32Array(product(dims));
  const levels = new Uint8Array(product(dims));
  for (let z = 0; z < dims.z; z += 1) for (let y = 0; y < dims.y; y += 1) for (let x = 0; x < dims.x; x += 1) {
    const leaf = leafAtFinestBrick(topology, nodes, x, y, z);
    if (!leaf) throw new Error(`Hierarchy has no leaf at finest brick ${x},${y},${z}`);
    const index = x + dims.x * (y + dims.y * z);
    brickIds[index] = leaf.id;
    levels[index] = leaf.level;
  }
  return { dims: { ...dims }, brickIds, levels };
}

export function buildHierarchy(scene: SceneDescription, oracle: BrickRefinementOracle = () => false): HierarchyTopology {
  const dims = topologyDimensions(scene);
  const topology: HierarchyTopology = {
    settings: { ...scene.hierarchy },
    finestCellLength_m: scene.nominalResolution.length_m,
    ...dims,
    origin_m: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
    physicalBounds: {
      min: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
      max: { x: scene.container.width_m / 2, y: scene.container.height_m, z: scene.container.depth_m / 2 }
    },
    bricks: [],
    leaves: [],
    pageTable: { dims: { x: 0, y: 0, z: 0 }, brickIds: new Uint32Array(), levels: new Uint8Array() },
    saturated: false
  };
  const nodes = new Map<string, HierarchyBrick>();
  for (let z = 0; z < dims.baseBrickDims.z; z += 1) for (let y = 0; y < dims.baseBrickDims.y; y += 1) for (let x = 0; x < dims.baseBrickDims.x; x += 1) {
    const brick: HierarchyBrick = { id: topology.bricks.length, level: 0, coord: { x, y, z }, parentId: 0xffffffff, childIds: [] };
    topology.bricks.push(brick);
    topology.leaves.push(brick);
    nodes.set(key(0, brick.coord), brick);
  }
  for (let level = 0; level + 1 < scene.hierarchy.levels && !topology.saturated; level += 1) {
    const candidates = topology.leaves.filter((brick) => brick.level === level);
    for (const brick of candidates) {
      if (oracle(brick, brickBounds(topology, brick), brickCellLength(topology, brick.level))) splitBrick(topology, brick, nodes);
    }
  }
  enforceBalance(topology, nodes);
  topology.leaves.sort((a, b) => a.level - b.level || a.coord.z - b.coord.z || a.coord.y - b.coord.y || a.coord.x - b.coord.x);
  topology.pageTable = makePageTable(topology, nodes);
  return topology;
}

function intersects(a: GridBounds, b: GridBounds): boolean {
  return a.min.x < b.max.x && a.max.x > b.min.x && a.min.y < b.max.y && a.max.y > b.min.y && a.min.z < b.max.z && a.max.z > b.min.z;
}

function expanded(bounds: GridBounds, amount: number): GridBounds {
  return {
    min: { x: bounds.min.x - amount, y: bounds.min.y - amount, z: bounds.min.z - amount },
    max: { x: bounds.max.x + amount, y: bounds.max.y + amount, z: bounds.max.z + amount }
  };
}

function bodyBounds(body: RigidBodyDescription, padding: number): GridBounds {
  const d = body.dimensions_m;
  const radius = body.shape === "sphere" ? d.x : body.shape === "box" ? Math.hypot(d.x, d.y, d.z) / 2 : Math.hypot(d.x, d.y / 2);
  return {
    min: { x: body.position_m.x - radius - padding, y: body.position_m.y - radius - padding, z: body.position_m.z - radius - padding },
    max: { x: body.position_m.x + radius + padding, y: body.position_m.y + radius + padding, z: body.position_m.z + radius + padding }
  };
}

function initialFluidBounds(scene: SceneDescription): GridBounds {
  const c = scene.container;
  if (scene.fluid.initialCondition === "tank-fill") return {
    min: { x: -c.width_m / 2, y: 0, z: -c.depth_m / 2 },
    max: { x: c.width_m / 2, y: c.height_m * c.fillFraction, z: c.depth_m / 2 }
  };
  const dam = damBreakFractions(c.fillFraction);
  return {
    min: { x: -c.width_m / 2, y: 0, z: -c.depth_m / 2 },
    max: { x: -c.width_m / 2 + c.width_m * dam.width, y: c.height_m * dam.height, z: -c.depth_m / 2 + c.depth_m * dam.depth }
  };
}

function intersectsBoundaryBand(brick: GridBounds, region: GridBounds, band: number): boolean {
  if (!intersects(brick, expanded(region, band))) return false;
  const inner: GridBounds = {
    min: { x: region.min.x + band, y: region.min.y + band, z: region.min.z + band },
    max: { x: region.max.x - band, y: region.max.y - band, z: region.max.z - band }
  };
  return inner.min.x >= inner.max.x || inner.min.y >= inner.max.y || inner.min.z >= inner.max.z || !(
    brick.min.x >= inner.min.x && brick.max.x <= inner.max.x &&
    brick.min.y >= inner.min.y && brick.max.y <= inner.max.y &&
    brick.min.z >= inner.min.z && brick.max.z <= inner.max.z
  );
}

/** Initial topology oracle shared by the WebGPU path and deterministic tests. */
export function buildSceneHierarchy(scene: SceneDescription): HierarchyTopology {
  const fluid = initialFluidBounds(scene);
  const h = scene.nominalResolution.length_m;
  const interfaceBand = h * scene.hierarchy.interfaceHaloCells;
  const solidBand = h * scene.hierarchy.solidHaloCells;
  const bodies = scene.rigidBodies.map((body) => bodyBounds(body, solidBand));
  return buildHierarchy(scene, (brick, bounds) => {
    const nextLevel = brick.level + 1;
    if (nextLevel <= scene.hierarchy.minimumFluidLevel && intersects(bounds, fluid)) return true;
    if (intersectsBoundaryBand(bounds, fluid, interfaceBand)) return true;
    return bodies.some((body) => intersects(bounds, body));
  });
}

export function leafFaceNeighbors(topology: HierarchyTopology, brick: HierarchyBrick, axis: 0 | 1 | 2, sign: -1 | 1): number[] {
  const scale = 2 ** (topology.settings.levels - 1 - brick.level);
  const start = { x: brick.coord.x * scale, y: brick.coord.y * scale, z: brick.coord.z * scale };
  const end = { x: start.x + scale, y: start.y + scale, z: start.z + scale };
  const ids = new Set<number>();
  for (let z = start.z; z < end.z; z += 1) for (let y = start.y; y < end.y; y += 1) for (let x = start.x; x < end.x; x += 1) {
    const p = { x, y, z };
    p[axis === 0 ? "x" : axis === 1 ? "y" : "z"] = sign > 0 ? end[axis === 0 ? "x" : axis === 1 ? "y" : "z"] : start[axis === 0 ? "x" : axis === 1 ? "y" : "z"] - 1;
    if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= topology.pageTable.dims.x || p.y >= topology.pageTable.dims.y || p.z >= topology.pageTable.dims.z) continue;
    ids.add(topology.pageTable.brickIds[p.x + topology.pageTable.dims.x * (p.y + topology.pageTable.dims.y * p.z)]);
  }
  return [...ids].sort((a, b) => a - b);
}

/**
 * Produces bounded child fractions while preserving the exact parent volume.
 * Details are deliberately representation-agnostic; a PLIC reconstruction can
 * supply them without changing restriction or conservation tests.
 */
export function conservativeChildFractions(parentFraction: number, details: readonly number[] = new Array(8).fill(0)): number[] {
  if (details.length !== 8) throw new Error("A refined Cartesian cell must have eight child details");
  const target = Math.max(0, Math.min(1, parentFraction)) * 8;
  const mean = details.reduce((sum, value) => sum + value, 0) / 8;
  const result = details.map((value) => Math.max(0, Math.min(1, parentFraction + value - mean)));
  for (let iteration = 0; iteration < 16; iteration += 1) {
    const error = target - result.reduce((sum, value) => sum + value, 0);
    if (Math.abs(error) <= 1e-12) break;
    const adjustable = result.map((value, index) => ({ value, index })).filter(({ value }) => error > 0 ? value < 1 : value > 0);
    if (adjustable.length === 0) break;
    const share = error / adjustable.length;
    for (const { index } of adjustable) result[index] = Math.max(0, Math.min(1, result[index] + share));
  }
  return result;
}

export function restrictVolumeFractions(childFractions: readonly number[], childVolumes?: readonly number[]): number {
  if (childFractions.length === 0 || (childVolumes && childVolumes.length !== childFractions.length)) throw new Error("Child fractions and volumes must be non-empty and aligned");
  const volumes = childVolumes ?? childFractions.map(() => 1);
  const total = volumes.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) throw new Error("Child volume must be positive");
  return childFractions.reduce((sum, value, index) => sum + value * volumes[index], 0) / total;
}

export function restrictFaceFluxes(childFluxes: readonly number[]): number {
  return childFluxes.reduce((sum, value) => sum + value, 0);
}

/**
 * Reference form of the pressure-level rigid-body Schur term Bᵀ M⁻¹ B.
 * Each column of `bodyJacobian` is one pressure degree of freedom's linear
 * and angular impulse on a body; the GPU applies this same rank-six term
 * matrix-free during every PCG operator evaluation.
 */
export function applyRigidLowRankPressureOperator(
  bodyJacobian: readonly (readonly number[])[],
  inverseMassInertia: readonly number[],
  pressure: readonly number[]
): number[] {
  if (inverseMassInertia.length !== 6) throw new Error("Rigid inverse mass/inertia must have six diagonal entries");
  if (bodyJacobian.length !== pressure.length || bodyJacobian.some((column) => column.length !== 6)) throw new Error("Rigid Jacobian columns must align with pressure values and have six entries");
  const bodyImpulse = new Array<number>(6).fill(0);
  for (let column = 0; column < pressure.length; column += 1) {
    for (let row = 0; row < 6; row += 1) bodyImpulse[row] += bodyJacobian[column][row] * pressure[column];
  }
  return bodyJacobian.map((column) => column.reduce((sum, value, row) => sum + value * inverseMassInertia[row] * bodyImpulse[row], 0));
}
