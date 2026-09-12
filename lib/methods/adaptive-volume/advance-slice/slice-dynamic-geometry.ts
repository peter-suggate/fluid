import { averageInflowStrength, createInflowGridBoundary } from "../../../core/inflow-boundary";
import type { Quaternion, RigidBodyDescription, Vec3 } from "../../../core/model";
import { quaternionInverseRotate } from "../../../core/rigid-body";
import { sceneShape } from "../../../core/scene-shape";
import { sampleSolidWorld } from "../../../core/solid-world";
import type { SliceNumericalTopology } from "./slice-stage-numerics";
import type { SliceSceneSeed } from "./slice-scene-seed";

/** Live pose supplied by the production rigid-body owner. */
export interface SliceRigidPose {
  readonly description: RigidBodyDescription;
  readonly position_m: Vec3;
  readonly orientation: Quaternion;
  readonly linearVelocity_m_s: Vec3;
  readonly angularVelocity_rad_s: Vec3;
}

export interface SliceDynamicGeometryInput {
  readonly seed: SliceSceneSeed;
  readonly topology: SliceNumericalTopology;
  readonly time_s: number;
  readonly dt_s: number;
  /** Current production poses. Omission uses the authored t=0 descriptors. */
  readonly bodies?: readonly SliceRigidPose[];
  /** Previous production poses used by the moving-capacity rate. */
  readonly previousBodies?: readonly SliceRigidPose[];
  /** Current compact density, in production rho units. */
  readonly density?: ArrayLike<number>;
  /** Persistent unserved 2-D source area in finest-cell-squared units. */
  readonly pendingSourceAreaFine?: number;
  /** Persistent production Kahan lane paired with pending source area. */
  readonly pendingSourceCompensation?: number;
  /** Previous accepted pressure membership used by production gsCell. */
  readonly pressureMember?: ArrayLike<number>;
  /** Accepted cell ids in production brick dispatch/reduction order. */
  readonly sourceReductionGroups?: readonly (readonly number[])[];
}

export interface SliceDynamicGeometry {
  /** Final compact cell-open fraction (compatibility alias for capacityAfter). */
  readonly capacity: Float32Array;
  /** Exact old/new geometric endpoints retained through volume transport. */
  readonly capacityBefore: Float32Array;
  readonly capacityAfter: Float32Array;
  /** (currentCapacity-previousCapacity)/dt, fraction/s. */
  readonly capacityRate: Float32Array;
  /** Extensive unit-depth source area per second, finest-cell squared/s. */
  readonly sourceRate: Float32Array;
  /** Production uses the arithmetic mean aperture until final publication. */
  readonly openFraction: Float32Array;
  readonly openFractionBefore: Float32Array;
  readonly openFractionAfter: Float32Array;
  readonly meanOpenFraction: Float32Array;
  /** Source-frame x/y row velocity in finest cells/s. */
  readonly solidVelocity: Float32Array;
  /** Prescribed production inflow coverage, zero for non-inflow rows. */
  readonly inflowCoverage: Float32Array;
  /** Interval-averaged authored in-plane inflow velocity, finest cells/s. */
  readonly inflowVelocityFine: readonly [number, number];
  readonly requestedSourceAreaFine: number;
  readonly sourceRateAreaFine: number;
  readonly sourceAvailableAreaFine: number;
  readonly sourceFactor: number;
  readonly sourceComponentCount: number;
  readonly sourceAnchoredComponentCount: number;
}

const f = Math.fround;
const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));
const components = (value: Vec3): readonly [number, number, number] =>
  [value.x, value.y, value.z];

function authoredPose(body: RigidBodyDescription): SliceRigidPose {
  return { description: body, position_m: body.position_m, orientation: body.orientation,
    linearVelocity_m_s: body.linearVelocity_m_s,
    angularVelocity_rad_s: body.angularVelocity_rad_s };
}

function contains(body: SliceRigidPose, world: Vec3): boolean {
  const local = quaternionInverseRotate(body.orientation, {
    x: world.x - body.position_m.x, y: world.y - body.position_m.y,
    z: world.z - body.position_m.z,
  });
  return sceneShape(body.description.shape).inside(body.description.dimensions_m, local);
}

function ownerAt(bodies: readonly SliceRigidPose[], world: Vec3): SliceRigidPose | undefined {
  return bodies.find(body => contains(body, world));
}

function velocityAt(body: SliceRigidPose, world: Vec3): Vec3 {
  const arm = { x: world.x - body.position_m.x, y: world.y - body.position_m.y,
    z: world.z - body.position_m.z };
  const w = body.angularVelocity_rad_s;
  return { x: body.linearVelocity_m_s.x + w.y * arm.z - w.z * arm.y,
    y: body.linearVelocity_m_s.y + w.z * arm.x - w.x * arm.z,
    z: body.linearVelocity_m_s.z + w.x * arm.y - w.y * arm.x };
}

function worldPoint(seed: SliceSceneSeed, fine: readonly [number, number]): Vec3 {
  const h = seed.viewport.sourceCellSize;
  return { x: seed.viewport.originX + fine[0] * h,
    y: seed.viewport.originY + fine[1] * h, z: seed.viewport.centerZ };
}

function rigidOpenAtFineCell(seed: SliceSceneSeed, bodies: readonly SliceRigidPose[],
  x: number, y: number): number {
  if (!bodies.length) return 1;
  const center = worldPoint(seed, [x + 0.5, y + 0.5]);
  const h = seed.viewport.sourceCellSize;
  let maximum = 0;
  for (const body of bodies) {
    let covered = 0;
    for (const dx of [-0.4, 0.4]) for (const dy of [-0.4, 0.4]) {
      if (contains(body, { x: center.x + dx * h, y: center.y + dy * h, z: center.z })) {
        covered += 0.25;
      }
    }
    maximum = Math.max(maximum, covered);
  }
  return 1 - maximum;
}

function compactCapacity(seed: SliceSceneSeed, topology: SliceNumericalTopology,
  bodies: readonly SliceRigidPose[]): Float32Array {
  const result = new Float32Array(topology.cells.length);
  const world = seed.production?.solidWorld;
  for (const cell of topology.cells) {
    let sum = 0, count = 0;
    for (let y = Math.floor(cell.minimum[1]); y < Math.ceil(cell.maximum[1]); y += 1) {
      for (let x = Math.floor(cell.minimum[0]); x < Math.ceil(cell.maximum[0]); x += 1) {
        const staticOpen = world ? 1 - sampleSolidWorld(world,
          [x, y, seed.viewport.centerCellZ]).solidFraction : 1;
        sum += staticOpen * rigidOpenAtFineCell(seed, bodies, x, y);
        count += 1;
      }
    }
    result[cell.id] = f(count ? sum / count : 0);
  }
  return result;
}

function staticRowOpen(seed: SliceSceneSeed, axis: 0 | 1,
  center: readonly [number, number], area: number): number {
  const world = seed.production?.solidWorld;
  if (!world) return 1;
  const plane = Math.round(center[axis]);
  const tangentAxis = axis === 0 ? 1 : 0;
  const start = Math.round(center[tangentAxis] - 0.5 * area);
  let open = 0, samples = 0;
  for (let tangent = start; tangent < start + Math.round(area); tangent += 1) {
    const negative: [number, number, number] = [0, 0, seed.viewport.centerCellZ];
    negative[axis] = plane - 1; negative[tangentAxis] = tangent;
    const positive = [...negative] as [number, number, number]; positive[axis] = plane;
    open += 1 - Math.max(sampleSolidWorld(world, negative).solidFraction,
      sampleSolidWorld(world, positive).solidFraction);
    samples += 1;
  }
  return samples ? open / samples : 0;
}

function dynamicRows(seed: SliceSceneSeed, topology: SliceNumericalTopology,
  bodies: readonly SliceRigidPose[]) {
  const openFraction = new Float32Array(topology.rows.length);
  const solidVelocity = new Float32Array(topology.rows.length);
  const h = seed.viewport.sourceCellSize;
  for (const row of topology.rows) {
    if (row.kind === "closed-world") continue;
    const center = worldPoint(seed, row.center);
    const tangent = row.area * h;
    let covered = 0, velocity = 0;
    for (const sign of [-0.35, 0.35]) {
      const point = row.axis === 0
        ? { x: center.x, y: center.y + sign * tangent, z: center.z }
        : { x: center.x + sign * tangent, y: center.y, z: center.z };
      const owner = ownerAt(bodies, point);
      if (!owner) continue;
      covered += 0.5;
      velocity += 0.5 * components(velocityAt(owner, point))[row.axis]! / h;
    }
    let fallback: SliceRigidPose | undefined;
    if (covered === 0) for (const normalSign of [-0.4, 0.4]) {
      for (const tangentSign of [-0.4, 0.4]) {
        const point = row.axis === 0
          ? { x: center.x + normalSign * Math.max(row.distance, 1) * h,
            y: center.y + tangentSign * tangent, z: center.z }
          : { x: center.x + tangentSign * tangent,
            y: center.y + normalSign * Math.max(row.distance, 1) * h, z: center.z };
        fallback ??= ownerAt(bodies, point);
      }
    }
    openFraction[row.id] = f(staticRowOpen(seed, row.axis, row.center, row.area)
      * (1 - covered));
    solidVelocity[row.id] = f(covered > 0 ? velocity / covered
      : fallback ? components(velocityAt(fallback, center))[row.axis]! / h : 0);
  }
  return { openFraction, solidVelocity };
}

function compensatedSum(values: Iterable<number>): number {
  let total = 0, correction = 0;
  for (const source of values) {
    const value = f(source - correction), next = f(total + value);
    correction = f(f(next - total) - value); total = next;
  }
  return total;
}

function productionGroupedSum(values: ArrayLike<number>,
  groups?: readonly (readonly number[])[]): number {
  if (!groups?.length) return compensatedSum(Array.from(values));
  return compensatedSum(groups.map(group => compensatedSum(group.map(id => values[id] ?? 0))));
}

function inflowFields(input: SliceDynamicGeometryInput, capacity: Float32Array,
  openFraction: Float32Array) {
  const { seed, topology } = input;
  const rates = new Float32Array(topology.cells.length);
  const rowCoverage = new Float32Array(topology.rows.length);
  const inflow = seed.production?.scene.fluid.inflow;
  if (!inflow) return { rates, rowCoverage, velocityFine: [0, 0] as const,
    requested: 0, planned: 0, available: 0, factor: 0,
    components: 0, anchoredComponents: 0 };
  const scene = seed.production!.scene;
  const boundary = createInflowGridBoundary(inflow, scene.container,
    seed.sourceAtlas!.dimensions);
  // The GPU host freezes the exact interval average for the whole outer step;
  // evaluating only the beginning of the interval changes ramp-edge doses.
  const strength = f(averageInflowStrength(inflow, input.time_s,
    input.time_s + input.dt_s));
  const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y,
    inflow.velocity_m_s.z);
  if (!(strength > 0 && speed > 0 && input.dt_s > 0)) {
    return { rates, rowCoverage, velocityFine: [0, 0] as const,
      requested: 0, planned: 0, available: 0, factor: 0,
      components: 0, anchoredComponents: 0 };
  }
  const h = seed.viewport.sourceCellSize;
  const effectiveSpeed = f(speed * strength);
  const velocityFine = [f(inflow.velocity_m_s.x * strength / h),
    f(inflow.velocity_m_s.y * strength / h)] as const;
  const direction = { x: inflow.velocity_m_s.x / speed,
    y: inflow.velocity_m_s.y / speed, z: inflow.velocity_m_s.z / speed };
  const outlet = boundary.outletCenter_m;
  const inPlaneSpeed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y);
  const planeOffset = Math.abs(seed.viewport.centerZ - outlet.z);
  const chord = planeOffset < inflow.radius_m
    ? 2 * Math.sqrt(inflow.radius_m ** 2 - planeOffset ** 2) : 0;
  const requested = f(inPlaneSpeed * chord * strength / (h * h));

  const dominantAxis = Math.abs(direction.y) > Math.abs(direction.x)
    && Math.abs(direction.y) >= Math.abs(direction.z) ? 1
    : Math.abs(direction.z) > Math.abs(direction.x) ? 2 : 0;
  for (const row of topology.rows) {
    if (dominantAxis === 2 || row.axis !== dominantAxis || openFraction[row.id]! <= 0) continue;
    const point = worldPoint(seed, row.center);
    const relative = { x: point.x - outlet.x, y: point.y - outlet.y,
      z: point.z - outlet.z };
    const axial = relative.x * direction.x + relative.y * direction.y
      + relative.z * direction.z;
    const radial = Math.hypot(relative.x - axial * direction.x,
      relative.y - axial * direction.y, relative.z - axial * direction.z);
    if (Math.abs(axial / h) > 0.51 * Math.max(row.distance, 1)) continue;
    const edge = Math.max(0.5 * row.distance, 0.5) * h;
    // Coverage is geometric. Production applies timing to desired velocity,
    // rather than shrinking the nozzle disk while it ramps.
    rowCoverage[row.id] = f(clamp01(0.5 - (radial - inflow.radius_m) / edge));
  }

  const weights = new Float32Array(topology.cells.length);
  const eligible = new Uint8Array(topology.cells.length);
  for (const cell of topology.cells) {
    const point = worldPoint(seed, cell.center);
    const relative = { x: point.x - outlet.x, y: point.y - outlet.y,
      z: point.z - outlet.z };
    const axial = (relative.x * direction.x + relative.y * direction.y
      + relative.z * direction.z) / h;
    const radial = Math.hypot(relative.x - axial * h * direction.x,
      relative.y - axial * h * direction.y,
      relative.z - axial * h * direction.z) / h;
    const minimumWidth = Math.min(cell.widths[0], cell.widths[1]);
    const lengthFine = Math.max(effectiveSpeed * input.dt_s / h, minimumWidth);
    const edge = Math.max(0.5 * minimumWidth, 0.5);
    const coverage = clamp01(0.5 - (radial - inflow.radius_m / h) / edge)
      * clamp01(0.5 + axial / edge) * clamp01(0.5 + (lengthFine - axial) / edge);
    const weight = f(coverage * capacity[cell.id]! * cell.area);
    weights[cell.id] = weight;
    const fill = (input.density?.[cell.id] ?? 0) / Math.max(capacity[cell.id]!, 1e-6);
    eligible[cell.id] = capacity[cell.id]! > 1e-8
      && (fill >= 0.5 || (input.pressureMember?.[cell.id] ?? 0) !== 0 || weight > 0) ? 1 : 0;
  }

  // Production's continuous source union/find connects gsCell members across
  // every accepted open row, then admits weight only on components with a
  // non-zero row-coefficient sum (a pressure/air anchor).
  const parent = Int32Array.from(topology.cells, cell => eligible[cell.id] ? cell.id : -1);
  const root = (id: number): number => {
    let current = id;
    for (let depth = 0; depth < 64 && parent[current] !== current; depth += 1) {
      current = parent[current]!;
    }
    return current;
  };
  for (const row of topology.rows) {
    if (openFraction[row.id]! <= 0) continue;
    const members = row.terms.filter(term => eligible[term.cellId]).map(term => root(term.cellId));
    if (!members.length) continue;
    const minimum = Math.min(...members);
    for (const member of members) parent[root(member)] = minimum;
  }
  for (let id = 0; id < parent.length; id += 1) if (parent[id]! >= 0) parent[id] = root(id);
  const componentRoots = new Set<number>();
  const anchored = new Set<number>();
  for (let id = 0; id < parent.length; id += 1) if (parent[id]! >= 0) componentRoots.add(parent[id]!);
  for (const row of topology.rows) {
    if (openFraction[row.id]! <= 0) continue;
    let rowRoot = -1, sum = 0, scale = 0;
    for (const term of row.terms) {
      if (!eligible[term.cellId]) continue;
      const candidate = parent[term.cellId]!;
      if (rowRoot < 0) rowRoot = candidate;
      else if (rowRoot !== candidate) continue;
      sum = f(sum + term.coefficient); scale = f(scale + Math.abs(term.coefficient));
    }
    if (rowRoot >= 0 && Math.abs(sum) > 9.5367431640625e-7 * scale) anchored.add(rowRoot);
  }
  for (const cell of topology.cells) {
    if (!anchored.has(parent[cell.id]!)) weights[cell.id] = 0;
  }
  const total = productionGroupedSum(weights, input.sourceReductionGroups);
  const previousPending = f(input.pendingSourceAreaFine ?? 0);
  const requestedArea = f(requested * input.dt_s);
  const pendingIncrement = f(requestedArea - (input.pendingSourceCompensation ?? 0));
  const pending = f(previousPending + pendingIncrement);
  const factor = total > 0 && pending > 0
    ? f(f(Math.min(pending, total) / total) / input.dt_s) : 0;
  for (const cell of topology.cells) {
    rates[cell.id] = f(weights[cell.id]! * factor);
  }
  const planned = productionGroupedSum(rates, input.sourceReductionGroups);
  return { rates, rowCoverage, velocityFine, requested, planned, available: total, factor,
    components: componentRoots.size, anchoredComponents: anchored.size };
}

/**
 * Reduce production dynamic geometry onto the accepted 2-D compact topology.
 * The caller owns rigid integration and the source ledger; this function owns
 * the same capacity/row quadrature and source geometry used by their stages.
 */
export function sliceDynamicGeometry(input: SliceDynamicGeometryInput): SliceDynamicGeometry {
  if (!(input.dt_s > 0) || !Number.isFinite(input.dt_s)) {
    throw new RangeError("slice dynamic geometry dt must be finite and positive");
  }
  const authored = input.seed.production?.scene.rigidBodies.map(authoredPose) ?? [];
  const bodies = input.bodies ?? authored;
  const previous = input.previousBodies ?? bodies;
  const capacityAfter = compactCapacity(input.seed, input.topology, bodies);
  const capacityBefore = compactCapacity(input.seed, input.topology, previous);
  const capacityRate = Float32Array.from(capacityAfter,
    (value, index) => f((value - capacityBefore[index]!) / input.dt_s));
  const rowsAfter = dynamicRows(input.seed, input.topology, bodies);
  const rowsBefore = dynamicRows(input.seed, input.topology, previous);
  const meanOpenFraction = Float32Array.from(rowsAfter.openFraction,
    (value, index) => f(0.5 * f(rowsBefore.openFraction[index]! + value)));
  const inflow = inflowFields(input, capacityAfter, meanOpenFraction);
  return { capacity: capacityAfter, capacityBefore, capacityAfter, capacityRate,
    sourceRate: inflow.rates, openFraction: meanOpenFraction,
    openFractionBefore: rowsBefore.openFraction,
    openFractionAfter: rowsAfter.openFraction, meanOpenFraction,
    solidVelocity: rowsAfter.solidVelocity,
    inflowCoverage: inflow.rowCoverage, inflowVelocityFine: inflow.velocityFine,
    requestedSourceAreaFine: inflow.requested,
    sourceRateAreaFine: inflow.planned, sourceAvailableAreaFine: inflow.available,
    sourceFactor: inflow.factor, sourceComponentCount: inflow.components,
    sourceAnchoredComponentCount: inflow.anchoredComponents };
}
