/**
 * Surface-driven environment refinement for the live sparse-brick planner.
 *
 * The shipped predicate splits an environment node for two reasons: the ground
 * surface passes through it, or more than
 * `OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET` primitive bounding boxes
 * overlap it. The second is a *cost* control — the voxeliser binds a fixed
 * number of candidates per leaf and silently drops the surplus — and it was
 * standing in for a *detail* rule it cannot express. A mushroom cap is one
 * primitive, so its node holds one candidate and never splits however deep the
 * tree is asked to go: the ground refines, the scenery does not.
 *
 * The rule here is the one the shape actually implies. A node splits when a
 * surface passes through it, unless that surface is a **plane parallel to a
 * coordinate plane** across the whole node. The crowding test stays, unioned,
 * because its own reason is still true.
 *
 * ## The exemption is opt-in, and off
 *
 * That "unless" is `planarExemption`, and it defaults to **false**: without it
 * every straddling node splits and the centre sample is the whole predicate.
 * The exemption is a second-order test — the residual against the centre's own
 * tangent plane, and for the ground the spread of its column extremes — so what
 * it buys is depth *not* spent where the field's curvature is small. What it
 * costs is that the surface then arrives at the screen as a coarse leaf: a
 * mound's near-level cap keeps its parent's voxels while the slope around it
 * refines, and the flat top of one of those voxels is a large axis-aligned
 * facet with a straight octree boundary against its refined neighbour. On a
 * broad shallow dome that reads as gridding.
 *
 * It is exposed as `SvoRenderTuning.environmentPlanarRefinementExemption` and
 * carried to the build, rather than being deleted, because the argument below
 * is still sound where it applies — a node the ground crosses *dead level* is
 * one a finer leaf would record identically — and because the page counts in
 * `svoEnvironmentRefinementMode` are the measured price of turning it off.
 *
 * ## Why the exemption is axis-alignment and not flatness
 *
 * This shipped first as "flat to within 2.56 degrees and a quarter cell", and
 * that is the wrong invariant. The primary is voxels-only by default
 * (`voxelsOnlyPrimary`, `lib/webgpu-svo-dry-scene.ts:2396`): it returns the
 * first solid cell along the DDA and shades it with `dryVoxelFaceNormal`, one
 * of six axis directions read off that cell's own bounds. So what reaches the
 * screen is a staircase whose amplitude is **one voxel, however flat the
 * surface is**. An oblique plane terraces at the coarse level and terraces
 * half as much at the fine one; only a plane lying parallel to a voxel face is
 * *exactly* representable, and only there does declining to subdivide cost
 * nothing at all.
 *
 * That is also what the ask actually said — "flat surfaces (0 gradient, e.g.
 * for smooth floors and walls)". Zero gradient on a height field is level
 * ground, which is exactly the axis-aligned case.
 *
 * The second reason is that a tolerance on *curvature* is a threshold a single
 * object can straddle. Measured on `garden-svo-lighting` at refinement depth 3
 * under the old rule, `garden/tree-knoll/canopy-main` — one ellipsoid — halted
 * 34 of its 6 357 leaves one level early because its flattest region cleared
 * 2.56 degrees while the rest of the same surface did not: patches of coarse
 * inside one smooth object, with straight octree boundaries between them.
 * Axis-alignment is not a threshold, it is a property authored geometry either
 * has or does not, so no smooth surface can straddle it.
 *
 * The alternative considered and rejected was to halt on the *screen-space*
 * budget the ray traversal already uses (`lodScreenSpacePixels`), so build time
 * and render time ask one question. Two things stop it. The tree is built once
 * per scene and the budget is a function of the camera, so honouring it means
 * rebuilding the topology on camera motion — and `lodScreenSpacePixels` already
 * terminates *traversal* per frame, which is the cheap half of that idea and is
 * shipped. It also would not fix this: one object spans a range of projected
 * sizes, so a projected-size threshold produces the same patchwork along a
 * different axis. Screen space belongs in the descent, not in the topology.
 *
 * Both halves of the split test are conservative in the direction that costs
 * pages rather than geometry: an unresolvable normal, a normal off-axis or off
 * its neighbours, and a distance the kind under-reports all answer "refine".
 */
import type { SparseBrickCoordinate } from "./sparse-brick-octree";
import {
  sparseSceneTerrainNodeColumnRange,
  sparseSceneTerrainNodeCoverage,
  type SparseSceneTerrainDomain,
  type SparseSceneTerrainField,
} from "../core/sparse-scene-terrain-field";
import { canonicalSvoPrimitive, sampleSvoPrimitive, type SvoPrimitiveDescriptor } from "./svo-primitive-abi";
import type { EnvironmentProxyPrimitive } from "../core/voxel-environments";

/** The lever's variable, named once so a test and a doc cannot drift from it. */
export const SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE = "FLUID_SVO_REFINEMENT_MODE";

export type SvoEnvironmentRefinementMode = "candidates" | "surface";

/**
 * Which detail rule the environment planner uses. `candidates` ships.
 *
 * `candidates` is the historical predicate byte for byte — ground surface, or
 * more overlapping primitive boxes than the per-leaf candidate budget.
 * `surface` is the rule at the top of this file, applied to the ground as well
 * as to scenery.
 *
 * A lever rather than a constant because the two arms differ by a large factor
 * in page count and the whole point of the measurement is to say by how much;
 * an A/B that needs a constant edited between arms cannot be interleaved.
 *
 * Measured with `tmp/sp18/census.ts`, node-mip pyramid pages. The middle column
 * is the first `surface` rule — flat to 2.56 degrees — and is here because the
 * difference between it and the shipped one is the price of the axis
 * requirement, and that price is the whole argument:
 *
 * | scene / depth              | `candidates` | flat-to-2.56 | axis-aligned |
 * |----------------------------|--------------|--------------|--------------|
 * | `garden-svo-lighting` d0   |        2 201 |        2 201 |        2 201 |
 * | `garden-svo-lighting` d1   |        4 889 |        9 396 |        9 484 |
 * | `garden-svo-lighting` d2   |       16 429 |       45 024 |       46 165 |
 * | `hero-garden-hose` d0      |       12 213 |       12 213 |       12 213 |
 * | `hero-garden-hose` d1      |       25 858 |       26 928 |       31 018 |
 * | `hero-garden-hose` d2      |       76 812 |      100 888 |      124 080 |
 *
 * Depth 0 is identical by construction — the predicate is only consulted below
 * the solver's own level.
 *
 * Against `candidates` the shipped rule is **+181 % on the lighting study and
 * +62 % on the hero** at depth 2. Against the flat-to-2.56 rule it is +2.5 %
 * and +23 %, and those two numbers are not the same change: the primitive half
 * costs almost nothing (the lighting study is nearly all scenery, and its
 * primitive splits moved 5 976 -> 5 936), while the hero's +23 % is almost
 * entirely the **ground**. Its pond basin is a large smooth slope, so
 * requiring level rather than planar took its terrain exemptions from 3 198 to
 * 660 and its terrain splits from 3 137 to 8 353.
 *
 * That is the real bill for this rule and it should be read before raising the
 * depth: a sloped ground is the hero's largest surface, and it now refines.
 *
 * It is also *cheaper to decide*. The axis gate settles a curved node on one
 * sample instead of nine, so the lighting study's stencil samples fell 7 040 ->
 * 1 815 and its planner 523 ms -> 338 ms while building a 2.8x larger tree.
 */
export function svoEnvironmentRefinementMode(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): SvoEnvironmentRefinementMode {
  const raw = environment?.[SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE];
  if (raw === undefined || raw === "") return "candidates";
  if (raw !== "candidates" && raw !== "surface") {
    throw new RangeError(`${SVO_ENVIRONMENT_REFINEMENT_MODE_ENVIRONMENT_VARIABLE} must be "candidates" or "surface"`);
  }
  return raw;
}

/**
 * How close to a coordinate axis a normal must be to count as *on* that axis.
 *
 * This is an **exactness epsilon, not a quality knob**. An authored face that
 * is parallel to a voxel face reports its normal exactly: the box branch of
 * `sampleSvoPrimitive` returns a literal `(0, ±1, 0)`, the cylinder cap the
 * same, and an unrotated record's `worldNormal` passes it through untouched. A
 * record rotated by a right angle costs a few ulp. Everything else — a bevel,
 * a slope, a curve — is off axis by milliradians at least, which is four
 * orders of magnitude above this.
 *
 * So there is nothing to tune here, and that is the point. `1e-6` admits float
 * noise and nothing else, which is what makes the exemption categorical rather
 * than a threshold a smooth object can straddle in one place and not another.
 */
export const SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE = 1 - 1e-6;

/**
 * Which of the six axis directions a normal is, or -1 for none of them.
 *
 * The comparison is against the *world* axes because the voxel lattice is
 * axis-aligned in world space; a plane is exactly representable when it is
 * parallel to a voxel face, and voxel faces have no other orientation.
 */
export function svoEnvironmentNormalAxis(normal: { x: number; y: number; z: number }): number {
  if (normal.x >= SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 0;
  if (normal.x <= -SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 1;
  if (normal.y >= SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 2;
  if (normal.y <= -SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 3;
  if (normal.z >= SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 4;
  if (normal.z <= -SVO_ENVIRONMENT_PLANARITY_AXIS_COSINE) return 5;
  return -1;
}

/**
 * How far the field may depart from its own tangent plane across the node,
 * measured in the node's own cells.
 *
 * The leaf resolves the surface at `nodeEdge / brickSize`, so a departure below
 * a fraction of one cell is a departure the finer leaf could not have recorded
 * either — subdividing buys nothing. A quarter cell is deliberately strict: it
 * is the scale at which the voxel payload's own quantisation starts to matter,
 * and erring low costs pages rather than geometry.
 */
export const SVO_ENVIRONMENT_PLANARITY_RESIDUAL_CELLS = 0.25;

/** Counters for the build-cost report; none of them change a decision. */
export interface SvoEnvironmentRefinementStatistics {
  /** Predicate invocations. */
  nodes: number;
  /** Nodes whose candidate set was rebuilt from the full catalogue. */
  fullScans: number;
  /** Nodes whose candidate set was narrowed from the parent's. */
  narrowedScans: number;
  /** Candidate slots visited while narrowing. */
  candidateVisits: number;
  /** `sampleSvoPrimitive` calls made by the straddle test. */
  straddleSamples: number;
  /** `sampleSvoPrimitive` calls made by the planarity stencil. */
  planaritySamples: number;
  /** Nodes split because the ground is non-planar across them. */
  terrainSplits: number;
  /** Nodes the ground passes through but flatly, and so did not split. */
  terrainPlanarExemptions: number;
  /** Nodes split because a primitive surface is non-planar across them. */
  primitiveSplits: number;
  /** Nodes split by the crowding budget alone. */
  crowdingSplits: number;
}

export interface SvoEnvironmentRefinementOptions {
  readonly primitives: readonly EnvironmentProxyPrimitive[];
  /**
   * The SVO record a proxy publishes — in practice
   * `svoDescriptorForEnvironmentProxy` bound to the scene.
   *
   * Injected rather than imported because `svo-scene-primitives` already
   * depends on `webgpu-octree-sparse-bricks`, and this module is reached *from*
   * that world; importing the mapping here would close the cycle
   * `lib/webgpu-live-svo-scene.ts:172-176` exists to keep open. The caller that
   * already owns the build supplies it.
   */
  readonly descriptorFor: (primitive: EnvironmentProxyPrimitive) => SvoPrimitiveDescriptor;
  readonly worldOrigin_m: readonly [number, number, number];
  /** Node edge in metres, indexed by level. The planner's own table. */
  readonly nodeEdge_m: readonly (readonly number[])[];
  readonly brickSize: number;
  readonly maximumDepth: number;
  readonly terrainField?: SparseSceneTerrainField;
  readonly terrainDomain: SparseSceneTerrainDomain;
  /** Overlapping primitive boxes above which the node splits regardless. */
  readonly crowdingTarget: number;
  /**
   * Whether a node the surface crosses may decline to subdivide because that
   * surface is flat across it. See the header: off, every straddling node
   * splits, and the second-order tests below are never taken.
   *
   * Omitted means off, which is the product default. Defaulting the other way
   * would hand the exemption — and the coarse leaves it leaves on a near-level
   * surface — to any caller that simply did not know to ask about it.
   */
  readonly planarExemption?: boolean;
}

export interface SvoEnvironmentRefinement {
  refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean;
  readonly statistics: SvoEnvironmentRefinementStatistics;
}

/**
 * Whether the ground is *level* across one node, and so exactly representable
 * without subdividing it.
 *
 * Level, not merely planar. A height field's surface normal is on the Y axis
 * exactly when its gradient is zero, so this is the same test the primitive
 * half applies — see the header — expressed in the one coordinate a height
 * field has. A constant slope is a plane, but it is an *oblique* plane: it
 * terraces at one voxel however fine the leaf, and halving the leaf halves the
 * step, so the finer leaf is worth having.
 *
 * One pyramid lookup, no corners: level ground is exactly the case where the
 * column extremes over the whole footprint coincide, and the extremes are the
 * lookup {@link sparseSceneTerrainNodeCoverage} already makes. Cheaper than
 * evaluating a field, and available because the ground *is* a height field —
 * `terrainColumnHeightsForLattice` baked it at the octree's own finest lattice
 * before any of this ran.
 */
export function sparseSceneTerrainNodePlanar(
  field: SparseSceneTerrainField,
  brickSize: number,
  level: number,
  maximumDepth: number,
  coordinate: SparseBrickCoordinate,
  tolerance_m: number,
): boolean {
  const range = sparseSceneTerrainNodeColumnRange(field, brickSize, level, maximumDepth, coordinate);
  return range.maximum_m - range.minimum_m <= tolerance_m;
}

/**
 * Where the node is sampled, in units of its own half-extent.
 *
 * Eight corners and six face centres. Corners alone have a blind spot that
 * matters for the residual half of the test: for a saddle the second-order term
 * is `(k1*u^2 + k2*v^2)/2`, and at a corner `|u| = |v|`, so a surface with
 * `k1 = -k2` — a torus's inner band, a smooth-union blend — reads as exactly
 * planar at every corner while a face centre sees the full `|k|h^2/2`. The
 * extra six are affordable because the axis gate now runs first and rejects
 * every curved node before any of them is taken.
 */
const STENCIL_SIGNS: readonly (readonly [number, number, number])[] = Object.freeze([
  [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
  [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
  [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
].map((offset) => Object.freeze(offset as [number, number, number])));

/** What one primitive's surface does across one node. */
export interface SvoEnvironmentPrimitiveNodeProbe {
  /** Whether the node's enclosing sphere straddles this primitive's surface. */
  straddles: boolean;
  /** Whether the surface is flat enough across the node to skip subdividing. */
  planar: boolean;
  /** Signed distance at the node centre. */
  centreDistance_m: number;
  /** The centre's surface normal, or null where the kind has no tangent plane. */
  normal: { x: number; y: number; z: number } | null;
  /** Which axis the centre normal is, or -1 for none. See {@link svoEnvironmentNormalAxis}. */
  centreAxis: number;
  /** Worst stencil alignment against the centre normal; 1 is exactly parallel. */
  worstAlignment: number;
  /** Worst departure from the centre's tangent plane, in metres. */
  worstResidual_m: number;
  /** `sampleSvoPrimitive` calls this probe made. */
  samples: number;
}

export interface SvoEnvironmentPrimitiveNodeProbeOptions {
  /**
   * Evaluate the whole stencil instead of returning at the first failure. Only
   * a diagnostic wants this — the predicate's answer is a boolean and the early
   * return is most of why it is affordable. It also overrides the exemption
   * gate below, so a probe can report the margins of a decision the shipped
   * policy no longer consults.
   */
  readonly exhaustive?: boolean;
  /**
   * Whether flatness may excuse a straddling node from subdividing at all.
   * Off — the default, and the product's — the centre sample is the whole
   * answer: nothing is exempt, so there is nothing the stencil could establish.
   */
  readonly planarExemption?: boolean;
}

/**
 * Whether one primitive's surface crosses one node, and whether flatly.
 *
 * Extracted from the predicate so the criterion is a unit rather than a loop
 * body: a probe can ask it for the margins behind a decision without a second
 * copy of the arithmetic, and a test can pin one shape against one node.
 */
export function svoEnvironmentPrimitiveNodeProbe(
  descriptor: SvoPrimitiveDescriptor,
  centre: { x: number; y: number; z: number },
  half: { x: number; y: number; z: number },
  tolerance_m: number,
  options: SvoEnvironmentPrimitiveNodeProbeOptions = {},
): SvoEnvironmentPrimitiveNodeProbe {
  const { exhaustive = false, planarExemption = false } = options;
  const circumradius_m = Math.hypot(half.x, half.y, half.z);
  const sample = sampleSvoPrimitive(descriptor, centre);
  const centreDistance_m = sample.signedDistance_m;
  const result: SvoEnvironmentPrimitiveNodeProbe = {
    straddles: true, planar: false, centreDistance_m, normal: sample.normal,
    centreAxis: -1, worstAlignment: 1, worstResidual_m: 0, samples: 1,
  };
  if (!Number.isFinite(centreDistance_m)) return result;
  if (Math.abs(centreDistance_m) > circumradius_m) {
    result.straddles = false;
    result.planar = true;
    return result;
  }
  // No exemption, so the surface being here is the whole answer and `planar`
  // stays false: there is no flatness this policy would accept, and the second
  // order the stencil measures is exactly what it declines to spend depth on.
  if (!planarExemption && !exhaustive) return result;
  const normal = sample.normal;
  // No tangent plane exists here — a torus axis, an ambiguous ellipsoid pole —
  // so there is nothing to be flat with respect to.
  if (!normal) return result;
  const axis = svoEnvironmentNormalAxis(normal);
  result.centreAxis = axis;
  // The gate, and the reason this predicate is cheaper than the one it
  // replaced: a curved or oblique surface is settled by the centre sample alone
  // and never pays for the stencil. On a scene whose expensive kind costs 13 us
  // a sample that is the difference between 1 and 15 of them.
  if (axis < 0 && !exhaustive) return result;
  let planar = axis >= 0;
  for (const [signX, signY, signZ] of STENCIL_SIGNS) {
    const offsetX = signX * half.x, offsetY = signY * half.y, offsetZ = signZ * half.z;
    result.samples += 1;
    const at = sampleSvoPrimitive(descriptor, {
      x: centre.x + offsetX, y: centre.y + offsetY, z: centre.z + offsetZ,
    });
    const predicted_m = centreDistance_m + normal.x * offsetX + normal.y * offsetY + normal.z * offsetZ;
    const residual_m = Math.abs(at.signedDistance_m - predicted_m);
    if (residual_m > result.worstResidual_m) result.worstResidual_m = residual_m;
    const there = at.normal;
    const alignment = there ? there.x * normal.x + there.y * normal.y + there.z * normal.z : -1;
    if (alignment < result.worstAlignment) result.worstAlignment = alignment;
    // The same face of the same plane, still parallel to the same voxel face.
    // A residual is what catches a field that keeps one hard-feature normal
    // across a region where it is no longer affine — every box beyond an edge.
    if (residual_m > tolerance_m || !there || svoEnvironmentNormalAxis(there) !== axis) {
      planar = false;
      if (!exhaustive) break;
    }
  }
  result.planar = planar;
  return result;
}

/**
 * The curvature-driven `refineEnvironmentLeaf`, with its candidate narrowing.
 *
 * The candidate set is carried down the recursion rather than rebuilt per node.
 * A node's overlapping primitives are a subset of its parent's, and the
 * planner's descent is depth-first — a parent is asked immediately before its
 * children — so one array per level is enough state to turn a per-node scan of
 * the whole catalogue into a scan of the parent's answer. At environment
 * refinement depth 3 the planner visits a quarter of a million nodes; the
 * difference is the whole catalogue each time against a list that halves.
 */
export function createSvoEnvironmentRefinement(
  options: SvoEnvironmentRefinementOptions,
): SvoEnvironmentRefinement {
  const {
    primitives, descriptorFor, worldOrigin_m: worldOrigin, nodeEdge_m,
    brickSize, maximumDepth, terrainField, terrainDomain, crowdingTarget,
    planarExemption = false,
  } = options;
  const statistics: SvoEnvironmentRefinementStatistics = {
    nodes: 0, fullScans: 0, narrowedScans: 0, candidateVisits: 0,
    straddleSamples: 0, planaritySamples: 0,
    terrainSplits: 0, terrainPlanarExemptions: 0, primitiveSplits: 0, crowdingSplits: 0,
  };
  const count = primitives.length;
  // Flat bounds, so narrowing is six typed-array reads rather than two object
  // dereferences and six property loads per candidate per node.
  const bounds = new Float64Array(count * 6);
  for (let index = 0; index < count; index += 1) {
    const { min, max } = primitives[index].aabb_m;
    bounds[index * 6] = min.x; bounds[index * 6 + 1] = min.y; bounds[index * 6 + 2] = min.z;
    bounds[index * 6 + 3] = max.x; bounds[index * 6 + 4] = max.y; bounds[index * 6 + 5] = max.z;
  }
  /**
   * Canonicalised once. `sampleSvoPrimitive` canonicalises its input on every
   * call, which is idempotent but allocates; handing it a descriptor that is
   * already canonical leaves only the copy.
   */
  const descriptors: SvoPrimitiveDescriptor[] = primitives.map(
    (primitive) => canonicalSvoPrimitive(descriptorFor(primitive)));
  const everything: number[] = [];
  for (let index = 0; index < count; index += 1) everything.push(index);

  interface ChainEntry { x: number; y: number; z: number; items: number[] }
  const chain: (ChainEntry | undefined)[] = [];

  const narrow = (level: number, coordinate: SparseBrickCoordinate): number[] => {
    const cached = chain[level];
    if (cached && cached.x === coordinate.x && cached.y === coordinate.y && cached.z === coordinate.z) {
      return cached.items;
    }
    const parent = level > 0 ? chain[level - 1] : undefined;
    const source = parent
      && parent.x === (coordinate.x >> 1) && parent.y === (coordinate.y >> 1) && parent.z === (coordinate.z >> 1)
      ? parent.items : everything;
    if (source === everything) statistics.fullScans += 1; else statistics.narrowedScans += 1;
    statistics.candidateVisits += source.length;
    const edge = nodeEdge_m[level];
    const loX = worldOrigin[0] + coordinate.x * edge[0], hiX = loX + edge[0];
    const loY = worldOrigin[1] + coordinate.y * edge[1], hiY = loY + edge[1];
    const loZ = worldOrigin[2] + coordinate.z * edge[2], hiZ = loZ + edge[2];
    const items: number[] = [];
    for (const index of source) {
      const base = index * 6;
      if (bounds[base + 3] < loX || bounds[base] > hiX) continue;
      if (bounds[base + 4] < loY || bounds[base + 1] > hiY) continue;
      if (bounds[base + 5] < loZ || bounds[base + 2] > hiZ) continue;
      items.push(index);
    }
    chain[level] = { x: coordinate.x, y: coordinate.y, z: coordinate.z, items };
    return items;
  };

  /**
   * Whether any candidate's surface passes non-planarly through the node.
   *
   * One evaluation at the centre settles the straddle: every kind reachable
   * from an authored proxy reports a distance that never *over*-states the
   * clearance to its own surface (`SparseSceneSolidReach`'s contract, and the
   * per-kind audit behind it), so `|d| > circumradius` proves the surface
   * misses the node's enclosing sphere and therefore the node. Only a candidate
   * that survives that pays for the eight-corner stencil.
   *
   * The two terms are five orders of magnitude apart per operation, which is
   * why the narrowing above exists, why the straddle test is a hard gate rather
   * than a hint, and why the axis gate inside the probe runs before the stencil.
   * Measured on `garden-svo-lighting`: 3.8 ns for one AABB overlap, 0.6-1.8 us
   * for one `sampleSvoPrimitive` on an exact kind, and **13.3 us on a
   * `smooth-union-cluster`** — the foliage field, whose distance walks a lobe
   * neighbourhood and whose normal is a four-tap on top of it. A node that
   * reaches the stencil pays fifteen of those; a curved one pays one. Foliage is
   * curved, so the gate is worth more than the stencil costs: the lighting
   * study's stencil samples fell 7 040 -> 1 815 when it landed.
   */
  const surfaceSplits = (level: number, coordinate: SparseBrickCoordinate, items: readonly number[]): boolean => {
    if (items.length === 0) return false;
    const edge = nodeEdge_m[level];
    const halfX = 0.5 * edge[0], halfY = 0.5 * edge[1], halfZ = 0.5 * edge[2];
    const centreX = worldOrigin[0] + coordinate.x * edge[0] + halfX;
    const centreY = worldOrigin[1] + coordinate.y * edge[1] + halfY;
    const centreZ = worldOrigin[2] + coordinate.z * edge[2] + halfZ;
    const tolerance_m = SVO_ENVIRONMENT_PLANARITY_RESIDUAL_CELLS
      * Math.min(edge[0], edge[1], edge[2]) / brickSize;
    const centre = { x: centreX, y: centreY, z: centreZ };
    const half = { x: halfX, y: halfY, z: halfZ };
    let planarAxis = -1;
    for (const index of items) {
      const probe = svoEnvironmentPrimitiveNodeProbe(
        descriptors[index], centre, half, tolerance_m, { planarExemption });
      statistics.straddleSamples += 1;
      statistics.planaritySamples += probe.samples - 1;
      if (!probe.straddles) continue;
      if (!probe.planar) return true;
      // Two flat faces meeting inside one node is a crease, and a crease is a
      // feature: neither face's plane describes the node's contents.
      if (planarAxis >= 0 && probe.centreAxis !== planarAxis) return true;
      planarAxis = probe.centreAxis;
    }
    return false;
  };

  return {
    statistics,
    refineEnvironmentLeaf(level: number, coordinate: SparseBrickCoordinate): boolean {
      statistics.nodes += 1;
      const items = narrow(level, coordinate);
      if (terrainField && sparseSceneTerrainNodeCoverage(
        terrainField, terrainDomain, brickSize, level, maximumDepth, coordinate) === "surface") {
        const tolerance_m = SVO_ENVIRONMENT_PLANARITY_RESIDUAL_CELLS * nodeEdge_m[level][1] / brickSize;
        if (!planarExemption
          || !sparseSceneTerrainNodePlanar(terrainField, brickSize, level, maximumDepth, coordinate, tolerance_m)) {
          // Without the exemption the coverage answer is the whole rule: ground
          // in the node is ground the finer leaf resolves at half the terrace.
          statistics.terrainSplits += 1;
          return true;
        }
        statistics.terrainPlanarExemptions += 1;
      }
      if (items.length > crowdingTarget) { statistics.crowdingSplits += 1; return true; }
      if (surfaceSplits(level, coordinate, items)) { statistics.primitiveSplits += 1; return true; }
      return false;
    },
  };
}
