import type { Vec3 } from "./model";

/**
 * The one table a rigid shape is declared in.
 *
 * A shape used to be five hand-edits, four of which failed *silently*. The
 * signed distance lived in `webgpu-uniform-reference.wgsl.ts` and again in
 * `octree-projection.wgsl.ts`; the support map and the volume lived in
 * `webgpu-rigid-body.ts`; the bounding radius lived in both solvers *and* in
 * `rigid-body.ts`, spelled two different ways. Nothing checked that the five
 * agreed. A shape added to four of them rendered, collided and displaced water
 * correctly and then leaked through its own walls in the fifth, with no error
 * anywhere.
 *
 * This follows `SVO_PRIMITIVE_KIND_TABLE` one level down. That table owns how a
 * shape is *drawn*; this one owns how it is *solved* — what is inside it, how
 * far away it is, how far it reaches along a direction, what it displaces, and
 * what it weighs. Both are keyed by an exhaustive `Record<SceneShapeName, T>`,
 * so a new shape does not type-check until every fact it owes has been stated.
 *
 * Each fact appears twice, in TypeScript and in WGSL, and that pairing is the
 * contract: `tests/scene-shape-parity.test.ts` evaluates both over a lattice of
 * points and holds them to 1e-6. Two languages is the floor — the host needs
 * the maths for mass properties and analytic picking, the GPU needs it for the
 * immersed boundary — but two *copies per consumer* was the actual problem, and
 * that is what is gone.
 *
 * ## Why the legacy spellings are preserved literally
 *
 * The four original shapes carry their expressions exactly as the shaders wrote
 * them, operation order included, rather than rephrased into whatever reads
 * best. Blessed trajectories in this repo are compared bit-identically, and
 * `length(vec2f(a, b))` and `sqrt(a*a + b*b)` do not always agree in the last
 * ulp. The one deliberate exception is noted on `boundingRadius` below.
 */
export type SceneShapeName = "sphere" | "box" | "capsule" | "cylinder" | "cup";

export interface SceneShapeKind {
  readonly name: SceneShapeName;
  /**
   * What a person calls it. Here rather than in a component because two
   * separate pickers already spelled the same four labels, and a shape added to
   * the table but not to both of them is a shape the solver can run and nobody
   * can place.
   */
  readonly label: string;
  /**
   * Stable GPU tag, packed into `positionShape.w` and rounded back out.
   * Never reassigned: recordings, captures and authored documents hold it.
   */
  readonly code: number;
  /** What the three dimension floats mean, in order. Documentation the packer can be checked against. */
  readonly dimensionLabels: readonly [string, string, string];
  /** Whether the shape encloses a cavity the fluid can occupy. Drives the resolution advice below. */
  readonly hollow: boolean;

  // ---- host maths -------------------------------------------------------

  /** Inside test in the shape's own frame. Not `distance <= 0` for the legacy four: see the note above. */
  readonly inside: (d: Vec3, local: Vec3) => boolean;
  /** Signed distance in the shape's own frame, negative inside, 1-Lipschitz. */
  readonly distance_m: (d: Vec3, local: Vec3) => number;
  /** Farthest reach along a unit direction in the shape's own frame — the collision support map. */
  readonly supportRadius_m: (d: Vec3, localDirection: Vec3) => number;
  /** Rotation-invariant radius about the centre. */
  readonly boundingRadius_m: (d: Vec3) => number;
  /** Displaced volume. */
  readonly volume_m3: (d: Vec3) => number;
  /** Principal inertia in the shape's own frame. Takes density so composite shapes keep their original operation order. */
  readonly inertia_kg_m2: (d: Vec3, density_kg_m3: number) => Vec3;

  // ---- presentation -----------------------------------------------------

  /**
   * Half-extents of the proxy box the legacy rigid raster draws the shape in.
   *
   * Held here because the same three lines were written out in the renderer, in
   * the smoke readbacks, in the dry-frame harness and again in WGSL, and a
   * fourth copy is how a shape ends up drawn one size in the app and another in
   * a capture. Note that the capsule's is its *segment* half height, so the two
   * caps stand outside this box: that is what the four copies already did, and
   * the SVO path — which is what actually draws a capsule today — carries the
   * corrected extent in `SVO_PRIMITIVE_KIND_TABLE`.
   */
  readonly renderHalfExtent_m: (d: Vec3) => readonly [number, number, number];
  /** Scene-linear base colour, indexed by shape code in every rigid palette. */
  readonly paletteLinear: readonly [number, number, number];
  /** Statements returning `vec3f`, the WGSL twin of `renderHalfExtent_m`. */
  readonly wgslRenderHalf: string;

  // ---- the same maths, in WGSL ------------------------------------------

  /** Statements returning `bool`. `d` is the dimension vec3f, `p` the local point. */
  readonly wgslInside: string;
  /** Statements returning `f32`. */
  readonly wgslDistance: string;
  /** Statements returning `f32`. `local` is the unit direction. */
  readonly wgslSupportRadius: string;
  /** Statements returning `f32`, from `d` alone. */
  readonly wgslBoundingRadius: string;
  /** Statements returning `f32`, from `d` alone. */
  readonly wgslVolume: string;
}

const TAU_OVER_THREE_TIMES_TWO = 4.1887902047863905;
const PI = 3.141592653589793;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/**
 * A cup's wall, clamped away from the degenerate ends.
 *
 * Authored thickness is a free field on a body a user can resize by dragging a
 * corner, so it has to survive being asked for a wall thicker than the cup. The
 * clamp is in both the host and the shader arm, at the same limits, because a
 * cup whose cavity the renderer draws and the solver denies is a cup that fills
 * with water on screen and holds none.
 */
export function cupWallThickness_m(d: Vec3): number {
  return clamp(d.z, 1e-4, Math.min(0.95 * d.x, 0.95 * d.y));
}

/** Radius of a cup's cavity. */
export function cupInnerRadius_m(d: Vec3): number {
  return d.x - cupWallThickness_m(d);
}

/** World-space height of a cup's inner floor above its centre. */
export function cupFloorOffset_m(d: Vec3): number {
  return -0.5 * d.y + cupWallThickness_m(d);
}

/**
 * A cup's signed distance in its own frame: the outer cylinder with a coaxial,
 * open-topped cavity subtracted. Standalone rather than inline so the table's
 * `inside` arm can reuse it without the table referencing itself while it is
 * still being initialized.
 */
export function cupDistance_m(d: Vec3, p: Vec3): number {
  const t = cupWallThickness_m(d);
  const outerQx = Math.hypot(p.x, p.z) - d.x;
  const outerQy = Math.abs(p.y) - 0.5 * d.y;
  const outer = Math.hypot(Math.max(outerQx, 0), Math.max(outerQy, 0)) + Math.min(Math.max(outerQx, outerQy), 0);
  const cavity = Math.max(Math.hypot(p.x, p.z) - (d.x - t), (-0.5 * d.y + t) - p.y);
  return Math.max(outer, -cavity);
}

/**
 * The name of the emitted cup distance function.
 *
 * The cup is the one arm long enough that inlining it into five ladders would
 * put five copies of it in a single shader. It is emitted once as a function
 * and every ladder calls it, which is the same reason this whole file exists,
 * one scale down.
 */
export const CUP_DISTANCE_FUNCTION = "sceneShapeCupDistance_m";

/**
 * The name of the emitted wall clamp.
 *
 * Emitted beside the distance rather than inlined into it because the SVO
 * renderer needs the same number to pick which of the cup's three authored
 * surfaces a shading point belongs to, and a second spelling of the clamp is a
 * second place for the cavity to move.
 */
export const CUP_WALL_FUNCTION = "sceneShapeCupWall_m";

const WGSL_CUP_PRELUDE = `let t = ${CUP_WALL_FUNCTION}(d);
  let outerQ = vec2f(length(p.xz) - d.x, abs(p.y) - 0.5 * d.y);
  let outer = length(max(outerQ, vec2f(0.0))) + min(max(outerQ.x, outerQ.y), 0.0);
  // The cavity is open upward by construction rather than by being tall: an
  // infinite-up cylinder capped by the inner floor. A finite cavity would need
  // a top far enough above the rim to never cap the cup, which is one more
  // number to get wrong, and a cavity whose lid sits just above a tall cup
  // turns the cup into a sealed flask that no water can enter.
  let cavity = max(length(p.xz) - (d.x - t), (-0.5 * d.y + t) - p.y);`;

/**
 * Every rigid shape, keyed by its TypeScript discriminant.
 *
 * Codes 0-3 are historical: they are the `shapeIndex` the renderer, both
 * solvers and every authored document already agree on.
 */
export const SCENE_SHAPE_TABLE = Object.freeze({
  sphere: {
    name: "sphere",
    label: "Sphere",
    code: 0,
    dimensionLabels: ["radius", "unused", "unused"],
    hollow: false,
    inside: (d, p) => Math.hypot(p.x, p.y, p.z) <= d.x,
    distance_m: (d, p) => Math.hypot(p.x, p.y, p.z) - d.x,
    supportRadius_m: (d) => d.x,
    boundingRadius_m: (d) => d.x,
    volume_m3: (d) => (4 / 3) * Math.PI * d.x ** 3,
    inertia_kg_m2: (d, density) => {
      const value = (2 / 5) * (density * ((4 / 3) * Math.PI * d.x ** 3)) * d.x ** 2;
      return { x: value, y: value, z: value };
    },
    renderHalfExtent_m: (d) => [d.x, d.x, d.x],
    paletteLinear: [0.95, 0.63, 0.29],
    wgslRenderHalf: `return vec3f(d.x);`,
    wgslInside: `return length(p) <= d.x;`,
    wgslDistance: `return length(p) - d.x;`,
    wgslSupportRadius: `return d.x;`,
    wgslBoundingRadius: `return d.x;`,
    wgslVolume: `return ${TAU_OVER_THREE_TIMES_TWO} * d.x * d.x * d.x;`,
  },
  box: {
    name: "box",
    label: "Box",
    code: 1,
    dimensionLabels: ["extent x", "extent y", "extent z"],
    hollow: false,
    inside: (d, p) => Math.abs(p.x) <= 0.5 * d.x && Math.abs(p.y) <= 0.5 * d.y && Math.abs(p.z) <= 0.5 * d.z,
    distance_m: (d, p) => {
      const q = { x: Math.abs(p.x) - 0.5 * d.x, y: Math.abs(p.y) - 0.5 * d.y, z: Math.abs(p.z) - 0.5 * d.z };
      const outside = Math.hypot(Math.max(q.x, 0), Math.max(q.y, 0), Math.max(q.z, 0));
      return outside + Math.min(Math.max(q.x, Math.max(q.y, q.z)), 0);
    },
    supportRadius_m: (d, local) => 0.5 * (Math.abs(local.x) * d.x + Math.abs(local.y) * d.y + Math.abs(local.z) * d.z),
    boundingRadius_m: (d) => 0.5 * Math.hypot(d.x, d.y, d.z),
    volume_m3: (d) => d.x * d.y * d.z,
    inertia_kg_m2: (d, density) => {
      const mass = density * d.x * d.y * d.z;
      return {
        x: mass * (d.y ** 2 + d.z ** 2) / 12,
        y: mass * (d.x ** 2 + d.z ** 2) / 12,
        z: mass * (d.x ** 2 + d.y ** 2) / 12,
      };
    },
    renderHalfExtent_m: (d) => [0.5 * d.x, 0.5 * d.y, 0.5 * d.z],
    paletteLinear: [0.48, 0.66, 0.96],
    wgslRenderHalf: `return 0.5 * d;`,
    wgslInside: `return all(abs(p) <= 0.5 * d);`,
    wgslDistance: `let q = abs(p) - 0.5 * d;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);`,
    wgslSupportRadius: `return 0.5 * (abs(local.x) * d.x + abs(local.y) * d.y + abs(local.z) * d.z);`,
    wgslBoundingRadius: `return 0.5 * length(d);`,
    wgslVolume: `return d.x * d.y * d.z;`,
  },
  capsule: {
    name: "capsule",
    label: "Capsule",
    code: 2,
    dimensionLabels: ["radius", "segment length", "unused"],
    hollow: false,
    inside: (d, p) => {
      const cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
      return Math.hypot(p.x, p.y - cy, p.z) <= d.x;
    },
    distance_m: (d, p) => {
      const cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
      return Math.hypot(p.x, p.y - cy, p.z) - d.x;
    },
    supportRadius_m: (d, local) => d.x + 0.5 * d.y * Math.abs(local.y),
    boundingRadius_m: (d) => d.x + 0.5 * d.y,
    volume_m3: (d) => Math.PI * d.x ** 2 * d.y + (4 / 3) * Math.PI * d.x ** 3,
    // Kept in the paper's own composite form — a cylinder plus two caps, each
    // weighed from density — rather than folded into a per-mass ratio. The
    // ratio is algebraically identical and not bit-identical, and this is the
    // inertia a blessed capsule trajectory was integrated with.
    inertia_kg_m2: (d, density) => {
      const radius = d.x;
      const cylinderLength = d.y;
      const cylinderMass = density * Math.PI * radius ** 2 * cylinderLength;
      const sphereMass = density * (4 / 3) * Math.PI * radius ** 3;
      const axial = 0.5 * cylinderMass * radius ** 2 + (2 / 5) * sphereMass * radius ** 2;
      const transverseCaps = sphereMass * ((83 / 320) * radius ** 2 + (cylinderLength / 2 + 3 * radius / 8) ** 2);
      const transverse = cylinderMass * (3 * radius ** 2 + cylinderLength ** 2) / 12 + transverseCaps;
      return { x: transverse, y: axial, z: transverse };
    },
    renderHalfExtent_m: (d) => [d.x, 0.5 * d.y, d.x],
    paletteLinear: [0.84, 0.42, 0.48],
    wgslRenderHalf: `return vec3f(d.x, 0.5 * d.y, d.x);`,
    wgslInside: `let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
  return length(vec3f(p.x, p.y - cy, p.z)) <= d.x;`,
    wgslDistance: `let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y);
  return length(vec3f(p.x, p.y - cy, p.z)) - d.x;`,
    wgslSupportRadius: `return d.x + 0.5 * d.y * abs(local.y);`,
    wgslBoundingRadius: `return d.x + 0.5 * d.y;`,
    wgslVolume: `return ${PI} * d.x * d.x * d.y + ${TAU_OVER_THREE_TIMES_TWO} * d.x * d.x * d.x;`,
  },
  cylinder: {
    name: "cylinder",
    label: "Cylinder",
    code: 3,
    dimensionLabels: ["radius", "height", "unused"],
    hollow: false,
    inside: (d, p) => p.x * p.x + p.z * p.z <= d.x * d.x && Math.abs(p.y) <= 0.5 * d.y,
    distance_m: (d, p) => {
      const qx = Math.hypot(p.x, p.z) - d.x;
      const qy = Math.abs(p.y) - 0.5 * d.y;
      return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
    },
    supportRadius_m: (d, local) => d.x * Math.hypot(local.x, local.z) + 0.5 * d.y * Math.abs(local.y),
    boundingRadius_m: (d) => Math.hypot(d.x, 0.5 * d.y),
    volume_m3: (d) => Math.PI * d.x ** 2 * d.y,
    inertia_kg_m2: (d, density) => {
      const mass = density * Math.PI * d.x ** 2 * d.y;
      return {
        x: mass * (3 * d.x ** 2 + d.y ** 2) / 12,
        y: 0.5 * mass * d.x ** 2,
        z: mass * (3 * d.x ** 2 + d.y ** 2) / 12,
      };
    },
    renderHalfExtent_m: (d) => [d.x, 0.5 * d.y, d.x],
    paletteLinear: [0.66, 0.52, 0.92],
    wgslRenderHalf: `return vec3f(d.x, 0.5 * d.y, d.x);`,
    wgslInside: `return p.x * p.x + p.z * p.z <= d.x * d.x && abs(p.y) <= 0.5 * d.y;`,
    wgslDistance: `let q = vec2f(length(p.xz) - d.x, abs(p.y) - 0.5 * d.y);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);`,
    wgslSupportRadius: `return d.x * length(local.xz) + 0.5 * d.y * abs(local.y);`,
    wgslBoundingRadius: `return length(vec2f(d.x, 0.5 * d.y));`,
    wgslVolume: `return ${PI} * d.x * d.x * d.y;`,
  },
  /**
   * An open-top vessel: the outer cylinder with a coaxial cavity subtracted.
   *
   * The first shape here that encloses a volume of fluid rather than displacing
   * one, which is the whole reason it exists — a cup you can dip, lift, carry
   * and pour is the shortest question a free-surface solver can be asked that
   * it cannot answer with a convex primitive.
   *
   * `max(outer, -cavity)` is the standard subtraction, and it is safe to sphere
   * trace: the max of two 1-Lipschitz functions is 1-Lipschitz, and negating a
   * distance preserves the bound. It is not *exact* just inside the rim — it
   * over-reports there, which is the conservative direction for a march and for
   * the solid fraction alike.
   *
   * **Resolution.** The wall has to be at least two cells thick to hold water.
   * `bodySolidFraction` samples eight corners at ±0.4 h, so a wall one cell
   * thick reads back a fraction near 0.5 on every cell it occupies, and a
   * half-solid cell is a face the projection lets water through. See
   * `cupResolutionAdvice` below, which states the cell size a given cup needs.
   */
  cup: {
    name: "cup",
    label: "Cup",
    code: 4,
    dimensionLabels: ["outer radius", "height", "wall thickness"],
    hollow: true,
    inside: (d, p) => cupDistance_m(d, p) <= 0,
    distance_m: cupDistance_m,
    // The outer silhouette: the cavity never reaches past the wall it is cut
    // from, so a cup supports exactly as far as the cylinder around it.
    supportRadius_m: (d, local) => d.x * Math.hypot(local.x, local.z) + 0.5 * d.y * Math.abs(local.y),
    boundingRadius_m: (d) => Math.hypot(d.x, 0.5 * d.y),
    volume_m3: (d) => {
      const t = cupWallThickness_m(d);
      return PI * (d.x * d.x * d.y - (d.x - t) * (d.x - t) * (d.y - t));
    },
    // A disc base plus a tube wall, each about its own centre and carried to
    // the cup's *frame origin* — its geometric centre — by the parallel axis
    // theorem.
    //
    // The frame origin, not the mass centre, because that is the point the rest
    // of the system already means by a body's position: the SDF is written
    // about it, the gizmo box is centred on it, and the integrator carries one
    // position per body. A cup's mass centre sits below it (the base is solid
    // and the walls are not), so a released cup does not right itself the way a
    // real one does — gravity is applied at the geometric centre, and there is
    // no offset for it to act through. Worth knowing before reading a tumble as
    // a solver defect; giving the integrator a mass-centre offset is a change
    // to every body, not to this shape.
    inertia_kg_m2: (d, density) => {
      const t = cupWallThickness_m(d);
      const outerRadius = d.x;
      const innerRadius = d.x - t;
      const baseMass = density * PI * outerRadius ** 2 * t;
      const baseCentre = -0.5 * d.y + 0.5 * t;
      const wallLength = d.y - t;
      const wallMass = density * PI * (outerRadius ** 2 - innerRadius ** 2) * wallLength;
      const wallCentre = 0.5 * t;
      const axial = 0.5 * baseMass * outerRadius ** 2
        + 0.5 * wallMass * (outerRadius ** 2 + innerRadius ** 2);
      const transverse = baseMass * (3 * outerRadius ** 2 + t ** 2) / 12 + baseMass * baseCentre ** 2
        + wallMass * (3 * (outerRadius ** 2 + innerRadius ** 2) + wallLength ** 2) / 12 + wallMass * wallCentre ** 2;
      return { x: transverse, y: axial, z: transverse };
    },
    renderHalfExtent_m: (d) => [d.x, 0.5 * d.y, d.x],
    paletteLinear: [0.86, 0.83, 0.76],
    wgslRenderHalf: `return vec3f(d.x, 0.5 * d.y, d.x);`,
    wgslInside: `return ${CUP_DISTANCE_FUNCTION}(d, p) <= 0.0;`,
    wgslDistance: `return ${CUP_DISTANCE_FUNCTION}(d, p);`,
    wgslSupportRadius: `return d.x * length(local.xz) + 0.5 * d.y * abs(local.y);`,
    wgslBoundingRadius: `return length(vec2f(d.x, 0.5 * d.y));`,
    wgslVolume: `let t = ${CUP_WALL_FUNCTION}(d);
  return ${PI} * (d.x * d.x * d.y - (d.x - t) * (d.x - t) * (d.y - t));`,
  },
} as const satisfies Record<SceneShapeName, SceneShapeKind>);

export const SCENE_SHAPE_NAMES = Object.freeze(
  Object.keys(SCENE_SHAPE_TABLE) as SceneShapeName[]) as readonly SceneShapeName[];

/** Shapes in GPU tag order, which is the order every emitted `switch` below is written in. */
export const SCENE_SHAPES_BY_CODE: readonly SceneShapeKind[] = Object.freeze(
  SCENE_SHAPE_NAMES.map((name) => SCENE_SHAPE_TABLE[name] as SceneShapeKind)
    .slice()
    .sort((a, b) => a.code - b.code));

export function sceneShape(name: SceneShapeName): SceneShapeKind {
  return SCENE_SHAPE_TABLE[name] as SceneShapeKind;
}

/** The GPU tag a shape is packed as. */
export function sceneShapeCode(name: SceneShapeName): number {
  return SCENE_SHAPE_TABLE[name].code;
}

/**
 * The rigid palette, indexed by shape code.
 *
 * One array rather than the four identical literals the renderer, the smoke
 * readbacks, the dry-frame harness and the publish shader each carried. Built
 * from the table so a colour cannot be added for a shape that does not exist,
 * or forgotten for one that does.
 */
export const SCENE_SHAPE_PALETTE_LINEAR: ReadonlyArray<readonly [number, number, number]> =
  Object.freeze(SCENE_SHAPES_BY_CODE.map((shape) => shape.paletteLinear));

/** Proxy half-extents the legacy rigid raster draws a body in. */
export function sceneShapeRenderHalfExtent_m(name: SceneShapeName, d: Vec3): readonly [number, number, number] {
  return sceneShape(name).renderHalfExtent_m(d);
}

/**
 * The finest cell a cup needs before it holds water, and what it has.
 *
 * Stated rather than enforced: a scene is allowed to carry an under-resolved
 * cup — half the point of dipping one is watching where the model gives out —
 * but a cup that pours out through its own base looks like a solver bug rather
 * than a sampling one, so the editor says which it is.
 */
export function cupResolutionAdvice(d: Vec3, cellSize_m: number): {
  readonly wallCells: number;
  readonly cavityCells: number;
  readonly holdsWater: boolean;
  readonly recommendedCellSize_m: number;
} {
  const wall = cupWallThickness_m(d);
  const wallCells = wall / cellSize_m;
  const cavityCells = 2 * cupInnerRadius_m(d) / cellSize_m;
  return {
    wallCells,
    cavityCells,
    // Two cells of wall is the floor: at one cell every wall cell reads back a
    // solid fraction near a half, and a half-solid face is one water passes.
    holdsWater: wallCells >= 2 && cavityCells >= 3,
    recommendedCellSize_m: Math.min(wall / 2, 2 * cupInnerRadius_m(d) / 4),
  };
}

// ---- WGSL emission ------------------------------------------------------

/**
 * Indent a shape arm's statements to sit inside an emitted `if` block.
 *
 * The arms are authored at the indentation a function body wants, so an arm
 * that spans lines (`box`, `cup`) reads correctly in the table *and* in the
 * generated shader. Without this the second line of every multi-line arm lands
 * at column three inside a block at column five, which is legal WGSL and
 * unreadable in a shader dump — and a shader dump is how every one of these is
 * actually debugged.
 */
function indentArm(body: string): string {
  return body.split("\n").map((line) => (line.trim().length > 0 ? `  ${line}` : "")).join("\n");
}

function emitDispatch(signature: string, arm: (shape: SceneShapeKind) => string, fallback: string): string {
  const arms = SCENE_SHAPES_BY_CODE
    .map((shape) => `  if (shape == ${shape.code}) {\n  ${indentArm(arm(shape))}\n  }`)
    .join("\n");
  return `fn ${signature} {\n${arms}\n  ${fallback}\n}`;
}

/**
 * Every shape fact, as one WGSL block.
 *
 * Emitted rather than written out because the alternative is what this file
 * replaced: the same four `if (shape == n)` ladders in three shaders, drifting.
 * A consumer pastes this once and calls `rigidShape*` from its own wrappers,
 * which is where the per-shader `RigidBody` struct — they differ — stays.
 *
 * Every ladder ends in an explicit neutral rather than in its last shape's arm.
 * The originals let an unrecognised tag fall through to the cylinder, so a
 * shape this build did not know about was silently solved as a solid cylinder;
 * a tag out of range now reads as absent, which is the direction that fails
 * visibly instead of filling a cup with rock.
 */
export function sceneShapeWgsl(): string {
  const constants = SCENE_SHAPES_BY_CODE
    .map((shape) => `const RIGID_SHAPE_${shape.name.toUpperCase()}: i32 = ${shape.code};`)
    .join("\n");
  return [
    "// Generated from SCENE_SHAPE_TABLE in lib/core/scene-shape.ts. Do not edit here.",
    constants,
    cupDistanceWgsl(),
    emitDispatch("rigidShapeInside(shape: i32, d: vec3f, p: vec3f) -> bool",
      (shape) => shape.wgslInside, "return false;"),
    emitDispatch("rigidShapeDistance(shape: i32, d: vec3f, p: vec3f) -> f32",
      (shape) => shape.wgslDistance, "return 1e20;"),
    emitDispatch("rigidShapeSupportRadius(shape: i32, d: vec3f, local: vec3f) -> f32",
      (shape) => shape.wgslSupportRadius, "return 0.0;"),
    emitDispatch("rigidShapeBoundingRadius(shape: i32, d: vec3f) -> f32",
      (shape) => shape.wgslBoundingRadius, "return 0.0;"),
    emitDispatch("rigidShapeVolume(shape: i32, d: vec3f) -> f32",
      (shape) => shape.wgslVolume, "return 0.0;"),
  ].join("\n");
}

/**
 * The cup's distance, as a standalone WGSL function.
 *
 * Exported on its own because the SVO render ABI needs exactly this and none of
 * the rest: a renderer has no use for a support map or a displaced volume, and
 * pasting the whole solver block into it would put five unused functions in
 * every shader that draws a scene. What matters is that the *body* is the same
 * text the solvers compile, so a cup cannot be drawn with one cavity and solved
 * with another.
 *
 * The dimensions are (outer radius, **full** height, wall thickness), matching
 * `SCENE_SHAPE_TABLE`. The SVO record convention is a half height, so its arm
 * doubles before calling — one visible conversion at one call site, rather than
 * a second spelling of the shape.
 */
export function cupDistanceWgsl(): string {
  const body = `  ${WGSL_CUP_PRELUDE}\n  return max(outer, -cavity);`;
  return [
    `fn ${CUP_WALL_FUNCTION}(d: vec3f) -> f32 { return clamp(d.z, 1e-4, min(0.95 * d.x, 0.95 * d.y)); }`,
    `fn ${CUP_DISTANCE_FUNCTION}(d: vec3f, p: vec3f) -> f32 {\n${body}\n}`,
  ].join("\n");
}

/**
 * The presentation facts, as WGSL.
 *
 * Separate from `sceneShapeWgsl` because only the shader that publishes render
 * records reads them, and a solver should not carry a palette it never looks
 * at. Same emission, same fallback discipline.
 */
export function sceneShapePresentationWgsl(): string {
  return [
    "// Generated from SCENE_SHAPE_TABLE in lib/core/scene-shape.ts. Do not edit here.",
    emitDispatch("rigidShapeRenderHalf(shape: i32, d: vec3f) -> vec3f",
      (shape) => shape.wgslRenderHalf, "return vec3f(0.0);"),
    emitDispatch("rigidShapePalette(shape: i32) -> vec3f",
      (shape) => `return vec3f(${shape.paletteLinear.join(", ")});`, "return vec3f(1.0);"),
  ].join("\n");
}
