import sharedDefaultScene from "./default-scene.json";
import { validateRefinementRegions } from "./octree-refinement-regions";
import { validateTerrain, type TerrainDescription } from "./terrain";
import type { EnvironmentId } from "./environments";
import { validateSceneryGraph, type SceneryGraph } from "./scenery-graph";
import type { DisplayGradeAuthoring, WaterOpticsAuthoring } from "./webgpu-lighting";

export type RunState = "paused" | "running";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

export type RigidShape = "sphere" | "box" | "capsule" | "cylinder";

export interface RigidBodyDescription {
  id: string;
  name: string;
  shape: RigidShape;
  dimensions_m: Vec3;
  density_kg_m3: number;
  position_m: Vec3;
  orientation: Quaternion;
  linearVelocity_m_s: Vec3;
  angularVelocity_rad_s: Vec3;
  restitution: number;
  friction: number;
  motion?: "dynamic" | "static";
}

export interface SceneDescription {
  schemaVersion: "1.0.0";
  sceneId: string;
  /** Optional subsystem declarations. Omission preserves all legacy systems. */
  systems?: {
    /** False builds the authored render world without fluid transport/authority. */
    fluid?: boolean;
  };
  /** Visible environment is part of the unified scene representation, not merely a backdrop. */
  environment?: EnvironmentId;
  /** How occupied SVO cells present their surface orientation. Omitted defaults to voxel-flat. */
  surfaceStyle?: "smooth" | "voxel-flat";
  /** Optional image-free lighting grade consumed by the SVO renderer. */
  lighting?: {
    /** Scene-linear directional key. Omitted fields retain the environment defaults. */
    directional?: {
      direction?: readonly [number, number, number];
      colorLinear?: readonly [number, number, number];
      intensity?: number;
    };
    /** Multipliers for the diffuse and prefiltered-specular environment terms. */
    environment?: {
      diffuseScale?: number;
      specularScale?: number;
      /** Optional scene-owned image-free backdrop/fill palette. */
      lowerRadianceLinear?: readonly [number, number, number];
      upperRadianceLinear?: readonly [number, number, number];
      accentRadianceLinear?: readonly [number, number, number];
    };
    /**
     * How the finished radiance becomes pixels: an exposure and a named tone
     * curve.
     *
     * Separate from `directional` on purpose. Everything above this line is a
     * fact about the set — where the sun is, how bright it is — and everything
     * in here is a fact about the camera that photographs it. Conflating them
     * is what makes a scene raise its key light to fight a transfer function,
     * which changes the shadows as a side effect of wanting a brighter wall.
     * Omitted is `DISPLAY_GRADE_NEUTRAL`; see `lib/webgpu-lighting.ts`.
     */
    grade?: DisplayGradeAuthoring;
  };
  randomSeed: number;
  duration_s: number;
  container: {
    width_m: number;
    height_m: number;
    depth_m: number;
    fillFraction: number;
    top: "open" | "closed";
    fluidWallMode: "free-slip" | "no-slip";
    /** Use the depth faces as 2D symmetry planes instead of physical walls. */
    depthBoundary?: "closed" | "symmetry";
    /**
     * Shape of the physical boundary. Omitted preserves the rectangular tank.
     * A spherical boundary is the largest sphere contained by the three
     * extents and is always closed; its exterior is solid.
     */
    shape?: "box" | "sphere";
    /**
     * Whether the domain is drawn as a glass vessel standing in the set.
     *
     * The container is two things at once: the solver's boundary, and — in
     * every environment except the garden — a tank you can see. A fresh scene
     * wants the first without the second, so that starting a scene hands over a
     * room rather than an aquarium nobody asked for; the tank is then something
     * to add. Absent means `glass`, so no authored document changes meaning.
     */
    vessel?: "glass" | "none";
  };
  /** Authoritative uniform lattice shared by scene geometry, SVO rendering, and fluid when enabled. */
  voxelDomain: {
    /** Requested finest physical spacing; integer container dimensions are rounded from this value. */
    finestCellSize_m: number;
    /** Payload edge length of each sparse octree terminal brick. */
    brickSize_cells: 4 | 8;
    /**
     * The finest voxel the *set* will be drawn at, when that is not
     * `finestCellSize_m`. Absent means the two are the same.
     *
     * These are two lattices, and conflating them is what makes authored detail
     * stop following the picture. `finestCellSize_m` is the octree's own — the
     * one the solver runs on and the one every world bound is measured in. But a
     * scene the solver does not own may spend extra octree levels below it
     * (`SvoRenderTuning.environmentRefinementDepth`, resolved in
     * `webgpu-octree-sparse-bricks.ts` as `renderCellSize = cellSize / 2^depth`),
     * so the voxel a floret or a bank is actually rasterised into can be four or
     * eight times smaller than the field above.
     *
     * A generator needs the *smaller* number, because every legibility floor it
     * carries is a count of voxels across a feature — see
     * `bonsaiCanopyLadder` — and a heightfield needs it because the voxeliser
     * takes one sample per finest column (`terrainColumnHeightsForLattice`).
     * Neither can read a render tuning, and neither should: this is the one
     * place the two lattices meet, and it travels with the document across the
     * structured-clone boundary into the render worker.
     *
     * Validated rather than defaulted, on the same argument as
     * `container.vessel` above: writing it into every document that round-trips
     * through `parseScene` would make the optional form change what an authored
     * scene says.
     */
    detailCellSize_m?: number;
    /**
     * Zero-rung cell retained while a dry environment is authored coarser.
     *
     * At negative refinement depths `finestCellSize_m` is the enlarged, actual
     * octree cell. This optional origin makes that signed rung reversible and
     * survives scene links; it is absent at zero and finer depths.
     */
    environmentRefinementBaseCellSize_m?: number;
    /** Optional minimum address-space bounds. Authored proxies may extend the sparse domain beyond them. */
    bounds_m?: {
      min: Vec3;
      max: Vec3;
    };
  };
  /** Optional ground heightfield inside the container; absent means a flat floor at y = 0. */
  terrain?: TerrainDescription;
  /**
   * Everything visible that is not water, terrain or a rigid body, as a
   * declarative graph the document owns outright. An environment preset seeds
   * it when the scene is created and is never consulted again; editing scenery
   * is an ordinary edit to these nodes. See lib/scenery-graph.ts.
   */
  scenery?: SceneryGraph;
  fluid: {
    density_kg_m3: number;
    dynamicViscosity_Pa_s: number;
    surfaceTension_N_m: number;
    gravity_m_s2: Vec3;
    initialCondition: "dam-break" | "tank-fill";
    /** Optional absolute size of the dam reservoir. */
    initialDamBreakDimensions_m?: Vec3;
    /**
     * Offset of the reservoir's minimum corner from the container's minimum
     * corner, in metres. Omitted (the legacy shape) anchors the reservoir in
     * the (-x, 0, -z) corner, which is what the GPU's closed-form t=0 phi
     * assumes; an authored offset therefore takes the host-rasterized seed
     * path instead of the analytic bootstrap. Only the box moves — this is what
     * lets the water body be dragged off the corner and reshaped.
     */
    initialDamBreakOrigin_m?: Vec3;
    /**
     * Optional world-space seeds for exact solver bricks. Each seed fills the
     * one brick containing it; multiple seeds create disconnected initial
     * bodies without allocating the space between them.
     */
    initialBrickSeeds_m?: Vec3[];
    /**
     * When true, seeded bricks are added on top of the ordinary initial
     * condition (tank fill or dam break) instead of replacing it. The ocean
     * scene uses this to raise a slab of water above a settled pool.
     */
    initialBrickSeedsAdditive?: boolean;
    /**
     * Analytic liquid volumes present at t = 0.
     *
     * Volumes are unioned with the ordinary tank-fill/dam-break condition and
     * with painted bricks. Boxes, spheres and hemispheres share one vocabulary
     * so scene authors do not need a solver-specific seeding field per shape.
     */
    initialLiquidVolumes?: InitialLiquidVolume[];
    inflow?: FluidInflow;
    /**
     * What the liquid looks like, as opposed to how it moves.
     *
     * Absorption and scatter are rates per metre, so the same coefficients
     * describe entirely different-looking water at tank depth and at pond
     * depth. Omitted, the renderer uses the frozen clean-water table in
     * `lib/webgpu-lighting.ts` and nothing about an existing document changes;
     * present, it is threaded to the composite as a uniform rather than
     * inlined into WGSL, which is what makes it authorable at all.
     */
    optics?: WaterOpticsAuthoring;
    /**
     * Authored boxes that bound the pressure-octree cell sizes inside them.
     * See `lib/octree-refinement-regions.ts`.
     *
     * Uniform-tier: the regions reach the GPU through the projection's params
     * buffer, so drawing or retuning one is a buffer write on the running
     * solver rather than a re-seed. That is what makes them an experiment
     * surface — `gpuSceneUniformKey` carries them and `gpuSceneSeedKey`
     * deliberately does not.
     */
    refinementRegions?: FluidRefinementRegion[];
  };
  nominalResolution: {
    length_m: number;
  };
  numerics: {
    fixedDt_s: number;
    maxDt_s: number;
    pressureRelativeTolerance: number;
    pressureMaxIterations: number;
  };
  rigidBodies: RigidBodyDescription[];
}

/**
 * A drawn box that bounds how finely or coarsely the pressure octree resolves
 * its fully contained leaves.
 *
 * Deliberately declarative about *meaning* rather than about machinery: `rule`
 * names what the box is for, so the next thing worth declaring over a region —
 * a viscosity, a solve budget, a visualization mask — arrives as another rule
 * rather than as another parallel list of boxes. Today there is exactly one.
 *
 * See `lib/octree-refinement-regions.ts` for the containment contract and for
 * why this is a uniform-tier input.
 */
export interface FluidRefinementRegion {
  id: string;
  min_m: Vec3;
  max_m: Vec3;
  rule: "minimum-cell-size";
  /** Smallest pressure-cell edge allowed inside, in finest cells. Power of two. */
  minimumCellSize_cells: number;
  /** Optional largest pressure-cell edge allowed inside, in finest cells.
   * Power of two. Omitted leaves coarsening entirely evidence-driven. */
  maximumCellSize_cells?: number;
}

/** A world-space analytic liquid volume present at t = 0. */
export type InitialLiquidVolume = InitialLiquidBox | InitialLiquidSphere | InitialLiquidHemisphere | InitialLiquidCylinder;

export interface InitialLiquidBox {
  shape: "box";
  min_m: Vec3;
  max_m: Vec3;
}

export interface InitialLiquidSphere {
  shape: "sphere";
  center_m: Vec3;
  radius_m: number;
}

export interface InitialLiquidHemisphere {
  shape: "hemisphere";
  center_m: Vec3;
  radius_m: number;
  /**
   * Outward normal of the flat face. The retained liquid satisfies
   * `dot(point - center, outwardNormal) <= 0`.
   */
  outwardNormal: Vec3;
}

/** A finite cylinder aligned with the world z axis. */
export interface InitialLiquidCylinder {
  shape: "cylinder";
  center_m: Vec3;
  radius_m: number;
  halfHeight_m: number;
}

export interface FluidInflow {
  center_m: Vec3;
  radius_m: number;
  length_m: number;
  velocity_m_s: Vec3;
  start_s: number;
  end_s: number;
  ramp_s: number;
}

export interface CameraState {
  azimuth_rad: number;
  elevation_rad: number;
  distance_m: number;
  target_m: Vec3;
  /**
   * The lens, as `tan(verticalFieldOfView / 2)` rather than an angle.
   *
   * A tangent and not an angle because that is the form every consumer needs:
   * the shaders multiply it by an NDC coordinate, the host inverses divide by
   * it, and a stored radian would be converted at each of ~20 sites with a
   * chance of one of them drifting. `lib/webgpu-camera.ts` holds the same
   * reasoning for the shared constant.
   *
   * Omitted means `CAMERA_TAN_HALF_FOV` (0.72, a 16.7 mm equivalent), which is
   * what this renderer has always drawn and what every golden frame in the repo
   * was captured at. Resolve it with `cameraTanHalfFov` rather than reading the
   * field, so an out-of-range value clamps instead of taking the frame down.
   */
  tanHalfFov?: number;
}

export interface MetricSample {
  t: number;
  frame_ms: number;
  volume_drift_pct: number;
  constraint_error: number;
  kinetic_energy_J: number;
}

export const BUILD_ID = "web-tall-cell-ab-1.3.0";
export const DEFAULT_GPU_CPU_TIMESTEP_RATIO = 4;

/**
 * The lattice, for everything. One number, not one per scene.
 *
 * Scenes used to name their own spacing — 50 mm here, 25 mm there, a droplet
 * constant somewhere else — so making the product finer meant finding and
 * agreeing every one of them, and a scene that was missed simply stayed coarse
 * without saying so. This is the single knob: change it here and every scene
 * that has no *physical* reason to differ follows.
 *
 * 6.25 mm because that is where the hero garden's set becomes legible — at
 * 25 mm the stepping discs merge into one mass, the coping's bullnose reads as
 * a tube and the bonsai's crown is a disc; at 6.25 mm each resolves.
 *
 * It is the *render* lattice. A solved scene may not be able to carry it — the
 * hero garden's fluid path overruns a one-workgroup-per-interface-leaf dispatch
 * below 7.5 mm — so a factory building for the solver clamps it upward and says
 * so. That clamp belongs to the system with the constraint, not to the scene.
 */
export const DEFAULT_FINEST_CELL_SIZE_M = 0.00625;

export const defaultScene: SceneDescription = sharedDefaultScene as SceneDescription;

export const defaultCamera: CameraState = {
  azimuth_rad: 0.72,
  elevation_rad: 0.42,
  distance_m: 1.9,
  target_m: { x: 0, y: 0.38, z: 0 }
};

export function cloneScene(scene: SceneDescription): SceneDescription {
  return JSON.parse(JSON.stringify(scene)) as SceneDescription;
}

/**
 * The renderer-wide SVO surface policy.
 *
 * Voxel faces are the default for every scene, including documents authored
 * before `surfaceStyle` existed. `smooth` remains an explicit opt-out; no scene
 * needs to opt in merely to receive the product default.
 */
export function sceneUsesFlatVoxelNormals(
  scene: Pick<SceneDescription, "surfaceStyle">,
): boolean {
  return scene.surfaceStyle !== "smooth";
}

export function canonicalScene(scene: SceneDescription): string {
  const stable = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => [key, stable(child)]));
    }
    return value;
  };
  return JSON.stringify(stable(scene));
}

// Scene documents are immutable at the application boundary: every editor and
// store mutation replaces the document. Keep the hot render loop keyed by that
// mutation surface instead of recursively sorting and cloning the same document
// every frame. The WeakMap deliberately stays out of serialization; callers
// that do not enter through `markSceneRevision` retain the canonical-string
// fallback in the renderer.
const sceneRevisionByDocument = new WeakMap<SceneDescription, number>();
let nextSceneDocumentRevision = 1;

/** Stamp a newly published immutable scene document and return it unchanged. */
export function markSceneRevision(scene: SceneDescription): SceneDescription {
  if (!sceneRevisionByDocument.has(scene)) {
    sceneRevisionByDocument.set(scene, nextSceneDocumentRevision);
    nextSceneDocumentRevision = nextSceneDocumentRevision >= Number.MAX_SAFE_INTEGER
      ? 1 : nextSceneDocumentRevision + 1;
  }
  return scene;
}

/** Monotonic application revision, or undefined for an external/unversioned document. */
export function sceneRevision(scene: SceneDescription): number | undefined {
  return sceneRevisionByDocument.get(scene);
}

export function serializeScene(scene: SceneDescription): string {
  return JSON.stringify(scene, null, 2) + "\n";
}

export function parseScene(input: string): SceneDescription {
  const parsed = JSON.parse(input) as SceneDescription;
  parsed.rigidBodies ??= [];
  parsed.container.fluidWallMode ??= "no-slip";
  parsed.fluid.initialCondition ??= "dam-break";
  parsed.fluid.surfaceTension_N_m ??= 0.072;
  const errors = validateScene(parsed);
  if (errors.length > 0) throw new Error(errors.join("; "));
  return parsed;
}

export function validateScene(scene: SceneDescription): string[] {
  const errors: string[] = [];
  if (scene.surfaceStyle !== undefined && scene.surfaceStyle !== "smooth" && scene.surfaceStyle !== "voxel-flat") {
    errors.push(`Unknown scene surface style ${String(scene.surfaceStyle)}`);
  }
  if (scene.schemaVersion !== "1.0.0") errors.push("Unsupported schema version");
  if (!scene.sceneId?.trim()) errors.push("Scene ID is required");
  if (scene.systems?.fluid !== undefined && typeof scene.systems.fluid !== "boolean") errors.push("Scene fluid-system flag must be boolean");
  const lighting = scene.lighting;
  if (lighting?.directional?.direction) {
    const direction = lighting.directional.direction;
    if (direction.length !== 3 || !direction.every(Number.isFinite) || Math.hypot(...direction) <= 1e-12) errors.push("Scene directional-light direction must be finite and non-zero");
  }
  if (lighting?.directional?.colorLinear) {
    const color = lighting.directional.colorLinear;
    if (color.length !== 3 || !color.every((value) => Number.isFinite(value) && value >= 0)) errors.push("Scene directional-light color must contain three non-negative finite channels");
  }
  if (lighting?.directional?.intensity !== undefined && (!Number.isFinite(lighting.directional.intensity) || lighting.directional.intensity < 0)) errors.push("Scene directional-light intensity must be non-negative and finite");
  if (lighting?.grade?.exposure !== undefined && (!Number.isFinite(lighting.grade.exposure) || lighting.grade.exposure <= 0)) errors.push("Scene grade exposure must be positive and finite");
  if (lighting?.grade?.toneCurve !== undefined && lighting.grade.toneCurve !== "reinhard" && lighting.grade.toneCurve !== "aces") errors.push("Scene grade tone curve must be reinhard or aces");
  if (lighting?.grade?.whiteBalance !== undefined) {
    const balance = lighting.grade.whiteBalance;
    if (balance.length !== 3 || !balance.every((value) => Number.isFinite(value) && value > 0)) errors.push("Scene grade white balance must contain three positive finite channel gains");
  }
  for (const [value, label] of [[lighting?.environment?.diffuseScale, "diffuse"], [lighting?.environment?.specularScale, "specular"]] as const) {
    if (value !== undefined && (!Number.isFinite(value) || value < 0)) errors.push(`Scene environment ${label} scale must be non-negative and finite`);
  }
  for (const [color, label] of [
    [lighting?.environment?.lowerRadianceLinear, "lower"],
    [lighting?.environment?.upperRadianceLinear, "upper"],
    [lighting?.environment?.accentRadianceLinear, "accent"],
  ] as const) {
    if (color !== undefined && (color.length !== 3
      || !color.every((value) => Number.isFinite(value) && value >= 0))) {
      errors.push(`Scene environment ${label} radiance must contain three non-negative finite channels`);
    }
  }
  if (!Number.isInteger(scene.randomSeed) || scene.randomSeed < 0) errors.push("Random seed must be a non-negative integer");
  if (!(scene.duration_s > 0)) errors.push("Duration must be positive");
  const c = scene.container;
  if (!c || !(c.width_m > 0) || !(c.height_m > 0) || !(c.depth_m > 0)) errors.push("Container dimensions must be positive");
  // Validated rather than defaulted: a default would write the field into every
  // document that round-trips through `parseScene`, and the whole point of the
  // optional form is that an authored scene is unchanged by its existence.
  if (c?.vessel !== undefined && c.vessel !== "glass" && c.vessel !== "none") errors.push("Container vessel must be 'glass' or 'none'");
  if (c?.shape !== undefined && c.shape !== "box" && c.shape !== "sphere") errors.push("Container shape must be 'box' or 'sphere'");
  if (c?.shape === "sphere" && c.top !== "closed") errors.push("A spherical container must be closed");
  if (!c || c.fillFraction < 0 || c.fillFraction > 1) errors.push("Fill fraction must be in [0, 1]");
  if (!c || !["free-slip", "no-slip"].includes(c.fluidWallMode)) errors.push("Unsupported fluid wall mode");
  if (c?.depthBoundary !== undefined && c.depthBoundary !== "closed" && c.depthBoundary !== "symmetry") {
    errors.push("Unsupported depth boundary");
  }
  const voxelDomain = scene.voxelDomain;
  if (!voxelDomain || !Number.isFinite(voxelDomain.finestCellSize_m) || !(voxelDomain.finestCellSize_m > 0)) errors.push("Voxel finest cell size must be positive and finite");
  if (!voxelDomain || (voxelDomain.brickSize_cells !== 4 && voxelDomain.brickSize_cells !== 8)) errors.push("Voxel brick size must be 4 or 8 cells");
  if (scene.systems?.fluid !== false && voxelDomain?.brickSize_cells === 4) errors.push("Fluid-enabled scenes require 8-cell voxel bricks");
  // Never coarser than the lattice it refines: the field names the voxel a set
  // is drawn *into*, and an octree spends levels downward from
  // `finestCellSize_m` or not at all. A larger value would be a request the tree
  // cannot service and would silently over-report how legible a feature is.
  if (voxelDomain?.detailCellSize_m !== undefined
    && (!(voxelDomain.detailCellSize_m > 0) || !Number.isFinite(voxelDomain.detailCellSize_m)
      || voxelDomain.detailCellSize_m > voxelDomain.finestCellSize_m + 1e-12)) {
    errors.push("Voxel detail cell size must be positive, finite and no coarser than the finest cell size");
  }
  if (voxelDomain?.environmentRefinementBaseCellSize_m !== undefined
    && (!(voxelDomain.environmentRefinementBaseCellSize_m > 0)
      || !Number.isFinite(voxelDomain.environmentRefinementBaseCellSize_m)
      || voxelDomain.environmentRefinementBaseCellSize_m > voxelDomain.finestCellSize_m + 1e-12)) {
    errors.push("Voxel environment refinement base cell must be positive, finite and no coarser than the active lattice");
  }
  if (voxelDomain?.bounds_m) {
    const { min, max } = voxelDomain.bounds_m;
    if (![min?.x, min?.y, min?.z, max?.x, max?.y, max?.z].every(Number.isFinite)) errors.push("Voxel domain bounds must be finite");
    else if (!(min.x < max.x && min.y < max.y && min.z < max.z)) errors.push("Voxel domain bounds must have positive extent");
  }
  if (!scene.fluid || !(scene.fluid.density_kg_m3 > 0)) errors.push("Fluid density must be positive");
  if (!scene.fluid || scene.fluid.dynamicViscosity_Pa_s < 0) errors.push("Dynamic viscosity cannot be negative");
  if (!scene.fluid || scene.fluid.surfaceTension_N_m < 0) errors.push("Surface tension cannot be negative");
  if (!scene.fluid || !["dam-break", "tank-fill"].includes(scene.fluid.initialCondition)) errors.push("Unsupported fluid initial condition");
  const optics = scene.fluid?.optics;
  if (optics) {
    for (const [value, label] of [[optics.absorption_mInv, "absorption"], [optics.scatter, "scatter"]] as const) {
      if (value !== undefined && (value.length !== 3 || !value.every((channel) => Number.isFinite(channel) && channel >= 0))) {
        errors.push(`Water ${label} must contain three non-negative finite channels`);
      }
    }
    // A refractive index under one inverts the interface and there is nothing
    // to refract at exactly one; the composite would then shade air as water.
    if (optics.indexOfRefraction !== undefined && !(optics.indexOfRefraction > 1 && optics.indexOfRefraction <= 2)) {
      errors.push("Water index of refraction must lie in (1, 2]");
    }
    for (const [value, label] of [[optics.fresnelF0, "Fresnel F0"], [optics.causticStrength, "caustic strength"]] as const) {
      if (value !== undefined && (!Number.isFinite(value) || value < 0 || value > 1)) errors.push(`Water ${label} must lie in [0, 1]`);
    }
  }
  const damDimensions = scene.fluid?.initialDamBreakDimensions_m;
  const damOrigin = scene.fluid?.initialDamBreakOrigin_m;
  if (damOrigin && !damDimensions) errors.push("Initial dam-break origin requires authored dam-break dimensions");
  if (damDimensions) {
    if (scene.fluid.initialCondition !== "dam-break") errors.push("Initial dam-break dimensions require the dam-break initial condition");
    if (![damDimensions.x, damDimensions.y, damDimensions.z].every((value) => Number.isFinite(value) && value > 0)) {
      errors.push("Initial dam-break dimensions must be positive and finite");
    } else if (damOrigin && ![damOrigin.x, damOrigin.y, damOrigin.z].every(Number.isFinite)) {
      errors.push("Initial dam-break origin must be finite");
    } else if (damOrigin
      ? damOrigin.x < -1e-9 || damOrigin.y < -1e-9 || damOrigin.z < -1e-9
        || damOrigin.x + damDimensions.x > c.width_m + 1e-9
        || damOrigin.y + damDimensions.y > c.height_m + 1e-9
        || damOrigin.z + damDimensions.z > c.depth_m + 1e-9
      : damDimensions.x > c.width_m || damDimensions.y > c.height_m || damDimensions.z > c.depth_m) {
      errors.push("Initial dam-break dimensions must fit inside the container");
    } else {
      const damFillFraction = damDimensions.x * damDimensions.y * damDimensions.z
        / (c.width_m * c.height_m * c.depth_m);
      if (Math.abs(damFillFraction - c.fillFraction) > 1e-9) {
        errors.push("Fill fraction must equal the authored dam-break volume fraction");
      }
    }
  }
  if (scene.fluid?.initialBrickSeeds_m) {
    if (!Array.isArray(scene.fluid.initialBrickSeeds_m) || scene.fluid.initialBrickSeeds_m.length === 0) errors.push("Initial fluid brick seeds must be a non-empty array");
    else for (const [index, seed] of scene.fluid.initialBrickSeeds_m.entries()) {
      if (![seed?.x, seed?.y, seed?.z].every(Number.isFinite)) errors.push(`Initial fluid brick seed ${index} must be finite`);
      else if (seed.x < -c.width_m / 2 || seed.x >= c.width_m / 2 || seed.y < 0 || seed.y >= c.height_m || seed.z < -c.depth_m / 2 || seed.z >= c.depth_m / 2) {
        errors.push(`Initial fluid brick seed ${index} must be inside the solver bounds`);
      }
    }
  }
  if (scene.fluid?.initialLiquidVolumes) {
    const volumes = scene.fluid.initialLiquidVolumes;
    if (!Array.isArray(volumes) || volumes.length === 0) errors.push("Initial liquid volumes must be a non-empty array");
    else for (const [index, volume] of volumes.entries()) {
      if (!volume || !["box", "sphere", "hemisphere", "cylinder"].includes(volume.shape)) {
        errors.push(`Initial liquid volume ${index} has an unsupported shape`);
        continue;
      }
      if (volume.shape === "box") {
        const min = volume.min_m;
        const max = volume.max_m;
        if (![min?.x, min?.y, min?.z, max?.x, max?.y, max?.z].every(Number.isFinite)) {
          errors.push(`Initial liquid box ${index} bounds must be finite`);
        } else if (!(min.x < max.x && min.y < max.y && min.z < max.z)) {
          errors.push(`Initial liquid box ${index} must have positive extent`);
        } else if (max.x <= -c.width_m / 2 || min.x >= c.width_m / 2 || max.y <= 0 || min.y >= c.height_m
          || max.z <= -c.depth_m / 2 || min.z >= c.depth_m / 2) {
          errors.push(`Initial liquid box ${index} must intersect the container`);
        }
        continue;
      }
      const centre = volume.center_m;
      if (![centre?.x, centre?.y, centre?.z].every(Number.isFinite)) errors.push(`Initial liquid ${volume.shape} ${index} centre must be finite`);
      else if (!(volume.radius_m > 0) || !Number.isFinite(volume.radius_m)) errors.push(`Initial liquid ${volume.shape} ${index} radius must be positive and finite`);
      else if (volume.shape === "cylinder" && (!(volume.halfHeight_m > 0) || !Number.isFinite(volume.halfHeight_m))) {
        errors.push(`Initial liquid cylinder ${index} half-height must be positive and finite`);
      }
      else if (centre.x < -c.width_m / 2 || centre.x > c.width_m / 2 || centre.y < 0 || centre.y > c.height_m
        || centre.z < -c.depth_m / 2 || centre.z > c.depth_m / 2) {
        errors.push(`Initial liquid ${volume.shape} ${index} centre must be inside the container`);
      }
      if (volume.shape === "hemisphere") {
        const normal = volume.outwardNormal;
        if (![normal?.x, normal?.y, normal?.z].every(Number.isFinite)
          || !(Math.hypot(normal.x, normal.y, normal.z) > 1e-12)) {
          errors.push(`Initial liquid hemisphere ${index} outward normal must be finite and non-zero`);
        }
      }
    }
  }
  const inflow = scene.fluid?.inflow;
  if (inflow) {
    const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
    if (!(inflow.radius_m > 0) || !(inflow.length_m > 0)) errors.push("Inflow radius and length must be positive");
    if (!(speed > 0)) errors.push("Inflow velocity must be non-zero");
    if (!(inflow.start_s >= 0) || !(inflow.end_s > inflow.start_s) || !(inflow.ramp_s >= 0)) errors.push("Inflow timing is invalid");
    if (inflow.center_m.x < -c.width_m / 2 || inflow.center_m.x > c.width_m / 2
      || inflow.center_m.y < 0 || inflow.center_m.y > c.height_m
      || inflow.center_m.z < -c.depth_m / 2 || inflow.center_m.z > c.depth_m / 2) errors.push("Inflow center must be inside the container");
  }
  if (c) errors.push(...validateRefinementRegions(scene.fluid?.refinementRegions, c));
  if (scene.terrain && c) errors.push(...validateTerrain(scene.terrain, c));
  if (scene.scenery) errors.push(...validateSceneryGraph(scene.scenery));
  if (!scene.nominalResolution || !(scene.nominalResolution.length_m > 0)) errors.push("Nominal resolution must be positive");
  if (!scene.numerics || !(scene.numerics.fixedDt_s > 0) || !(scene.numerics.maxDt_s > 0)) errors.push("Time steps must be positive");
  if (scene.numerics && scene.numerics.fixedDt_s > scene.numerics.maxDt_s) errors.push("Fixed time step exceeds maximum time step");
  if (!Array.isArray(scene.rigidBodies)) errors.push("Rigid bodies must be an array");
  else {
    const ids = new Set<string>();
    for (const body of scene.rigidBodies) {
      if (!body.id || ids.has(body.id)) errors.push("Rigid body IDs must be unique and non-empty");
      ids.add(body.id);
      if (!(["sphere", "box", "capsule", "cylinder"] as string[]).includes(body.shape)) errors.push(`Unsupported rigid shape ${body.shape}`);
      if (!(body.dimensions_m.x > 0) || !(body.dimensions_m.y > 0) || !(body.dimensions_m.z > 0)) errors.push(`Body ${body.id} dimensions must be positive`);
      if (!(body.density_kg_m3 > 0)) errors.push(`Body ${body.id} density must be positive`);
      if (body.restitution < 0 || body.restitution > 1) errors.push(`Body ${body.id} restitution must be in [0, 1]`);
      if (body.friction < 0) errors.push(`Body ${body.id} friction cannot be negative`);
      if (body.motion && !["dynamic", "static"].includes(body.motion)) errors.push(`Unsupported motion type for body ${body.id}`);
    }
  }
  return errors;
}

export function createRunManifest(scene: SceneDescription, adapter: string) {
  return {
    runSchemaVersion: "1.0.0",
    buildId: BUILD_ID,
    createdAt: new Date().toISOString(),
    solverMode: "eulerian",
    precision: { cpu: "binary64", gpu: "f32" },
    browser: typeof navigator === "undefined" ? "node" : navigator.userAgent,
    webgpuAdapter: adapter,
    scene
  };
}
