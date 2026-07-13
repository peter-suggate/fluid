import sharedDefaultScene from "../native/Sources/FluidMetal/Resources/default-scene.json";

export type ViewMode = "scientific" | "presentation";
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
}

export interface HierarchySettings {
  /** Number of physical AMR levels. One is the uniform-grid limit. */
  levels: number;
  /** Cubic cells per leaf brick edge. WebGPU currently uses four. */
  brickSize: number;
  /** Hard allocation ceiling; refinement reports saturation instead of overflowing. */
  maxActiveBricks: number;
  /** Finest-level cells retained on either side of a reconstructed free surface. */
  interfaceHaloCells: number;
  /** Finest-level cells retained around current and predicted rigid-body bounds. */
  solidHaloCells: number;
  /** Number of simulation steps between topology updates. */
  regridInterval: number;
  /** Consecutive inactive topology updates required before a brick can coarsen. */
  coarsenDelay: number;
  /** Dimensionless detail thresholds. Zero forces finest resolution everywhere wet. */
  velocityErrorTolerance: number;
  vorticityTolerance: number;
  /** Lowest permitted leaf level in occupied fluid, counted from the coarsest level. */
  minimumFluidLevel: number;
}

export interface SceneDescription {
  schemaVersion: "1.0.0";
  sceneId: string;
  randomSeed: number;
  duration_s: number;
  container: {
    width_m: number;
    height_m: number;
    depth_m: number;
    fillFraction: number;
    top: "open" | "closed";
    fluidWallMode: "free-slip" | "no-slip";
  };
  fluid: {
    density_kg_m3: number;
    dynamicViscosity_Pa_s: number;
    surfaceTension_N_m: number;
    gravity_m_s2: Vec3;
    initialCondition: "dam-break" | "tank-fill";
  };
  nominalResolution: {
    length_m: number;
  };
  hierarchy: HierarchySettings;
  numerics: {
    fixedDt_s: number;
    maxDt_s: number;
    pressureRelativeTolerance: number;
    pressureMaxIterations: number;
  };
  rigidBodies: RigidBodyDescription[];
}

export interface CameraState {
  azimuth_rad: number;
  elevation_rad: number;
  distance_m: number;
  target_m: Vec3;
}

export interface MetricSample {
  t: number;
  frame_ms: number;
  volume_drift_pct: number;
  constraint_error: number;
  kinetic_energy_J: number;
}

export const BUILD_ID = "web-stage12.0-1.2.0";

export const defaultScene: SceneDescription = sharedDefaultScene as SceneDescription;

export const defaultCamera: CameraState = {
  azimuth_rad: 0.72,
  elevation_rad: 0.42,
  distance_m: 2.65,
  target_m: { x: 0, y: 0.38, z: 0 }
};

export function cloneScene(scene: SceneDescription): SceneDescription {
  return JSON.parse(JSON.stringify(scene)) as SceneDescription;
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

export function serializeScene(scene: SceneDescription): string {
  return JSON.stringify(scene, null, 2) + "\n";
}

export function parseScene(input: string): SceneDescription {
  const parsed = JSON.parse(input) as SceneDescription;
  parsed.rigidBodies ??= [];
  parsed.container.fluidWallMode ??= "no-slip";
  parsed.fluid.initialCondition ??= "dam-break";
  parsed.fluid.surfaceTension_N_m ??= 0.072;
  parsed.hierarchy ??= {
    levels: 1,
    brickSize: 4,
    maxActiveBricks: 32768,
    interfaceHaloCells: 3,
    solidHaloCells: 3,
    regridInterval: 12,
    coarsenDelay: 3,
    velocityErrorTolerance: 0.08,
    vorticityTolerance: 0.12,
    minimumFluidLevel: 0
  };
  const errors = validateScene(parsed);
  if (errors.length > 0) throw new Error(errors.join("; "));
  return parsed;
}

export function validateScene(scene: SceneDescription): string[] {
  const errors: string[] = [];
  if (scene.schemaVersion !== "1.0.0") errors.push("Unsupported schema version");
  if (!scene.sceneId?.trim()) errors.push("Scene ID is required");
  if (!Number.isInteger(scene.randomSeed) || scene.randomSeed < 0) errors.push("Random seed must be a non-negative integer");
  if (!(scene.duration_s > 0)) errors.push("Duration must be positive");
  const c = scene.container;
  if (!c || !(c.width_m > 0) || !(c.height_m > 0) || !(c.depth_m > 0)) errors.push("Container dimensions must be positive");
  if (!c || c.fillFraction < 0 || c.fillFraction > 1) errors.push("Fill fraction must be in [0, 1]");
  if (!c || !["free-slip", "no-slip"].includes(c.fluidWallMode)) errors.push("Unsupported fluid wall mode");
  if (!scene.fluid || !(scene.fluid.density_kg_m3 > 0)) errors.push("Fluid density must be positive");
  if (!scene.fluid || scene.fluid.dynamicViscosity_Pa_s < 0) errors.push("Dynamic viscosity cannot be negative");
  if (!scene.fluid || scene.fluid.surfaceTension_N_m < 0) errors.push("Surface tension cannot be negative");
  if (!scene.fluid || !["dam-break", "tank-fill"].includes(scene.fluid.initialCondition)) errors.push("Unsupported fluid initial condition");
  if (!scene.nominalResolution || !(scene.nominalResolution.length_m > 0)) errors.push("Nominal resolution must be positive");
  const hierarchy = scene.hierarchy;
  if (!hierarchy) errors.push("Hierarchy settings are required");
  else {
    if (!Number.isInteger(hierarchy.levels) || hierarchy.levels < 1 || hierarchy.levels > 6) errors.push("Hierarchy levels must be an integer in [1, 6]");
    if (!Number.isInteger(hierarchy.brickSize) || ![2, 4, 8].includes(hierarchy.brickSize)) errors.push("Hierarchy brick size must be 2, 4, or 8");
    if (!Number.isInteger(hierarchy.maxActiveBricks) || hierarchy.maxActiveBricks < 1) errors.push("Hierarchy brick budget must be a positive integer");
    if (!Number.isInteger(hierarchy.interfaceHaloCells) || hierarchy.interfaceHaloCells < 0) errors.push("Interface halo must be a non-negative integer");
    if (!Number.isInteger(hierarchy.solidHaloCells) || hierarchy.solidHaloCells < 0) errors.push("Solid halo must be a non-negative integer");
    if (!Number.isInteger(hierarchy.regridInterval) || hierarchy.regridInterval < 1) errors.push("Regrid interval must be a positive integer");
    if (!Number.isInteger(hierarchy.coarsenDelay) || hierarchy.coarsenDelay < 1) errors.push("Coarsen delay must be a positive integer");
    if (!(hierarchy.velocityErrorTolerance >= 0) || !(hierarchy.vorticityTolerance >= 0)) errors.push("Hierarchy error tolerances cannot be negative");
    if (!Number.isInteger(hierarchy.minimumFluidLevel) || hierarchy.minimumFluidLevel < 0 || hierarchy.minimumFluidLevel >= hierarchy.levels) errors.push("Minimum fluid level must lie within the hierarchy");
  }
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
