export type SolverMode = "eulerian" | "particle" | "compare";
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
    gravity_m_s2: Vec3;
    initialCondition: "dam-break" | "tank-fill";
  };
  nominalResolution: {
    length_m: number;
  };
  numerics: {
    fixedDt_s: number;
    maxDt_s: number;
    pressureRelativeTolerance: number;
    pressureMaxIterations: number;
    particleSpacing_m: number;
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

export const BUILD_ID = "web-stage8-0.8.0";

export const defaultScene: SceneDescription = {
  schemaVersion: "1.0.0",
  sceneId: "interactive-water-box",
  randomSeed: 20260712,
  duration_s: 20,
  container: {
    width_m: 1.2,
    height_m: 0.9,
    depth_m: 0.8,
    fillFraction: 0.22,
    top: "open",
    fluidWallMode: "free-slip"
  },
  fluid: {
    density_kg_m3: 998.2,
    dynamicViscosity_Pa_s: 0.001002,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "dam-break"
  },
  nominalResolution: { length_m: 0.025 },
  numerics: {
    fixedDt_s: 0.004,
    maxDt_s: 0.008,
    pressureRelativeTolerance: 1e-8,
    pressureMaxIterations: 1000,
    particleSpacing_m: 0.025
  },
  rigidBodies: [
    {
      id: "body-sphere-1",
      name: "Cork sphere",
      shape: "sphere",
      dimensions_m: { x: 0.09, y: 0.09, z: 0.09 },
      density_kg_m3: 240,
      position_m: { x: -0.16, y: 1.18, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: { x: 0.18, y: 0, z: 0 },
      angularVelocity_rad_s: { x: 0, y: 0, z: 2.2 },
      restitution: 0.42,
      friction: 0.38
    },
    {
      id: "body-box-1",
      name: "Dense box",
      shape: "box",
      dimensions_m: { x: 0.17, y: 0.13, z: 0.14 },
      density_kg_m3: 1450,
      position_m: { x: 0.17, y: 1.34, z: 0.02 },
      orientation: { w: 0.965925826, x: 0, y: 0, z: 0.258819045 },
      linearVelocity_m_s: { x: -0.1, y: -0.1, z: 0 },
      angularVelocity_rad_s: { x: 1.2, y: 0.4, z: -0.7 },
      restitution: 0.24,
      friction: 0.55
    }
  ]
};

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
  parsed.fluid.initialCondition ??= "dam-break";
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
  if (!scene.fluid || !(scene.fluid.density_kg_m3 > 0)) errors.push("Fluid density must be positive");
  if (!scene.fluid || scene.fluid.dynamicViscosity_Pa_s < 0) errors.push("Dynamic viscosity cannot be negative");
  if (!scene.fluid || !["dam-break", "tank-fill"].includes(scene.fluid.initialCondition)) errors.push("Unsupported fluid initial condition");
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
    }
  }
  return errors;
}

export function createRunManifest(scene: SceneDescription, mode: SolverMode, adapter: string) {
  return {
    runSchemaVersion: "1.0.0",
    buildId: BUILD_ID,
    createdAt: new Date().toISOString(),
    solverMode: mode,
    precision: { cpu: "binary64", gpu: "f32" },
    browser: typeof navigator === "undefined" ? "node" : navigator.userAgent,
    webgpuAdapter: adapter,
    scene
  };
}
