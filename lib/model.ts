import sharedDefaultScene from "../native/Sources/FluidMetal/Resources/default-scene.json";
import { validateTerrain, type TerrainDescription } from "./terrain";
import type { EnvironmentId } from "./environments";

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
    };
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
  };
  /** Authoritative uniform lattice shared by scene geometry, SVO rendering, and fluid when enabled. */
  voxelDomain: {
    /** Requested finest physical spacing; integer container dimensions are rounded from this value. */
    finestCellSize_m: number;
    /** Payload edge length of each sparse octree terminal brick. */
    brickSize_cells: 4 | 8;
    /** Optional minimum address-space bounds. Authored proxies may extend the sparse domain beyond them. */
    bounds_m?: {
      min: Vec3;
      max: Vec3;
    };
  };
  /** Optional ground heightfield inside the container; absent means a flat floor at y = 0. */
  terrain?: TerrainDescription;
  fluid: {
    density_kg_m3: number;
    dynamicViscosity_Pa_s: number;
    surfaceTension_N_m: number;
    gravity_m_s2: Vec3;
    initialCondition: "dam-break" | "tank-fill";
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
    inflow?: FluidInflow;
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
  const errors = validateScene(parsed);
  if (errors.length > 0) throw new Error(errors.join("; "));
  return parsed;
}

export function validateScene(scene: SceneDescription): string[] {
  const errors: string[] = [];
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
  for (const [value, label] of [[lighting?.environment?.diffuseScale, "diffuse"], [lighting?.environment?.specularScale, "specular"]] as const) {
    if (value !== undefined && (!Number.isFinite(value) || value < 0)) errors.push(`Scene environment ${label} scale must be non-negative and finite`);
  }
  if (!Number.isInteger(scene.randomSeed) || scene.randomSeed < 0) errors.push("Random seed must be a non-negative integer");
  if (!(scene.duration_s > 0)) errors.push("Duration must be positive");
  const c = scene.container;
  if (!c || !(c.width_m > 0) || !(c.height_m > 0) || !(c.depth_m > 0)) errors.push("Container dimensions must be positive");
  if (!c || c.fillFraction < 0 || c.fillFraction > 1) errors.push("Fill fraction must be in [0, 1]");
  if (!c || !["free-slip", "no-slip"].includes(c.fluidWallMode)) errors.push("Unsupported fluid wall mode");
  const voxelDomain = scene.voxelDomain;
  if (!voxelDomain || !Number.isFinite(voxelDomain.finestCellSize_m) || !(voxelDomain.finestCellSize_m > 0)) errors.push("Voxel finest cell size must be positive and finite");
  if (!voxelDomain || (voxelDomain.brickSize_cells !== 4 && voxelDomain.brickSize_cells !== 8)) errors.push("Voxel brick size must be 4 or 8 cells");
  if (scene.systems?.fluid !== false && voxelDomain?.brickSize_cells === 4) errors.push("Fluid-enabled scenes require 8-cell voxel bricks");
  if (voxelDomain?.bounds_m) {
    const { min, max } = voxelDomain.bounds_m;
    if (![min?.x, min?.y, min?.z, max?.x, max?.y, max?.z].every(Number.isFinite)) errors.push("Voxel domain bounds must be finite");
    else if (!(min.x < max.x && min.y < max.y && min.z < max.z)) errors.push("Voxel domain bounds must have positive extent");
  }
  if (!scene.fluid || !(scene.fluid.density_kg_m3 > 0)) errors.push("Fluid density must be positive");
  if (!scene.fluid || scene.fluid.dynamicViscosity_Pa_s < 0) errors.push("Dynamic viscosity cannot be negative");
  if (!scene.fluid || scene.fluid.surfaceTension_N_m < 0) errors.push("Surface tension cannot be negative");
  if (!scene.fluid || !["dam-break", "tank-fill"].includes(scene.fluid.initialCondition)) errors.push("Unsupported fluid initial condition");
  if (scene.fluid?.initialBrickSeeds_m) {
    if (!Array.isArray(scene.fluid.initialBrickSeeds_m) || scene.fluid.initialBrickSeeds_m.length === 0) errors.push("Initial fluid brick seeds must be a non-empty array");
    else for (const [index, seed] of scene.fluid.initialBrickSeeds_m.entries()) {
      if (![seed?.x, seed?.y, seed?.z].every(Number.isFinite)) errors.push(`Initial fluid brick seed ${index} must be finite`);
      else if (seed.x < -c.width_m / 2 || seed.x >= c.width_m / 2 || seed.y < 0 || seed.y >= c.height_m || seed.z < -c.depth_m / 2 || seed.z >= c.depth_m / 2) {
        errors.push(`Initial fluid brick seed ${index} must be inside the solver bounds`);
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
  if (scene.terrain && c) errors.push(...validateTerrain(scene.terrain, c));
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
