/**
 * Production scene -> native centre-Z plane for the CPU Advance Lab.
 *
 * This is a reduction, not a second scene catalogue.  The scene document,
 * SolidWorld, initial liquid raster and generation-zero sparse atlas all come
 * from the same constructors used by the WebGPU Sparse CM12 solver.  Canvas Y
 * grows down, so scalar rows and Y velocity/gravity are reflected exactly once
 * at this boundary.
 */
import { CM12_PAPER_DT_S } from "../../../core/cm12-numerics";
import {
  baseInitialLiquidFractionAtCell,
  initialLiquidFractionAtCell,
} from "../../../core/initial-fluid";
import { resolveMethodValues } from "../../../core/method-contract";
import type { RigidBodyDescription, SceneDescription, Vec3 } from "../../../core/model";
import { quaternionInverseRotate } from "../../../core/rigid-body";
import { sceneDocument, type SceneDefinition } from "../../../core/scene-definition";
import { sceneCellSizes_m, sceneLatticeDimensions } from "../../../core/scene-lattice";
import { sceneShape } from "../../../core/scene-shape";
import { SCENE_CATALOG } from "../../../core/scenes";
import {
  fluidSolidWorldForScene,
  sampleSolidWorld,
  type SolidWorld,
} from "../../../core/solid-world";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from "../method";
import {
  initializeSparseBrickAtlasFromScene,
  sparseCM12InitialActiveBrickKeys,
} from "../sparse-brick-atlas";
import type {
  SliceSceneDynamicMetadata,
  SliceSceneSeed,
} from "./slice-scene-seed";

export interface AdvanceProductionSceneOption {
  readonly id: string;
  readonly label: string;
  readonly blurb: string;
  readonly audience: SceneDefinition["audience"];
  readonly shelf: string;
}

/** Full authored production catalogue, in the library's stable reading order. */
export const ADVANCE_PRODUCTION_SCENES: readonly AdvanceProductionSceneOption[] =
  Object.freeze(SCENE_CATALOG.map((definition) => Object.freeze({
    id: definition.id,
    label: definition.name,
    blurb: definition.blurb,
    audience: definition.audience,
    shelf: definition.shelf,
  })));

export const DEFAULT_ADVANCE_PRODUCTION_SCENE_ID = "water-box-dam-break";

export function advanceProductionSceneDefinition(id: string): SceneDefinition {
  const definition = SCENE_CATALOG.find((candidate) => candidate.id === id);
  if (!definition) throw new RangeError(`Unknown production scene ${id}`);
  return definition;
}

const cellIndex = (nx: number, x: number, canvasY: number): number => canvasY * nx + x;
const xRowIndex = (nx: number, x: number, canvasY: number): number => canvasY * (nx + 1) + x;
const yRowIndex = (nx: number, x: number, canvasYFace: number): number => canvasYFace * nx + x;

function bodyContains(body: RigidBodyDescription, world: Vec3): boolean {
  const local = quaternionInverseRotate(body.orientation, {
    x: world.x - body.position_m.x,
    y: world.y - body.position_m.y,
    z: world.z - body.position_m.z,
  });
  return sceneShape(body.shape).inside(body.dimensions_m, local);
}

/**
 * Two-dimensional counterpart of production's rigid quadrature.
 *
 * Production samples eight offsets through the containing 3D voxel.  A
 * literal plane has no z thickness, so the reduced capacity samples the four
 * matching x/y offsets at physical z=0.  The untouched 3D body descriptor and
 * atlas remain on the seed so this deliberate dimensional reduction is
 * inspectable rather than confused with a production voxel average.
 */
function rigidSolidFractionAtPlane(
  scene: SceneDescription,
  sourceX: number,
  sourceY: number,
  cellSize: readonly [number, number, number],
): number {
  if (scene.rigidBodies.length === 0) return 0;
  const center = {
    x: -0.5 * scene.container.width_m + (sourceX + 0.5) * cellSize[0],
    y: (sourceY + 0.5) * cellSize[1],
    z: 0,
  };
  let maximum = 0;
  for (const body of scene.rigidBodies) {
    let inside = 0;
    for (let corner = 0; corner < 4; corner += 1) {
      const point = {
        x: center.x + ((corner & 1) ? 0.4 : -0.4) * cellSize[0],
        y: center.y + ((corner & 2) ? 0.4 : -0.4) * cellSize[1],
        z: center.z,
      };
      if (bodyContains(body, point)) inside += 1;
    }
    maximum = Math.max(maximum, inside / 4);
  }
  return maximum;
}

function rigidVelocityAt(body: RigidBodyDescription, world: Vec3): Vec3 {
  const arm = {
    x: world.x - body.position_m.x,
    y: world.y - body.position_m.y,
    z: world.z - body.position_m.z,
  };
  const angular = body.angularVelocity_rad_s;
  return {
    x: body.linearVelocity_m_s.x + angular.y * arm.z - angular.z * arm.y,
    y: body.linearVelocity_m_s.y + angular.z * arm.x - angular.x * arm.z,
    z: body.linearVelocity_m_s.z + angular.x * arm.y - angular.y * arm.x,
  };
}

/** First authored body owns a point, matching production's rigid owner loop. */
function rigidOwnerAt(scene: SceneDescription, world: Vec3): RigidBodyDescription | undefined {
  return scene.rigidBodies.find((body) => bodyContains(body, world));
}

function staticFaceOpen(
  world: SolidWorld,
  negative: readonly [number, number, number],
  positive: readonly [number, number, number],
): number {
  return 1 - Math.max(
    sampleSolidWorld(world, negative).solidFraction,
    sampleSolidWorld(world, positive).solidFraction,
  );
}

function dynamicMetadata(scene: SceneDescription): SliceSceneDynamicMetadata[] {
  const entries: SliceSceneDynamicMetadata[] = scene.rigidBodies.map((body) => ({
    kind: "rigid",
    label: body.name,
    supported: true,
    detail: "The production descriptor, pose/contact integration, moving-capacity GCL and delayed fluid-reaction exchange advance on the centre slab.",
    payload: {
      id: body.id,
      shape: body.shape,
      dimensions_m: body.dimensions_m,
      density_kg_m3: body.density_kg_m3,
      position_m: body.position_m,
      orientation: body.orientation,
      linearVelocity_m_s: body.linearVelocity_m_s,
      angularVelocity_rad_s: body.angularVelocity_rad_s,
      motion: body.motion ?? "dynamic",
    },
  }));
  if (scene.fluid.inflow) {
    const velocity = scene.fluid.inflow.velocity_m_s;
    const zDominant = Math.abs(velocity.z) > Math.max(Math.abs(velocity.x), Math.abs(velocity.y));
    entries.push({
    kind: "source",
    label: "Fluid inflow",
    supported: !zDominant,
    detail: zDominant
      ? "The production nozzle emits primarily through z. Its descriptor is retained, but the 2D port emits no invented in-plane source."
      : "The production timing, outlet geometry and source ledger are advanced as a unit-depth centre-plane reduction.",
    payload: { ...scene.fluid.inflow },
  });
  }
  return entries;
}

function limitationsFor(
  scene: SceneDescription,
  density: Float32Array,
  centerCellZ: number,
): string[] {
  const limitations = [
    `The physical z=0 plane uses production cell z=${centerCellZ} under the lattice's half-open cell convention.`,
    "Static SolidWorld, material and initial-liquid scalars are production volume averages in that containing source voxel. Analytic rigid geometry is instead intersected at literal z=0, because a 2D control volume has no z thickness.",
  ];
  const symmetric = scene.container.depthBoundary === "symmetry";
  if (!symmetric) limitations.push(
    "This 2D advance omits z-face transport, ∂w/∂z, and pressure coupling through the two z faces. The centre plane is an exact initial-state section; later frames are equivalent to production only for z-invariant flow.",
  );
  if (scene.systems?.fluid === false) limitations.push(
    "This production scene disables fluid, so its centre slice is intentionally dry.",
  );
  if (!density.some((value) => value > 0)) limitations.push(
    "No authored liquid intersects the centre-Z source cell; the truthful slice starts empty.",
  );
  if (scene.rigidBodies.length > 0) limitations.push(
    "Rigid bodies are intersected at physical z=0 with the 2D four-point counterpart of production's eight-point voxel quadrature. Pose/contact integration retains the full production shape descriptor; displaced volume and mean-fluid reaction are the containing centre slab's dimensional reduction.",
  );
  if (scene.fluid.inflow) limitations.push(
    "An in-plane inflow uses the production timing, outlet weights and pending ledger, reduced to unit depth at z=0. Its 2D area rate is the nozzle's centre-plane chord flux, not the complete 3D discharge.",
  );
  if (scene.fluid.inflow && Math.abs(scene.fluid.inflow.velocity_m_s.z)
    > Math.max(Math.abs(scene.fluid.inflow.velocity_m_s.x),
      Math.abs(scene.fluid.inflow.velocity_m_s.y))) limitations.push(
    "The authored nozzle points primarily through z. The 2D reduction reports it but injects no in-plane liquid or velocity.",
  );
  if (Math.abs(scene.fluid.gravity_m_s2.z) > 0) limitations.push(
    `The production z acceleration (${scene.fluid.gravity_m_s2.z} m/s²) is outside the 2D plane.`,
  );
  if (Math.abs(scene.fluid.initialVelocity_m_s?.z ?? 0) > 0) limitations.push(
    `The production initial z velocity (${scene.fluid.initialVelocity_m_s!.z} m/s) is retained for inspection and omitted from 2D transport.`,
  );
  return limitations;
}

/** Reader-chosen departures from what the scene document itself asks for. */
export interface ProductionSceneSliceOverrides {
  /**
   * Step size, in seconds, in place of the document's own.
   *
   * A scene carries the step its quality profile resolved to — the CM12 paper
   * regime gives 1/30 s, other profiles give the document's `fixedDt_s`. A lab
   * that compares scenes side by side wants one step for all of them, so it
   * says so here rather than inheriting a per-scene one it never chose.
   */
  readonly dt?: number;
}

/**
 * Construct an exact native finest-lattice readback and production sparse
 * generation for one authored scene's physical centre-Z plane.
 */
export function productionSceneSliceSeed(definition: SceneDefinition,
  overrides: ProductionSceneSliceOverrides = {}): SliceSceneSeed {
  const scene = sceneDocument(definition);
  const dimensions3 = sceneLatticeDimensions(scene) as [number, number, number];
  const [nx, ny, nz] = dimensions3;
  const centerCellZ = Math.floor(nz / 2);
  const cellSize = sceneCellSizes_m(scene) as [number, number, number];
  const world = fluidSolidWorldForScene(scene);

  const values = resolveMethodValues(
    adaptiveMassMethod,
    definition.methodProfile?.methodId === "adaptive-volume"
      ? definition.methodProfile.quality : "balanced",
    definition.methodProfile?.methodId === "adaptive-volume"
      ? definition.methodProfile.overrides : {},
  );
  const options = adaptiveMassSolverOptions(values);
  const curvedInitialLiquidNeedsFineFrontier = scene.fluid.initialLiquidVolumes
    ?.some((volume) => volume.shape !== "box") ?? false;
  const sourceAtlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions3,
    brickFineResolution: options.brickFineResolution,
    solidWorld: world,
    maximumMacroSpanBricks: options.maximumMacroSpanBricks,
    surfaceFineRings: options.surfaceFineRings,
    coarseFirstCurvatureTolerance: options.activityPolicy?.coarseFirst
      ? options.activityPolicy.curvatureTolerance : undefined,
    initialSurfaceCoarseningBiasRings: options.activityPolicy?.activitySignals
      && !curvedInitialLiquidNeedsFineFrontier ? 1 : 0,
  });
  const initiallyActiveBrickKeys = sparseCM12InitialActiveBrickKeys(
    scene,
    sourceAtlas,
    options.activityPolicy?.coarseFirst && !scene.fluid.refinementRegions?.length ? 2 : 1,
  );

  const capacity = new Float32Array(nx * ny);
  const density = new Float32Array(nx * ny);
  const materialId = new Uint16Array(nx * ny);
  for (let sourceY = 0; sourceY < ny; sourceY += 1) for (let x = 0; x < nx; x += 1) {
    const canvasY = ny - 1 - sourceY;
    const i = cellIndex(nx, x, canvasY);
    const solid = sampleSolidWorld(world, [x, sourceY, centerCellZ]);
    const staticOpen = 1 - solid.solidFraction;
    const rigid = rigidSolidFractionAtPlane(scene, x, sourceY, cellSize);
    capacity[i] = staticOpen * (1 - rigid);
    const base = baseInitialLiquidFractionAtCell(scene, x, sourceY,
      centerCellZ, dimensions3);
    // This is exactly the generation-zero Sparse CM12 source scalar.  It uses
    // the static SolidWorld; moving-body capacity is a later geometric stage.
    density[i] = staticOpen * initialLiquidFractionAtCell(
      scene, x, sourceY, centerCellZ, dimensions3, base);
    materialId[i] = solid.materialId;
  }

  const velocity = scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
  const velocityX = new Float32Array((nx + 1) * ny);
  const velocityY = new Float32Array(nx * (ny + 1));
  const apertureX = new Float32Array((nx + 1) * ny);
  const apertureY = new Float32Array(nx * (ny + 1));
  const solidVelocityX = new Float32Array((nx + 1) * ny);
  const solidVelocityY = new Float32Array(nx * (ny + 1));
  const velocityZ = new Float32Array(nx * ny);
  for (let canvasY = 0; canvasY < ny; canvasY += 1) for (let x = 0; x <= nx; x += 1) {
    const sourceY = ny - 1 - canvasY;
    const row = xRowIndex(nx, x, canvasY);
    velocityX[row] = velocity.x;
    const staticOpen = x === 0 || x === nx ? 0 : staticFaceOpen(world,
      [x - 1, sourceY, centerCellZ], [x, sourceY, centerCellZ]);
    let covered = 0, wallVelocity = 0;
    const face = {
      x: -0.5 * scene.container.width_m + x * cellSize[0],
      y: (sourceY + 0.5) * cellSize[1],
      z: 0,
    };
    for (const sign of [-0.35, 0.35]) {
      const point = { ...face, y: face.y + sign * cellSize[1] };
      const owner = rigidOwnerAt(scene, point);
      if (!owner) continue;
      covered += 0.5;
      wallVelocity += 0.5 * rigidVelocityAt(owner, point).x;
    }
    let fallbackOwner: RigidBodyDescription | undefined;
    if (covered === 0) for (const axisSign of [-0.4, 0.4]) {
      for (const tangentSign of [-0.4, 0.4]) {
        fallbackOwner ??= rigidOwnerAt(scene, {
          x: face.x + axisSign * cellSize[0],
          y: face.y + tangentSign * cellSize[1], z: 0,
        });
      }
    }
    const rigidOpen = 1 - covered;
    apertureX[row] = staticOpen * rigidOpen;
    solidVelocityX[row] = covered > 0 ? wallVelocity / covered
      : fallbackOwner ? rigidVelocityAt(fallbackOwner, face).x : 0;
  }
  // Source +Y points upward while canvas +Y points down.  Source face fy maps
  // to canvas face ny-fy, with its sign reversed.
  for (let sourceFaceY = 0; sourceFaceY <= ny; sourceFaceY += 1) {
    const canvasFaceY = ny - sourceFaceY;
    for (let x = 0; x < nx; x += 1) {
      const row = yRowIndex(nx, x, canvasFaceY);
      velocityY[row] = -velocity.y;
      const isBottom = sourceFaceY === 0;
      const isTop = sourceFaceY === ny;
      const staticOpen = isBottom || (isTop && scene.container.top !== "open")
        ? 0 : isTop ? 1 : staticFaceOpen(world,
          [x, sourceFaceY - 1, centerCellZ], [x, sourceFaceY, centerCellZ]);
      let covered = 0, wallVelocity = 0;
      const face = {
        x: -0.5 * scene.container.width_m + (x + 0.5) * cellSize[0],
        y: sourceFaceY * cellSize[1],
        z: 0,
      };
      for (const sign of [-0.35, 0.35]) {
        const point = { ...face, x: face.x + sign * cellSize[0] };
        const owner = rigidOwnerAt(scene, point);
        if (!owner) continue;
        covered += 0.5;
        wallVelocity += 0.5 * -rigidVelocityAt(owner, point).y;
      }
      let fallbackOwner: RigidBodyDescription | undefined;
      if (covered === 0) for (const axisSign of [-0.4, 0.4]) {
        for (const tangentSign of [-0.4, 0.4]) {
          fallbackOwner ??= rigidOwnerAt(scene, {
            x: face.x + tangentSign * cellSize[0],
            y: face.y + axisSign * cellSize[1], z: 0,
          });
        }
      }
      const rigidOpen = 1 - covered;
      apertureY[row] = staticOpen * rigidOpen;
      solidVelocityY[row] = covered > 0 ? wallVelocity / covered
        : fallbackOwner ? -rigidVelocityAt(fallbackOwner, face).y : 0;
    }
  }
  velocityZ.fill(velocity.z);

  const dynamic = dynamicMetadata(scene);
  const documentDt = options.timeStep === "paper" ? CM12_PAPER_DT_S
    : scene.numerics.fixedDt_s;
  const dt = overrides.dt ?? documentDt;
  return {
    id: definition.id,
    label: definition.name,
    note: definition.blurb,
    dimensions: [nx, ny],
    boundary: {
      xMin: "closed",
      xMax: "closed",
      yMin: "closed",
      yMax: scene.container.top === "open" ? "open" : "closed",
      z: scene.container.depthBoundary === "symmetry" ? "symmetry" : "omitted",
    },
    viewport: {
      originX: -0.5 * scene.container.width_m,
      originY: 0,
      cellSizeX: cellSize[0],
      cellSizeY: cellSize[1],
      sourceCellSize: scene.voxelDomain.finestCellSize_m,
      centerCellZ,
      sourceCellCenterZ: -0.5 * scene.container.depth_m
        + (centerCellZ + 0.5) * cellSize[2],
      centerZ: 0,
    },
    capacity,
    density,
    velocityX,
    velocityY,
    apertureX,
    apertureY,
    solidVelocityX,
    solidVelocityY,
    velocityZ,
    materialId,
    sourceAtlas,
    gravity: [scene.fluid.gravity_m_s2.x, -scene.fluid.gravity_m_s2.y || 0],
    dt,
    densityKgM3: scene.fluid.density_kg_m3,
    dynamic: dynamic.length ? dynamic : undefined,
    limitations: limitationsFor(scene, density, centerCellZ),
    // Extra production policy retained structurally for the sparse CPU port.
    production: {
      scene,
      values,
      options,
      initiallyActiveBrickKeys,
      solidWorld: world,
    },
  };
}

export function productionSceneSliceSeedById(id: string,
  overrides: ProductionSceneSliceOverrides = {}): SliceSceneSeed {
  return productionSceneSliceSeed(advanceProductionSceneDefinition(id), overrides);
}
