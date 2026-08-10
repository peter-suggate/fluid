/**
 * The parts of a headless SVO dry-frame run that must not be written twice.
 *
 * `tools/benchmark-svo-dry-frame-gpu.ts` grew four blocks that are not
 * benchmark logic at all — they are mirrors of what `FluidLabRenderer` does on
 * every presentation frame, and they are the reason a headless frame is worth
 * anything as evidence:
 *
 *   - Dawn/Metal bring-up with `requiredFluidDeviceLimits`, which is what keeps
 *     the derived-lighting stack affordable; a device that cannot host it
 *     silently withdraws the opacity pyramid and renders ~15x slow.
 *   - The 416-byte view-uniform packing.
 *   - The CPU rigid-body packing.
 *   - The `SparseVoxelDrySceneData` assembly: primitives, glass, thick glass,
 *     the material table, the terrain heightfield and the lighting mirrors.
 *
 * A second consumer that copied those would not be testing production, it would
 * be testing its copy of production. So they live here, and both the benchmark
 * and `tools/run-svo-dry-render-smoke.ts` call them.
 *
 * Also here: `svoScenePrimitiveBrickDensity`, the CPU mirror of the live
 * voxelizer's per-brick candidate binning, previously private inside
 * `tools/svo-fine-voxel-capacity.ts`. It is the only way to see the silent
 * `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` drop from the host, because the GPU
 * raises `SPARSE_SCENE_MAINTENANCE_OVERFLOW.candidates` and then deliberately
 * declines to let it block the revision (`lib/webgpu-sparse-scene-proxies.ts`),
 * with no readback path to any CPU consumer.
 *
 * This module has no top-level side effects and reads no environment variables;
 * every caller passes what it wants explicitly.
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { environmentIndex, type EnvironmentId } from "../lib/environments";
import { cameraPosition } from "../lib/math";
import { sceneUsesFlatVoxelNormals, type CameraState, type SceneDescription } from "../lib/model";
import { cameraTanHalfFov } from "../lib/webgpu-camera";
import { boundingRadius, initializeRigidBodies } from "../lib/rigid-body";
import {
  buildDefaultSvoMaterialRecords,
  packSvoMaterialTable,
  svoMaterialFromEnvironmentProxyMaterial,
  svoMaterialFunctionIdForEnvironmentProxy,
} from "../lib/svo-material-abi";
import { svoPrimitiveCandidateBounds } from "../lib/svo-primitive-candidates";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoScenePrimitives, type SvoScenePrimitiveBuild } from "../lib/svo-scene-primitives";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildSvoTerrainMaterial, sceneTerrainSurfaceModel } from "../lib/svo-terrain-material";
import {
  MAX_TERRAIN_FEATURES,
  sceneHasTerrain,
  TERRAIN_DEFAULT_FLAT,
  TERRAIN_UNION_EXPONENT,
  terrainSampleGrid,
} from "../lib/terrain";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { SVO_CAMERA_CHANGING_FRAME } from "../lib/webgpu-renderer";
import {
  buildSparseVoxelDrySceneLightingMirrors,
  packSvoDrySceneTerrainHeightfield,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelSceneRenderSource } from "../lib/webgpu-voxel-debug";

/** Floats in the 416-byte view uniform block (`FluidLabRenderer`, webgpu-renderer.ts). */
export const SVO_VIEW_UNIFORM_FLOATS = 104;

// ---------------------------------------------------------------------------
// Dawn/Metal bring-up
// ---------------------------------------------------------------------------

export interface DawnRenderDeviceOptions {
  /**
   * Path to the `webgpu` node module. Defaults to the repo's own, which is what
   * `WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js` names explicitly in
   * every `test:webgpu:*` script.
   */
  readonly modulePath?: string;
  /** `enable-dawn-features=` list, e.g. `["skip_validation"]`. */
  readonly dawnFeatures?: readonly string[];
  /** Fail closed rather than fall back to wall timing. */
  readonly requireTimestampQuery?: boolean;
  readonly requireShaderF16?: boolean;
  readonly label?: string;
}

/**
 * Every Dawn instance this process has created, retained forever.
 *
 * dawn-node's `GPU` object owns the native Dawn instance that the adapter and
 * device hang off. When bring-up lived at a benchmark's top level the instance
 * was a module-level `const` and outlived the process by construction; moved
 * behind a function it became a local, and V8 collected it as soon as the call
 * returned — taking the live device with it and segfaulting the next command
 * submission. A returned reference alone is not enough, because a caller that
 * destructures only `{ device }` drops it again.
 */
const retainedDawnInstances: GPU[] = [];

export interface DawnRenderDevice {
  /** The Dawn instance, also retained module-side; never let it be collected. */
  readonly gpu: GPU;
  readonly adapter: GPUAdapter;
  readonly adapterInfo: { vendor: string; architecture: string; device: string; description: string };
  readonly device: GPUDevice;
  readonly timestampsSupported: boolean;
  /**
   * Live list of uncaptured Dawn validation messages. Empty is part of every
   * lane's contract; a nonempty list means the frame that was measured is not
   * the frame production would draw.
   */
  readonly validationErrors: string[];
}

/**
 * Bring up Dawn on Metal with the production device limits.
 *
 * `requiredFluidDeviceLimits(adapter.limits)` is load-bearing rather than
 * defensive: it requests the adapter's advertised `maxTextureDimension2D`
 * (16384 on an M1 Max) instead of WebGPU's 8192 default, which is what sizes
 * the node-mip page directory. A device without it can fail to host the
 * pyramid, and cone lighting then falls back to exact traversal for every
 * shadow and GI ray with only a `console.warn`.
 *
 * Two things measured here 2026-08-03, because the record says otherwise and
 * cost an afternoon: the page ceiling is *not* `maxTextureDimension2D`, it is
 * `webGpuSvoNodeMipMaximumPages` = `limit * floor(limit / 2)` with a 2048
 * floor, so even the smallest device that function accepts offers ~2M pages.
 * And dawn-node ignores a `requiredLimits` entry requesting *less* than the
 * adapter advertises, so device limits cannot be used to inject this fault.
 */
export async function createDawnRenderDevice(options: DawnRenderDeviceOptions = {}): Promise<DawnRenderDevice> {
  const modulePath = options.modulePath
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const dawnFeatures = (options.dawnFeatures ?? []).map((feature) => feature.trim()).filter(Boolean);
  const gpu = create([
    "backend=metal",
    ...(dawnFeatures.length > 0 ? [`enable-dawn-features=${dawnFeatures.join(",")}`] : []),
  ]);
  retainedDawnInstances.push(gpu);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "no Metal adapter — this lane did not execute on the GPU");
  const adapterInfo = {
    vendor: adapter.info?.vendor ?? "",
    architecture: adapter.info?.architecture ?? "",
    device: adapter.info?.device ?? "",
    description: adapter.info?.description ?? "",
  };
  assert.ok(adapterInfo.vendor || adapterInfo.architecture || adapterInfo.description,
    "adapter info is empty — refusing to report a result that may not have run on real hardware");
  const timestampsSupported = adapter.features.has("timestamp-query");
  assert.ok(!options.requireTimestampQuery || timestampsSupported,
    "timestamp-query was required but the adapter cannot provide it; no wall-time fallback was selected");
  assert.ok(!options.requireShaderF16 || adapter.features.has("shader-f16"),
    "shader-f16 was required but Dawn does not advertise it");
  const device = await adapter.requestDevice({
    label: options.label,
    requiredFeatures: [
      ...(timestampsSupported ? ["timestamp-query" as GPUFeatureName] : []),
      ...(options.requireShaderF16 ? ["shader-f16" as GPUFeatureName] : []),
      // Mirrors `fluidExecutionDeviceFeatures`: without it the live radiance
      // atlas cannot use `rg11b10ufloat` storage and silently doubles, so this
      // lane would measure an allocation the browser renderer never pays.
      ...(adapter.features.has("texture-formats-tier1") ? ["texture-formats-tier1" as GPUFeatureName] : []),
    ],
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) =>
    validationErrors.push((event as GPUUncapturedErrorEvent).error.message));
  return { gpu, adapter, adapterInfo, device, timestampsSupported, validationErrors };
}

// ---------------------------------------------------------------------------
// Uniform mirrors of FluidLabRenderer
// ---------------------------------------------------------------------------

export interface SvoDryViewUniformInputs {
  readonly scene: SceneDescription;
  readonly camera: CameraState;
  readonly environmentId: EnvironmentId;
  readonly info: { nx: number; ny: number; nz: number };
  readonly bodyCount: number;
  readonly width: number;
  readonly height: number;
  /** Publishes the camera-changing sentinel, selecting the moving-quality tier. */
  readonly cameraMoving?: boolean;
  readonly overlay?: { mode: number; opacity: number };
}

/** Mirror of the FluidLabRenderer 416-byte view-uniform packing (webgpu-renderer.ts). */
export function packSvoDryViewUniforms(inputs: SvoDryViewUniformInputs): Float32Array<ArrayBuffer> {
  const { scene, camera, environmentId, info, bodyCount, width, height, overlay } = inputs;
  const position = cameraPosition(camera);
  // options.x: DEFAULT_SVO_RENDER_DIAGNOSTICS maximumTraversalDepth(21)*512 + maximumNodeVisits(256).
  const diagnosticControl = 21 * 512 + 256;
  // viewport.w carries only the camera-quality tier: the changing sentinel or
  // the stable sentinel. Temporal accumulation and checkerboard phase no longer
  // share this lane.
  const cameraState = inputs.cameraMoving ? SVO_CAMERA_CHANGING_FRAME : -1;
  const uniform = new Float32Array([
    width, height, 0, cameraState,
    // cameraPosition.w is the authored vertical aperture, exactly as it is in
    // FluidLabRenderer. Overlay mode belongs to the diagnostic lane below; it
    // must not silently turn a photographic scene lens back into the 0.72
    // historical fallback in headless evidence renders.
    position.x, position.y, position.z, cameraTanHalfFov(camera),
    camera.target_m.x, camera.target_m.y, camera.target_m.z, overlay ? overlay.opacity : 0.82,
    scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
    diagnosticControl, scene.voxelDomain.finestCellSize_m, Math.min(bodyCount, 12), 1,
    info.nx, info.ny, info.nz, 3,
    0, 0.5, 1, 0,
    environmentIndex(environmentId), 0, 0, 0,
  ]);
  const packed = new Float32Array(SVO_VIEW_UNIFORM_FLOATS);
  packed.set(uniform, 0);
  if (sceneHasTerrain(scene) && scene.terrain) {
    const terrain = scene.terrain;
    const features = terrain.features.slice(0, MAX_TERRAIN_FEATURES);
    packed.set([1, terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
    features.forEach((feature, index) => {
      packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
      packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
    });
  }
  return packed;
}

/** Mirror of the FluidLabRenderer CPU rigid-body packing (webgpu-renderer.ts). */
export function packSvoDryRigidBodies(scene: SceneDescription): { data: Float32Array<ArrayBuffer>; count: number } {
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const bodyData = new Float32Array(12 * 16);
  const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
  const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
  bodies.slice(0, 12).forEach((body, index) => {
    const offset = index * 16;
    const d = body.description.dimensions_m;
    const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
    const color = palette[shapeIndex[body.description.shape]];
    bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
    bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
    bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
    bodyData.set([color[0], color[1], color[2], 0], offset + 12);
  });
  return { data: bodyData, count: bodies.length };
}

// ---------------------------------------------------------------------------
// SparseVoxelDrySceneData assembly
// ---------------------------------------------------------------------------

export interface SvoDrySceneAssembly {
  readonly drySceneData: SparseVoxelDrySceneData;
  readonly scenePrimitives: SvoScenePrimitiveBuild;
  readonly sceneGlass: ReturnType<typeof buildSvoSceneGlass>;
  readonly sceneThickGlass: ReturnType<typeof buildSvoSceneThickGlass>;
  readonly terrainSurface: ReturnType<typeof sceneTerrainSurfaceModel>;
}

/**
 * Exact mirror of the FluidLabRenderer solver-attachment dry-scene data
 * assembly. `source` must already carry a published `structural` domain; the
 * glass builder needs its cell size to decide pane thickness.
 */
export function buildSvoDrySceneAssembly(
  scene: SceneDescription,
  source: SparseVoxelSceneRenderSource,
): SvoDrySceneAssembly {
  const scenePrimitives = buildSvoScenePrimitives(scene);
  const sceneGlass = buildSvoSceneGlass(scene, { cellSize_m: source.structural!.domain.cellSize_m });
  const sceneThickGlass = buildSvoSceneThickGlass(scene);
  const thickReplacedPaneKey = sceneThickGlass.metadata.find(({ replacesThinPaneKey }) => Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
  const thickReplacedPaneId = sceneGlass.metadata.find(({ key }) => key === thickReplacedPaneKey)?.paneId;
  const terrainSurface = sceneTerrainSurfaceModel(scene);
  const terrainMaterial = scenePrimitives.analyticTerrain && terrainSurface === "garden-terrain"
    ? buildSvoTerrainMaterial(scene)
    : undefined;
  const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
  const lightingMirrors = buildSparseVoxelDrySceneLightingMirrors(scene, 1);
  const drySceneData: SparseVoxelDrySceneData = {
    renderRevision: 1,
    primitiveRecords: scenePrimitives.packedRecords,
    primitiveCandidates: scenePrimitives.primitiveCandidates!,
    materialRecords: packSvoMaterialTable([
      ...buildDefaultSvoMaterialRecords(1, { terrainSurface }),
      ...scenePrimitives.metadata.map((primitive) => svoMaterialFromEnvironmentProxyMaterial(
        primitive.materialId,
        primitive.material,
        1,
        // Keep the headless production mirror on the same scene-wide closure
        // as FluidLabRenderer. Otherwise a porcelain scene silently falls back
        // to per-object semantic materials here (for example, granite coping),
        // invalidating every material and primary-visibility comparison.
        svoMaterialFunctionIdForEnvironmentProxy(primitive, terrainSurface),
      )),
    ]),
    materialRevision: 1,
    ownerBase: scene.rigidBodies.length,
    skippedOwnerId: scenePrimitives.openShellOwnerId,
    terrainMaterialId: scenePrimitives.analyticTerrain?.materialId,
    terrainMaterialMetadata: terrainMaterial?.packedMetadata,
    terrainMaterialCacheKey: terrainMaterial?.cacheKey,
    // A sculpted vessel is terrain the eight-feature uniform mirror cannot
    // express. Production publishes the grid into the scene arena; without this
    // a headless frame renders a flat plane at `baseHeight_m` and calls it the scene.
    terrainHeightfield: packSvoDrySceneTerrainHeightfield(terrainSampleGrid(scene.terrain)),
    // The same door, for the aggregate kind. Without it every cluster record
    // names a zeroed arena block, the shader reads that as "not resolved", and
    // the crown of the hero's bonsai draws as nothing at all — which is what a
    // headless frame showed before this line existed.
    clusterBlocks: scenePrimitives.clusterBlocks,
    // And the same door for the tape kind, which fails identically without it:
    // every `field-program` record would name a zeroed block and draw nothing.
    fieldProgramBlocks: scenePrimitives.fieldProgramBlocks,
    glassRecords: sceneGlass.packedRecords,
    glassCacheKey: sceneGlass.cacheKey,
    thickGlassRecords: sceneThickGlass.packedRecords,
    thickGlassRevision: sceneThickGlass.revision,
    thickGlassCacheKey: sceneThickGlass.cacheKey,
    thickGlassReplacedThinPaneId: thickReplacedPaneId,
    primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass[0]?.paneId,
    primaryCompositeOwnedGlassPaneCount: compositorOwnedGlass.length,
    ...lightingMirrors,
    // Presentation policy is scene data too. Omitting this made every headless
    // fidelity frame silently exercise the default baked analytic normals even
    // when the product document requested voxel-flat surfaces.
    flatVoxelNormals: sceneUsesFlatVoxelNormals(scene),
  };
  return { drySceneData, scenePrimitives, sceneGlass, sceneThickGlass, terrainSurface };
}

// ---------------------------------------------------------------------------
// Per-brick candidate density
// ---------------------------------------------------------------------------

export interface SvoBrickLattice {
  /** World-space position of cell (0, 0, 0)'s minimum corner. */
  readonly worldOrigin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  readonly dimensionsCells: readonly [number, number, number];
  readonly brickSize: number;
}

export interface SvoBrickDensity {
  /** Finite primitives binned; the terrain heightfield is not one of them. */
  readonly primitives: number;
  readonly occupiedBricks: number;
  readonly maximumPerBrick: number;
  /** Bricks holding more than `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK`. */
  readonly overflowedBricks: number;
  readonly brickEdge_m: number;
}

/**
 * How crowded the busiest brick gets.
 *
 * The live voxelizer keeps `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` primitives
 * per dirty brick and raises `CANDIDATE_OVERFLOW` for the rest, which are then
 * simply not written into that brick's voxels
 * (`lib/webgpu-sparse-scene-proxies.ts`). That makes the ceiling on *generated
 * objects* a density rather than a total: 64 primitives per (brickSize x cell)
 * cube of world, which shrinks by 8x every time the lattice halves. A scene
 * library of parameterised species instantiated freely runs into this long
 * before it runs into the 4 096 record arenas.
 *
 * This is a CPU mirror, and deliberately a conservative one: it spreads each
 * primitive over every brick its candidate AABB touches, exactly as
 * `binDirtyBrickCandidates` does on the GPU, so it counts a primitive in a
 * brick the surface may not actually reach. It over-reports rather than
 * under-reports, which is the right direction for a ceiling nobody can observe
 * at runtime — the GPU raises the overflow bit and then deliberately declines
 * to let it block the revision, with no readback path to any CPU consumer.
 */
export function svoScenePrimitiveBrickDensity(
  descriptors: SvoScenePrimitiveBuild["descriptors"],
  lattice: SvoBrickLattice,
  candidatesPerBrick: number,
): SvoBrickDensity {
  const { worldOrigin_m: origin, cellSize_m: cellSize, dimensionsCells: cells, brickSize } = lattice;
  const bricks = cells.map((value) => Math.max(1, Math.ceil(value / brickSize))) as [number, number, number];
  const counts = new Map<number, number>();
  let primitives = 0;
  for (const descriptor of descriptors) {
    if (descriptor.kind === "terrain-heightfield") continue;
    primitives += 1;
    const bounds = svoPrimitiveCandidateBounds(descriptor as never);
    const minimum = [bounds.minimum_m.x, bounds.minimum_m.y, bounds.minimum_m.z];
    const maximum = [bounds.maximum_m.x, bounds.maximum_m.y, bounds.maximum_m.z];
    const first = minimum.map((value, axis) => Math.max(0, Math.floor((value - origin[axis]) / (cellSize[axis] * brickSize))));
    const last = maximum.map((value, axis) => Math.min(bricks[axis] - 1, Math.floor((value - origin[axis]) / (cellSize[axis] * brickSize))));
    for (let bz = first[2]; bz <= last[2]; bz += 1) for (let by = first[1]; by <= last[1]; by += 1) for (let bx = first[0]; bx <= last[0]; bx += 1) {
      const key = (bz * bricks[1] + by) * bricks[0] + bx;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  let maximum = 0;
  let overflowed = 0;
  for (const count of counts.values()) {
    maximum = Math.max(maximum, count);
    if (count > candidatesPerBrick) overflowed += 1;
  }
  return {
    primitives,
    occupiedBricks: counts.size,
    maximumPerBrick: maximum,
    overflowedBricks: overflowed,
    brickEdge_m: cellSize[0] * brickSize,
  };
}
