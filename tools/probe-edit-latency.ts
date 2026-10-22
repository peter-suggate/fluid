/**
 * Edit-latency probe: what one edit costs between the user's input and a
 * converged frame, measured on the product's own edit path.
 *
 * The probe is the browser around the product, not a copy of it. It constructs
 * `WebGPURenderWorkerClient` over a canvas, lets the client load
 * `lib/core/webgpu-render-worker.ts` as its worker, and then does what
 * `WebGPUViewport` does: `draw` a document every frame, and hand an edit to
 * `validateLiveSolidEdit`. Which messages cross, in what order, and what the
 * worker's `FluidLabRenderer` does with them is all the product's own code.
 * Client and worker share this one process; messages between them are
 * structured-cloned and delivered on a later turn, as a port would.
 *
 * Everything the probe reports about the GPU comes from one recording layer
 * in front of the device, so the two modes produce the same columns:
 *
 *   --device=fake   a stand-in device that executes nothing. Host time,
 *                   uploads, encoded passes, dispatches and resource creation
 *                   are exact; nothing that needs a readback is reported, and
 *                   the world build uses the product's host-side brick
 *                   selection in place of the GPU one it cannot read back.
 *   --device=dawn   the Dawn/Metal instance of `createDawnRenderDevice`, and
 *                   the device the renderer requests from it. Adds
 *                   wall time per frame, the frame the edit first shows in and
 *                   the frame the image stops changing in, and the statuses
 *                   that arrive by GPU readback (surface mesh, lighting).
 *
 * Internal status is read through what the product already exposes (the
 * worker's status messages, public status getters). Stage timings wrap methods
 * by name where they still exist; a missing one is reported as "not reported",
 * never as a failure of the probe.
 *
 * Run `--help` for the options and the budgets.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { encodeRgbPng } from "../lib/core/png-codec";
import { meshVolumeWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { registerSceneMeshAssets } from "./blender/register-assets";

import { applyDocumentDelta, documentDelta } from "../lib/core/document-delta";
import { resolvedSun, sunAngles, sunDirection, withSun } from "../lib/core/editor-sun";
import { sceneryLightFixture, withSceneryLightColor, withSceneryLightStrength } from "../lib/core/scenery-lights";
import { createEmptyScene } from "../lib/core/empty-scene";
import type { EditorRay } from "../lib/core/editor-entity";
import { scenerySurfaceAt } from "../lib/core/editor-scenery";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { penTool } from "../lib/core/voxel-editor/plugins/pen";
import { toolValues } from "../lib/core/voxel-editor/plugin";
import { beginToolTransaction } from "../lib/core/voxel-editor/transaction";
import type { GPUStatus } from "../lib/core/gpu-status";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/core/model";
import type { EffectiveRendererStatus } from "../lib/core/renderer-status";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneCellSizes_m, sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { cameraForPreset, getScenePreset } from "../lib/core/scenes";
import { sceneWithSolidStroke, type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { terrainHeightAt } from "../lib/core/terrain";
import {
  WebGPURenderWorkerClient,
  type WebGPURenderWorkerRequest,
  type WebGPURenderWorkerResponse,
} from "../lib/core/webgpu-render-worker-client";

const HELP = `Edit-latency probe — how long an edit takes to show, to converge, and what the host and GPU did.

Usage
  node --import tsx tools/probe-edit-latency.ts --device=fake [options]
  WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \\
    node --import tsx tools/probe-edit-latency.ts --device=dawn [options]

Options
  --device=fake|dawn   fake (default): recording stand-in, no GPU; the world build takes the product's
                       host-side brick selection, since nothing can be read back. dawn: the Dawn/Metal
                       device the renderer requests for itself.
  --scene=<id>         hero-garden-hose-x10 (default), any scene preset id, or "hall".
  --edit=<name>[,..]   one or more of the edits below, or "all" (default).
  --fluid=off|scene    off (default): the document's fluid system is switched off, so the renderer owns
                       the live sparse scene directly — the display path. scene: as authored, which also
                       builds the method's solver and its solid-edit path.
  --simulation-steps=N opt-in advancing smoke (default 0, paused clock): requires Dawn and --fluid=scene.
                       Advance N baseline/before/after/undo checkpoint steps and one per pen sample;
                       verify production completion receipts. Interaction scenes use Box3D coupling;
                       other scenes keep rigid-body poses fixed. Not a timing run.
  --size=WxH           viewport in CSS pixels at device pixel ratio 1 (default 480x270).
  --site=x,z           metres; where solid edits land (default: the scene centre, 0,0).
  --pen-node=<text>[,..]
                       restrict pen target to matching node id or asset; a list cycles per stroke.
  --pen-size=N         pen diameter in voxels (default 3, range 1–64).
  --pen-samples=N      samples per pen stroke (default 16, range 2–512); travel grows up to the prop width.
  --pen-camera=<view>  benchmark (default), or close for a closer view of the edited surface.
  --sample-cache=<arm> product (default), or reference to compile the prior uncached imported sampler.
  --pen-images=<dir>   Dawn only: save settled baseline, edited, and undo PNGs outside sample timing.
  --max-frames=N       frame cap per edit (default 600; 240 with --device=dawn).
  --json=<path>        write everything the tables are made from.
  --help

Set FLUID_EDIT_PROBE_PROGRESS=1 to print build status changes on stderr.

Edits
  none            an equal document under a new identity: what any document change costs before its
                  content matters.
  voxel           one voxel filled on the terrain surface at the site.
  erase           the voxel removed again (runs after voxel, over its document).
  voxel-inside    one surface voxel carved where it sits strictly inside an existing brick.
  sphere          an ellipsoid 64 cells across resting on the surface, east of the site.
  slab            a 32 x 1 x 32 brick slab two bricks above the terrain, west of the site.
  undo            back to the base document from one accepted voxel, as an undo does.
  palette-tint    one scenery palette's tint.
  node-material   one scenery node's material record.
  fluid-optics    the fluid's absorption.
  material-model  one scenery node's materialModel.
  pen-draw        16 samples of the live Pen tool on a nearby imported paver/lounger, then undo.
  pen-carve       the same live stroke with Carve, including imported scenery, then undo.
  "all" runs them in that order over one world build, returning to the base document between edits.
  If the renderer stops, the edits left are listed with the command that runs them from a fresh world.

What is reported per edit
  host ms: the accept (call to answer), each instrumented stage, and every frame (draw call to its
  answer); the messages the client sent, the time the caller was held inside the client, the time
  spent structured-cloning each message type and in the worker's handler for it;
  uploads by buffer label; compute passes by label with direct and indirect dispatches; draws;
  GPU objects created after the world was built, by label; frames of scene maintenance and of
  voxelization until quiescent; scene revisions published; which rebuild keys changed, and the
  cost of the rebuild if one ran. With --device=dawn also wall ms per frame (to
  onSubmittedWorkDone), the first frame the edit's screen rectangle differs in, the frame the
  image stops changing in, and the surface-mesh and lighting status.

Budgets (the "budget" column; a row that cannot be measured says so instead of passing)
  host accept                < 5 ms (accept plus the first frame's draw time over an idle frame's)
  uploads                    <= 64 KiB, or 1 KiB per dirty brick with its halo if that is larger
  scene revisions            at most 1: the published scene's revision word moves at most once, and the
                             scene is published at most once to the voxelizer and once to the renderer
  voxelization frames        1 for voxel, voxel-inside, erase, undo and sphere; <= 2 for slab; 0 otherwise
  mesh build (dawn)          a solid edit: incremental, bricks rebuilt <= dirty bricks with a one-brick
                             halo, arena not reallocated; any other edit: no build
  GPU objects created        0 after the world build
  indirect dispatches        0
  rebuild                    none, for every edit; a changed rebuild key is reported as REBUILD
  first visible frame (dawn) <= 2 frames after the accept

The world build is timed separately and is never part of an edit's numbers.`;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

const EDIT_NAMES = ["none", "voxel", "erase", "voxel-inside", "sphere", "slab", "undo",
  "palette-tint", "node-material", "fluid-optics", "material-model", "lamp-colour", "lamp-strength",
  "sun-colour", "sun-strength", "sun-direction", "pen-draw", "pen-carve"] as const;
type EditName = typeof EDIT_NAMES[number];

interface Options {
  device: "fake" | "dawn";
  scene: string;
  edits: EditName[];
  fluid: "off" | "scene";
  simulationSteps: number;
  width: number;
  height: number;
  site: readonly [number, number];
  maxFrames: number;
  penNode?: string;
  penSize: number;
  penSamples: number;
  penCamera: "benchmark" | "close";
  sampleCache: "product" | "reference";
  penImages?: string;
  json?: string;
}

function parseOptions(argv: readonly string[]): Options {
  const values = new Map<string, string>();
  for (const argument of argv) {
    if (argument === "--help" || argument === "-h") { console.log(HELP); process.exit(0); }
    const match = /^--([a-z-]+)=(.*)$/.exec(argument);
    if (!match) throw new Error(`Unrecognised argument "${argument}". Run with --help.`);
    values.set(match[1]!, match[2]!);
  }
  const take = (name: string) => { const value = values.get(name); values.delete(name); return value; };
  const device = take("device") ?? "fake";
  if (device !== "fake" && device !== "dawn") throw new Error(`--device must be fake or dawn, not "${device}".`);
  const fluid = take("fluid") ?? "off";
  if (fluid !== "off" && fluid !== "scene") throw new Error(`--fluid must be off or scene, not "${fluid}".`);
  const requested = (take("edit") ?? "all").split(",").map((name) => name.trim()).filter(Boolean);
  const edits = requested.includes("all") ? [...EDIT_NAMES] : requested.map((name) => {
    if (!(EDIT_NAMES as readonly string[]).includes(name)) {
      throw new Error(`Unknown edit "${name}". One of: ${EDIT_NAMES.join(", ")}, all.`);
    }
    return name as EditName;
  });
  const size = /^(\d+)x(\d+)$/.exec(take("size") ?? "480x270");
  if (!size) throw new Error("--size must look like 480x270.");
  const site = (take("site") ?? "0,0").split(",").map(Number);
  if (site.length !== 2 || site.some((value) => !Number.isFinite(value))) throw new Error("--site must look like 0.4,-1.2.");
  const maxFrames = Number(take("max-frames") ?? (device === "dawn" ? 240 : 600));
  if (!Number.isInteger(maxFrames) || maxFrames < 8) throw new Error("--max-frames must be an integer of at least 8.");
  const options: Options = {
    device, fluid, edits, maxFrames, simulationSteps: Number(take("simulation-steps") ?? 0),
    scene: take("scene") ?? "hero-garden-hose-x10",
    width: Number(size[1]), height: Number(size[2]),
    site: [site[0]!, site[1]!],
    json: take("json"),
    penNode: take("pen-node"), penSize: Number(take("pen-size") ?? 3), penImages: take("pen-images"),
    penSamples: Number(take("pen-samples") ?? 16), penCamera: (take("pen-camera") ?? "benchmark") as Options["penCamera"],
    sampleCache: (take("sample-cache") ?? "product") as Options["sampleCache"],
  };
  if (!Number.isInteger(options.penSize) || options.penSize < 1 || options.penSize > 64) throw new Error("--pen-size must be an integer from 1 to 64.");
  if (!Number.isInteger(options.penSamples) || options.penSamples < 2 || options.penSamples > 512) throw new Error("--pen-samples must be an integer from 2 to 512.");
  if (options.penCamera !== "benchmark" && options.penCamera !== "close") throw new Error("--pen-camera must be benchmark or close.");
  if (options.sampleCache !== "product" && options.sampleCache !== "reference") throw new Error("--sample-cache must be product or reference.");
  if (options.penImages && device !== "dawn") throw new Error("--pen-images requires --device=dawn.");
  if (!Number.isInteger(options.simulationSteps) || options.simulationSteps < 0 || options.simulationSteps > 64) throw new Error("--simulation-steps must be an integer from 0 to 64.");
  if (options.simulationSteps && (device !== "dawn" || fluid !== "scene")) throw new Error("--simulation-steps requires --device=dawn --fluid=scene.");
  if (options.simulationSteps && edits.some(edit => edit !== "pen-draw" && edit !== "pen-carve" && edit !== "none")) throw new Error("Advancing smoke supports --edit=pen-draw,pen-carve or none.");
  if (values.size > 0) throw new Error(`Unrecognised option --${[...values.keys()][0]}. Run with --help.`);
  return options;
}

// ---------------------------------------------------------------------------
// Recording layer
// ---------------------------------------------------------------------------

interface PassTally { direct: number; indirect: number; workgroups: number }
interface GpuTally {
  uploadBytes: number;
  uploads: Record<string, { writes: number; bytes: number }>;
  passes: Record<string, PassTally>;
  directDispatches: number;
  indirectDispatches: number;
  renderPasses: number;
  directDraws: number;
  indirectDraws: number;
  copies: number;
  clears: number;
  submits: number;
  /** GPU objects created, by kind, and the first of them by label. */
  creations: Record<string, number>;
  created: string[];
}
interface StageTally {
  calls: number;
  /** Time on the thread inside the call. */
  ms: number;
  /** For a call that returned a promise: until it settled. */
  settled_ms?: number;
  dispatches: number;
  uploadBytes: number;
  /** Whether any call returned `true`; maintenance methods answer "work remains" this way. */
  returnedTrue?: boolean;
}

const CREATED_LABEL_CAP = 48;
/** Passes the probe can time between two reads. */
const PASS_TIMING_CAPACITY = 2048;
interface GpuPassTimeline {
  /** First recorded beginning to last recorded end; excludes waiting before the first GPU pass. */
  span_ms: number;
  /** Union of pass intervals, avoiding double-counting overlapping backend timestamps. */
  covered_ms: number;
  sum_ms: number;
  intervals: { label: string; begin_ms: number; end_ms: number }[];
}
const emptyTally = (): GpuTally => ({ uploadBytes: 0, uploads: {}, passes: {}, directDispatches: 0, indirectDispatches: 0,
  renderPasses: 0, directDraws: 0, indirectDraws: 0, copies: 0, clears: 0, submits: 0, creations: {}, created: [] });

class Recorder {
  referenceSampleCache = false;
  sampleCacheReplacements = 0;
  readonly bufferMemory = { liveBytes: 0, peakBytes: 0, byLabel: {} as Record<string, number> };
  /** Texel storage the product asked for: every mip and layer, block-compressed formats not modelled. */
  readonly textureMemory = { liveBytes: 0, peakBytes: 0, byLabel: {} as Record<string, number> };
  gpu = emptyTally();
  stages: Record<string, StageTally> = {};
  /** Whoever last ran an instrumented method, by owner name; status getters are read off these. */
  readonly instances = new Map<string, Record<string, unknown>>();
  readonly instrumented: string[] = [];
  readonly notInstrumented: string[] = [];
  private readonly pipelines = new WeakMap<object, string>();
  /** Timestamp queries the probe writes around every pass the product begins. A real device only. */
  private timing?: { device: GPUDevice; querySet: GPUQuerySet; resolved: GPUBuffer; read: GPUBuffer; labels: string[]; untimed: number };
  lastPassTimeline?: GpuPassTimeline;

  /** Time every pass on `device` from here on. */
  timePasses(device: GPUDevice): void {
    if (!device.features.has("timestamp-query")) return;
    const bytes = PASS_TIMING_CAPACITY * 16;
    this.timing = { device, labels: [], untimed: 0,
      querySet: device.createQuerySet({ label: "Probe pass times", type: "timestamp", count: PASS_TIMING_CAPACITY * 2 }),
      resolved: device.createBuffer({ label: "Probe pass times", size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
      read: device.createBuffer({ label: "Probe pass time readback", size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) };
  }

  /** `descriptor` with the probe's timestamps on it, unless the product put its own there. */
  timed<T extends { label?: string; timestampWrites?: unknown }>(descriptor: T | undefined): T | undefined {
    const timing = this.timing;
    if (!timing || descriptor?.timestampWrites) return descriptor;
    if (timing.labels.length === PASS_TIMING_CAPACITY) { timing.untimed += 1; return descriptor; }
    const slot = timing.labels.push(descriptor?.label || "(unlabelled pass)") - 1;
    return { ...descriptor, timestampWrites: { querySet: timing.querySet, beginningOfPassWriteIndex: slot * 2, endOfPassWriteIndex: slot * 2 + 1 } } as unknown as T;
  }

  /** GPU time inside each pass begun since the last call, summed by label. */
  async takePassTimes(): Promise<Record<string, number> | undefined> {
    this.lastPassTimeline = undefined;
    const timing = this.timing;
    if (!timing) return undefined;
    const labels = timing.labels, times: Record<string, number> = {};
    if (timing.untimed) times["(passes beyond the probe's query set)"] = timing.untimed;
    timing.labels = []; timing.untimed = 0;
    if (!labels.length) return times;
    const encoder = timing.device.createCommandEncoder({ label: "Probe pass times" });
    encoder.resolveQuerySet(timing.querySet, 0, labels.length * 2, timing.resolved, 0);
    encoder.copyBufferToBuffer(timing.resolved, 0, timing.read, 0, labels.length * 16);
    timing.device.queue.submit([encoder.finish()]);
    await timing.read.mapAsync(GPUMapMode.READ, 0, labels.length * 16);
    const stamps = new BigUint64Array(timing.read.getMappedRange(0, labels.length * 16));
    const valid = labels.flatMap((label, slot) => {
      const begin = stamps[slot * 2]!, end = stamps[slot * 2 + 1]!;
      return end > begin ? [{ label, begin, end }] : [];
    });
    if (valid.length) {
      const origin = valid.reduce((minimum, interval) => interval.begin < minimum ? interval.begin : minimum, valid[0]!.begin);
      const intervals = valid.map(({ label, begin, end }) => ({ label,
        begin_ms: Number(begin - origin) / 1e6, end_ms: Number(end - origin) / 1e6 }));
      let covered = 0, until = 0;
      for (const interval of [...intervals].sort((a, b) => a.begin_ms - b.begin_ms)) {
        covered += Math.max(0, interval.end_ms - Math.max(until, interval.begin_ms));
        until = Math.max(until, interval.end_ms);
      }
      this.lastPassTimeline = { span_ms: until, covered_ms: covered,
        sum_ms: intervals.reduce((sum, interval) => sum + interval.end_ms - interval.begin_ms, 0), intervals };
    }
    labels.forEach((label, slot) => {
      const began = stamps[slot * 2]!, ended = stamps[slot * 2 + 1]!;
      if (ended > began) times[label] = (times[label] ?? 0) + Number(ended - began) / 1e6;
    });
    timing.read.unmap();
    return times;
  }

  take(): { gpu: GpuTally; stages: Record<string, StageTally> } {
    const taken = { gpu: this.gpu, stages: this.stages };
    this.gpu = emptyTally();
    this.stages = {};
    return taken;
  }

  /** How long each asynchronously compiled pipeline took to resolve, from its request. */
  readonly compiles: { label: string; ms: number }[] = [];

  created(kind: string, label: string | undefined, detail = ""): void {
    this.gpu.creations[kind] = (this.gpu.creations[kind] ?? 0) + 1;
    if (this.gpu.created.length < CREATED_LABEL_CAP) this.gpu.created.push(`${kind}: ${label || "(unlabelled)"}${detail && ` ${detail}`}`);
  }

  upload(label: string | undefined, bytes: number): void {
    const entry = this.gpu.uploads[label || "(unlabelled)"] ??= { writes: 0, bytes: 0 };
    entry.writes += 1;
    entry.bytes += bytes;
    this.gpu.uploadBytes += bytes;
  }

  namePipeline(pipeline: object, descriptor: { label?: string; compute?: { entryPoint?: string } }): void {
    this.pipelines.set(pipeline, descriptor.compute?.entryPoint ?? descriptor.label ?? "(unlabelled pipeline)");
  }

  dispatch(passLabel: string | undefined, pipeline: object | undefined, indirect: boolean, workgroups: number): void {
    const entry = pipeline ? this.pipelines.get(pipeline) : undefined;
    const key = passLabel && entry && passLabel !== entry ? `${passLabel} > ${entry}` : entry ?? passLabel ?? "(unlabelled pass)";
    const pass = this.gpu.passes[key] ??= { direct: 0, indirect: 0, workgroups: 0 };
    if (indirect) { pass.indirect += 1; this.gpu.indirectDispatches += 1; }
    else { pass.direct += 1; pass.workgroups += workgroups; this.gpu.directDispatches += 1; }
  }

  /** Time a method by name on a prototype that may no longer have it. */
  instrument(owner: string, prototype: unknown, methods: readonly string[]): void {
    for (const method of methods) {
      const holder = prototype as Record<string, unknown> | undefined;
      const original = holder?.[method];
      const name = `${owner}.${method}`;
      if (!holder || typeof original !== "function") { this.notInstrumented.push(name); continue; }
      const recorder = this;
      holder[method] = function instrumentedStage(this: Record<string, unknown>, ...args: unknown[]) {
        recorder.instances.set(owner, this);
        const gpu = recorder.gpu;
        const dispatches = gpu.directDispatches + gpu.indirectDispatches, uploads = gpu.uploadBytes;
        const stages = recorder.stages;
        const started = performance.now();
        let result: unknown;
        try { result = (original as (...values: unknown[]) => unknown).apply(this, args); }
        finally {
          const stage = stages[name] ??= { calls: 0, ms: 0, dispatches: 0, uploadBytes: 0 };
          stage.calls += 1;
          stage.ms += performance.now() - started;
          stage.dispatches += gpu.directDispatches + gpu.indirectDispatches - dispatches;
          stage.uploadBytes += gpu.uploadBytes - uploads;
        }
        const stage = stages[name]!;
        if (result === true) stage.returnedTrue = true;
        if (result && typeof (result as PromiseLike<unknown>).then === "function") {
          const settle = () => { stage.settled_ms = (stage.settled_ms ?? 0) + performance.now() - started; };
          (result as PromiseLike<unknown>).then(settle, settle);
        }
        return result;
      };
      this.instrumented.push(name);
    }
  }
}

function writtenBytes(data: ArrayBufferLike | ArrayBufferView, dataOffset = 0, size?: number): number {
  const elementBytes = ArrayBuffer.isView(data) ? (data as unknown as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1 : 1;
  return size !== undefined ? size * elementBytes : data.byteLength - dataOffset * elementBytes;
}

/** Pass every property through to `target`, binding its methods to it, except the ones named. */
function overlay<T extends object>(target: T, overrides: Record<string, unknown>): T {
  const bound = new Map<string | symbol, unknown>();
  return new Proxy(target, {
    get(inner, property) {
      if (typeof property === "string" && property in overrides) return overrides[property];
      const value = Reflect.get(inner, property, inner);
      if (typeof value !== "function") return value;
      let method = bound.get(property);
      if (!method) { method = (value as (...args: unknown[]) => unknown).bind(inner); bound.set(property, method); }
      return method;
    },
  });
}

type Method = (...args: never[]) => unknown;

/** Bytes per texel of the uncompressed formats the product allocates; an unknown format counts four. */
function texelBytes(format: string): number {
  if (format === "depth24plus-stencil8" || format === "depth32float-stencil8") return 8;
  if (format === "depth24plus" || format === "depth32float") return 4;
  if (format === "depth16unorm") return 2;
  if (format === "rg11b10ufloat" || format === "rgb9e5ufloat" || format === "rgb10a2unorm" || format === "rgb10a2uint") return 4;
  const channels = /^(rgba|bgra|rg|r)(\d+)/.exec(format);
  return channels ? channels[1]!.replace("bgra", "rgba").length * Number(channels[2]) / 8 : 4;
}
function textureBytes(descriptor: GPUTextureDescriptor): number {
  const size = descriptor.size as GPUExtent3DDict | number[];
  const [width, height = 1, depth = 1] = Array.isArray(size) ? size : [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];
  const volume = descriptor.dimension === "3d";
  let texels = 0;
  for (let mip = 0; mip < (descriptor.mipLevelCount ?? 1); mip++)
    texels += Math.max(1, width >> mip) * Math.max(1, height >> mip) * (volume ? Math.max(1, depth >> mip) : depth);
  return texels * texelBytes(descriptor.format) * (descriptor.sampleCount ?? 1);
}

/**
 * The device the product is handed: `raw`, with every call the probe counts
 * routed through `recorder`. Each call is forwarded with exactly the arguments
 * it was made with, so the native device sees what the product passed.
 */
function recordingDevice(raw: GPUDevice, recorder: Recorder): GPUDevice {
  const createBuffer = (descriptor: GPUBufferDescriptor) => {
    recorder.created("buffer", descriptor.label, `${descriptor.size} B`);
    const buffer = raw.createBuffer(descriptor);
    const memory = recorder.bufferMemory, label = descriptor.label ?? "(unlabelled)";
    memory.liveBytes += descriptor.size;
    memory.peakBytes = Math.max(memory.peakBytes, memory.liveBytes);
    memory.byLabel[label] = (memory.byLabel[label] ?? 0) + descriptor.size;
    const destroy = buffer.destroy.bind(buffer);
    let live = true;
    buffer.destroy = () => {
      if (live) { live = false; memory.liveBytes -= descriptor.size; memory.byLabel[label] -= descriptor.size; }
      destroy();
    };
    return buffer;
  };
  const forward = <T extends object>(owner: T, method: unknown, args: unknown[]) => Reflect.apply(method as Method, owner, args) as unknown;
  const textureDetail = (descriptor: GPUTextureDescriptor) => {
    const size = descriptor.size as GPUExtent3DDict | number[];
    const extent = Array.isArray(size) ? size : [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];
    return `${extent.join("x")} ${descriptor.format}`;
  };
  const create = (kind: string, method: unknown, detail?: (descriptor: never) => string) => (...args: unknown[]) => {
    const descriptor = args[0] as { label?: string } | undefined;
    recorder.created(kind, descriptor?.label, descriptor && detail ? detail(descriptor as never) : "");
    return forward(raw, method, args);
  };
  const pipeline = (kind: string, method: unknown) => (...args: unknown[]) => {
    const descriptor = (args[0] ?? {}) as { label?: string; compute?: { entryPoint?: string; module?: GPUShaderModule } };
    recorder.created(kind, descriptor.label);
    const made = forward(raw, method, args) as object | Promise<object>;
    const started = performance.now();
    const label = `${descriptor.label || descriptor.compute?.module?.label || "(unlabelled)"}${descriptor.compute?.entryPoint ? ` > ${descriptor.compute.entryPoint}` : ""}`;
    // Report a stuck native compile while it is stuck, not only after startup
    // succeeds. This is opt-in because ordinary edit reports stay compact.
    const slow = process.env.FLUID_GPU_COMPILE_TRACE === "1"
      ? setTimeout(() => console.error(`Still compiling after 5s: ${label}`), 5000) : undefined;
    slow?.unref();
    const name = (value: object) => { recorder.namePipeline(value, descriptor); return value; };
    const compiled = (value: object) => {
      recorder.compiles.push({ label, ms: performance.now() - started });
      return name(value);
    };
    if (typeof (made as Promise<object>).then === "function") return (made as Promise<object>).then(compiled).finally(() => clearTimeout(slow));
    clearTimeout(slow);
    return name(made);
  };
  const queue = overlay(raw.queue, {
    writeBuffer(...args: unknown[]) {
      const [buffer, , data, dataOffset, size] = args as [GPUBuffer, number, ArrayBufferLike | ArrayBufferView, number?, number?];
      recorder.upload(buffer?.label, writtenBytes(data, dataOffset, size));
      return forward(raw.queue, raw.queue.writeBuffer, args);
    },
    writeTexture(...args: unknown[]) {
      const [destination, data, layout] = args as [GPUTexelCopyTextureInfo, ArrayBufferLike | ArrayBufferView, GPUTexelCopyBufferLayout?];
      recorder.upload(destination?.texture?.label, data.byteLength - (layout?.offset ?? 0));
      return forward(raw.queue, raw.queue.writeTexture, args);
    },
    submit(...args: unknown[]) { recorder.gpu.submits += 1; return forward(raw.queue, raw.queue.submit, args); },
  });
  const computePass = (pass: GPUComputePassEncoder, label: string | undefined) => {
    let bound: GPUComputePipeline | undefined;
    return overlay(pass, {
      setPipeline(...args: unknown[]) { bound = args[0] as GPUComputePipeline; return forward(pass, pass.setPipeline, args); },
      dispatchWorkgroups(...args: unknown[]) {
        const [x, y = 1, z = 1] = args as number[];
        recorder.dispatch(label, bound, false, x! * y * z);
        return forward(pass, pass.dispatchWorkgroups, args);
      },
      dispatchWorkgroupsIndirect(...args: unknown[]) {
        recorder.dispatch(label, bound, true, 0);
        return forward(pass, pass.dispatchWorkgroupsIndirect, args);
      },
    });
  };
  const bundleDraws = new WeakMap<GPURenderBundle, { direct: number; indirect: number }>();
  const renderBundle = (pass: GPURenderBundleEncoder) => {
    const tally = { direct: 0, indirect: 0 };
    const drawn = (counter: "direct" | "indirect", method: unknown) => (...args: unknown[]) => {
      tally[counter] += 1; return forward(pass, method, args);
    };
    return overlay(pass, {
      draw: drawn("direct", pass.draw), drawIndexed: drawn("direct", pass.drawIndexed),
      drawIndirect: drawn("indirect", pass.drawIndirect), drawIndexedIndirect: drawn("indirect", pass.drawIndexedIndirect),
      finish(...args: unknown[]) {
        const bundle = forward(pass, pass.finish, args) as GPURenderBundle;
        bundleDraws.set(bundle, tally); return bundle;
      },
    });
  };
  const renderPass = (pass: GPURenderPassEncoder) => {
    const drawn = (counter: "directDraws" | "indirectDraws", method: unknown) => (...args: unknown[]) => { recorder.gpu[counter] += 1; return forward(pass, method, args); };
    return overlay(pass, {
      draw: drawn("directDraws", pass.draw), drawIndexed: drawn("directDraws", pass.drawIndexed),
      drawIndirect: drawn("indirectDraws", pass.drawIndirect), drawIndexedIndirect: drawn("indirectDraws", pass.drawIndexedIndirect),
      executeBundles(bundles: Iterable<GPURenderBundle>) {
        const list = [...bundles];
        for (const bundle of list) {
          const tally = bundleDraws.get(bundle);
          recorder.gpu.directDraws += tally?.direct ?? 0;
          recorder.gpu.indirectDraws += tally?.indirect ?? 0;
        }
        return pass.executeBundles(list);
      },
    });
  };
  const encoder = (inner: GPUCommandEncoder) => {
    const counted = (counter: "copies" | "clears", method: unknown) => (...args: unknown[]) => { recorder.gpu[counter] += 1; return forward(inner, method, args); };
    return overlay(inner, {
      beginComputePass: (...args: unknown[]) => computePass(forward(inner, inner.beginComputePass, [recorder.timed(args[0] as GPUComputePassDescriptor | undefined)]) as GPUComputePassEncoder,
        (args[0] as GPUComputePassDescriptor | undefined)?.label),
      beginRenderPass: (...args: unknown[]) => { recorder.gpu.renderPasses += 1; return renderPass(forward(inner, inner.beginRenderPass, [recorder.timed(args[0] as GPURenderPassDescriptor)]) as GPURenderPassEncoder); },
      clearBuffer: counted("clears", inner.clearBuffer),
      copyBufferToBuffer: counted("copies", inner.copyBufferToBuffer),
      copyBufferToTexture: counted("copies", inner.copyBufferToTexture),
      copyTextureToBuffer: counted("copies", inner.copyTextureToBuffer),
      copyTextureToTexture: counted("copies", inner.copyTextureToTexture),
    });
  };
  return overlay(raw, {
    queue,
    createBuffer,
    createTexture: (descriptor: GPUTextureDescriptor) => {
      recorder.created("texture", descriptor.label, textureDetail(descriptor));
      const texture = raw.createTexture(descriptor);
      const memory = recorder.textureMemory, label = descriptor.label ?? "(unlabelled)", size = textureBytes(descriptor);
      memory.liveBytes += size;
      memory.peakBytes = Math.max(memory.peakBytes, memory.liveBytes);
      memory.byLabel[label] = (memory.byLabel[label] ?? 0) + size;
      const destroy = texture.destroy.bind(texture);
      let live = true;
      texture.destroy = () => {
        if (live) { live = false; memory.liveBytes -= size; memory.byLabel[label] -= size; }
        destroy();
      };
      return texture;
    },
    createBindGroup: create("bind group", raw.createBindGroup),
    createBindGroupLayout: create("bind group layout", raw.createBindGroupLayout),
    createPipelineLayout: create("pipeline layout", raw.createPipelineLayout),
    createSampler: create("sampler", raw.createSampler),
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => {
      let code = descriptor.code;
      if (recorder.referenceSampleCache) {
        const cached = meshVolumeWGSL("sceneFieldProgramBlockVolume", "loadArenaWord", true);
        if (code.includes(cached)) {
          code = code.replace(cached, meshVolumeWGSL("sceneFieldProgramBlockVolume", "loadArenaWord", false));
          recorder.sampleCacheReplacements++;
        }
      }
      recorder.created("shader module", descriptor.label);
      return raw.createShaderModule({ ...descriptor, code });
    },
    createQuerySet: create("query set", raw.createQuerySet),
    createComputePipeline: pipeline("compute pipeline", raw.createComputePipeline),
    createComputePipelineAsync: pipeline("compute pipeline", raw.createComputePipelineAsync),
    createRenderPipeline: pipeline("render pipeline", raw.createRenderPipeline),
    createRenderPipelineAsync: pipeline("render pipeline", raw.createRenderPipelineAsync),
    createCommandEncoder: (...args: unknown[]) => encoder(forward(raw, raw.createCommandEncoder, args) as GPUCommandEncoder),
    createRenderBundleEncoder: (...args: unknown[]) => renderBundle(forward(raw, raw.createRenderBundleEncoder, args) as GPURenderBundleEncoder),
  });
}

// ---------------------------------------------------------------------------
// The stand-in device
// ---------------------------------------------------------------------------

/**
 * A GPUDevice that allocates and executes nothing.
 *
 * Its limits and features are those of an Apple-silicon Metal adapter, because
 * the product sizes its arenas and picks its formats from them. A mapped range
 * reads back zero, so anything the product learns by readback stays at its
 * initial value: the report says "not reported" for those, and the world build
 * is switched to the selection that needs no readback (`selectBricksOnTheHost`).
 */
function createFakeGpu(): GPU {
  const anything = (): unknown => new Proxy(function () {}, {
    get: (_target, key) => key === "then" ? undefined : anything(),
    apply: () => anything(),
  });
  const pipeline = (descriptor: { label?: string }) => ({ label: descriptor?.label, getBindGroupLayout: () => ({}) });
  const limits = {
    maxTextureDimension1D: 16_384, maxTextureDimension2D: 16_384, maxTextureDimension3D: 2_048, maxTextureArrayLayers: 2_048,
    maxBindGroups: 4, maxBindingsPerBindGroup: 1_000, maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 16,
    maxStorageBuffersPerShaderStage: 10, maxStorageTexturesPerShaderStage: 8, maxUniformBuffersPerShaderStage: 12,
    maxUniformBufferBindingSize: 65_536, maxStorageBufferBindingSize: 4_294_967_292, maxBufferSize: 4_294_967_296,
    minUniformBufferOffsetAlignment: 256, minStorageBufferOffsetAlignment: 32, maxVertexBuffers: 8, maxVertexAttributes: 30,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 128, maxComputeWorkgroupStorageSize: 32_768,
    maxComputeInvocationsPerWorkgroup: 1_024, maxComputeWorkgroupSizeX: 1_024, maxComputeWorkgroupSizeY: 1_024,
    maxComputeWorkgroupSizeZ: 1_024, maxComputeWorkgroupsPerDimension: 65_535,
  };
  const features = new Set(["shader-f16", "subgroups", "texture-formats-tier1"]);
  const info = { vendor: "probe", architecture: "stand-in", device: "", description: "recording stand-in device", subgroupMinSize: 4, subgroupMaxSize: 64 };
  const device = {
    label: "probe stand-in device", limits, features, adapterInfo: info,
    queue: { writeBuffer() {}, writeTexture() {}, submit() {}, onSubmittedWorkDone: () => Promise.resolve(), copyExternalImageToTexture() {} },
    createBuffer: (descriptor: GPUBufferDescriptor) => ({
      label: descriptor.label ?? "", size: descriptor.size, usage: descriptor.usage, mapState: "unmapped", destroy() {},
      getMappedRange: (_offset?: number, size?: number) => new ArrayBuffer(size ?? descriptor.size),
      unmap() {}, mapAsync: () => Promise.resolve(),
    }),
    createTexture: (descriptor: GPUTextureDescriptor) => {
      const size = descriptor.size as GPUExtent3DDict | number[];
      const extent = Array.isArray(size) ? size : [size.width, size.height ?? 1, size.depthOrArrayLayers ?? 1];
      return { label: descriptor.label ?? "", width: extent[0] ?? 1, height: extent[1] ?? 1, depthOrArrayLayers: extent[2] ?? 1,
        format: descriptor.format, dimension: descriptor.dimension ?? "2d", usage: descriptor.usage,
        mipLevelCount: descriptor.mipLevelCount ?? 1, sampleCount: descriptor.sampleCount ?? 1,
        createView: () => ({}), destroy() {} };
    },
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}), createBindGroup: () => ({}), createSampler: () => ({}),
    createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({ label: descriptor?.label, getCompilationInfo: () => Promise.resolve({ messages: [] }) }),
    createQuerySet: () => ({ destroy() {} }),
    createCommandEncoder: () => anything(),
    createRenderBundleEncoder: () => anything(),
    createComputePipeline: pipeline, createRenderPipeline: pipeline,
    createComputePipelineAsync: (descriptor: { label?: string }) => Promise.resolve(pipeline(descriptor)),
    createRenderPipelineAsync: (descriptor: { label?: string }) => Promise.resolve(pipeline(descriptor)),
    pushErrorScope() {}, popErrorScope: () => Promise.resolve(null),
    addEventListener() {}, removeEventListener() {}, destroy() {},
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
  const adapter = {
    features, limits, info,
    requestDevice: () => Promise.resolve(device),
  };
  const globals = globalThis as unknown as Record<string, unknown>;
  globals.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
  globals.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
  globals.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  globals.GPUMapMode ??= { READ: 1, WRITE: 2 };
  return { requestAdapter: () => Promise.resolve(adapter), getPreferredCanvasFormat: () => "bgra8unorm" } as unknown as GPU;
}

// ---------------------------------------------------------------------------
// The browser around the client and its worker
// ---------------------------------------------------------------------------

const SWAP_FORMAT: GPUTextureFormat = "bgra8unorm";

interface FrameImage { width: number; height: number; pixels: Uint8Array }

function saveCanvasPng(path: string, frame: FrameImage): void {
  const rgb = new Uint8Array(frame.width * frame.height * 3);
  for (let i = 0; i < frame.width * frame.height; i++) {
    rgb[3 * i] = frame.pixels[4 * i + 2]!;
    rgb[3 * i + 1] = frame.pixels[4 * i + 1]!;
    rgb[3 * i + 2] = frame.pixels[4 * i]!;
  }
  writeFileSync(path, encodeRgbPng({ width: frame.width, height: frame.height, rgb }));
}

/** What crossed between the client and its worker, and what handling it cost. */
interface Traffic {
  /** Message types the client posted, in order. */
  sent: string[];
  /** Time structured-cloning messages, by type; the worker's answers are marked "(answer)". */
  clone_ms: Record<string, number>;
  /** Time in the worker's message handler before it first yields, by message type. */
  handler_ms: Record<string, number>;
}

function addTo(totals: Record<string, number>, key: string, value: number): void {
  totals[key] = (totals[key] ?? 0) + value;
}

type FrameResponse = Extract<WebGPURenderWorkerResponse, { type: "frame" }>;
interface BuildRecord { kind: string; ms: number; label: string }
type MessageListener = (event: { data: unknown }) => void;

/**
 * Everything `WebGPURenderWorkerClient` and `webgpu-render-worker.ts` expect
 * of a browser, in one process: the `Worker` the client constructs, the
 * worker's global scope, the canvas, and `navigator.gpu`.
 *
 * Messages cross as they do between threads — structured-cloned when posted,
 * delivered on a later turn of the event loop, in order — and the time spent
 * cloning them is kept, because it is part of what an edit costs.
 */
class BrowserHost {
  readonly statuses: GPUStatus[] = [];
  readonly builds: BuildRecord[] = [];
  readonly validationErrors: string[] = [];
  gpuInfo?: Extract<WebGPURenderWorkerResponse, { type: "gpu-info" }>["info"];
  readonly advanceCompletions_s: number[] = [];
  readonly simulationErrors: string[] = [];
  effectiveStatus?: EffectiveRendererStatus;
  failure?: string;
  /** The device the renderer was granted, without the recording layer. */
  rawDevice?: GPUDevice;
  private traffic: Traffic = { sent: [], clone_ms: {}, handler_ms: {} };
  private buildStarted_ms?: number;
  private buildKind = "";
  private readonly scope = new EventTarget();
  private readonly clientListeners = new Set<MessageListener>();
  private readonly waiters = new Set<{ test(message: WebGPURenderWorkerResponse): boolean; resolve(message: WebGPURenderWorkerResponse): void; reject(error: Error): void }>();
  private workerLoaded?: Promise<unknown>;
  private swapTexture?: GPUTexture;
  private readback?: GPUBuffer;
  private readonly offscreen: { width: number; height: number; getContext(kind: string): unknown };
  /** The canvas element the client is constructed over. */
  readonly canvas: HTMLCanvasElement;

  /**
   * `gpu` is asked for an adapter and a device by the renderer itself, with the
   * features and limits it asks a browser for; `record` puts the recording
   * layer in front of the device it is granted.
   */
  constructor(gpu: GPU, private readonly real: boolean, viewport: { width: number; height: number }, record: (raw: GPUDevice) => GPUDevice) {
    const host = this;
    this.offscreen = {
      width: 1, height: 1,
      getContext: (kind: string) => kind !== "webgpu" ? null : { configure() {}, unconfigure() {}, getCurrentTexture: () => host.currentTexture() },
    };
    this.canvas = {
      transferControlToOffscreen: () => this.offscreen,
      getBoundingClientRect: () => ({ width: viewport.width, height: viewport.height }),
    } as unknown as HTMLCanvasElement;
    const globals = globalThis as unknown as Record<string, unknown>;
    // The render worker module addresses its realm as `self`, and the
    // compilation manager admits a device only inside a WorkerGlobalScope.
    // This process is that worker too: it has no UI thread to protect.
    const scope = overlay(globalThis as unknown as Record<string, unknown>, {
      addEventListener: this.scope.addEventListener.bind(this.scope),
      removeEventListener: this.scope.removeEventListener.bind(this.scope),
      postMessage: (message: WebGPURenderWorkerResponse) => this.fromWorker(message),
    });
    Object.defineProperty(globalThis, "self", { value: scope, configurable: true, writable: true });
    globals.WorkerGlobalScope = class { static [Symbol.hasInstance](value: unknown) { return value === globalThis; } };
    globals.requestAnimationFrame ??= (callback: (time: number) => void) => setTimeout(() => callback(performance.now()), 0);
    globals.cancelAnimationFrame ??= (handle: ReturnType<typeof setTimeout>) => clearTimeout(handle);
    globals.Worker = class {
      constructor() { host.workerLoaded = import("../lib/core/webgpu-render-worker"); }
      addEventListener(type: string, listener: MessageListener) { if (type === "message") host.clientListeners.add(listener); }
      removeEventListener(_type: string, listener: MessageListener) { host.clientListeners.delete(listener); }
      postMessage(message: WebGPURenderWorkerRequest) { host.toWorker(message); }
      terminate() {}
    };
    const navigatorGpu = overlay(gpu, {
      getPreferredCanvasFormat: () => SWAP_FORMAT,
      requestAdapter: async (options?: GPURequestAdapterOptions) => {
        const adapter = await gpu.requestAdapter(options);
        return adapter && overlay(adapter, { requestDevice: async (descriptor?: GPUDeviceDescriptor) => {
          const raw = await adapter.requestDevice(descriptor);
          raw.addEventListener("uncapturederror", (event) => this.validationErrors.push((event as GPUUncapturedErrorEvent).error.message));
          this.rawDevice = raw;
          return record(raw);
        } });
      },
    });
    if (typeof navigator === "undefined") Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
    Object.defineProperty(navigator, "gpu", { value: navigatorGpu, configurable: true });
  }

  private currentTexture(): GPUTexture {
    const { width, height } = this.offscreen;
    if (!this.swapTexture || this.swapTexture.width !== width || this.swapTexture.height !== height) {
      this.swapTexture?.destroy();
      // Created on the raw device: the canvas belongs to the browser, not to the edit path.
      this.swapTexture = this.rawDevice!.createTexture({ label: "Probe canvas", size: [width, height], format: SWAP_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
    }
    return this.swapTexture;
  }

  /** `Worker.postMessage`: cloned now, handled by the worker on a later turn. The canvas is transferred, not cloned. */
  private toWorker(message: WebGPURenderWorkerRequest): void {
    this.traffic.sent.push(message.type);
    const started = performance.now();
    const delivered = message.type === "attach" ? message : structuredClone(message);
    addTo(this.traffic.clone_ms, message.type, performance.now() - started);
    void this.workerLoaded!.then(() => setImmediate(() => {
      const event = new Event("message") as Event & { data: WebGPURenderWorkerRequest };
      event.data = delivered;
      const handling = performance.now();
      this.scope.dispatchEvent(event);
      addTo(this.traffic.handler_ms, message.type, performance.now() - handling);
    }));
  }

  /** The worker's `postMessage`: cloned now, handled by the client on a later turn. */
  private fromWorker(message: WebGPURenderWorkerResponse): void {
    const started = performance.now();
    const delivered = structuredClone(message);
    addTo(this.traffic.clone_ms, `${message.type} (answer)`, performance.now() - started);
    setImmediate(() => {
      this.observe(delivered);
      for (const listener of [...this.clientListeners]) listener({ data: delivered });
      for (const waiter of [...this.waiters]) {
        if (!waiter.test(delivered)) continue;
        this.waiters.delete(waiter);
        waiter.resolve(delivered);
      }
    });
  }

  private observe(message: WebGPURenderWorkerResponse): void {
    if (message.type === "gpu-info") {
      this.gpuInfo = message.info;
      for (const failure of [message.info.simulationPipelineError, message.info.topologyGenerationError,
        message.info.simulationFailure && JSON.stringify(message.info.simulationFailure)]) {
        if (failure && !this.simulationErrors.includes(failure)) this.simulationErrors.push(failure);
      }
    }
    if (message.type === "advance-completed") this.advanceCompletions_s.push(message.time_s);
    if (message.type === "effective-renderer-status") this.effectiveStatus = message.status;
    // Renderer-only worlds announce source readiness before compiling their
    // presentation. That optional compilation need not emit another lifecycle
    // "ready" status, so use the actual submitted, active presentation as its
    // completion receipt. Otherwise the probe can keep calling settled frames
    // a build forever. Native mesh convergence is still checked in settle().
    if (message.type === "frame" && message.metrics.presentationSubmitted
      && this.effectiveStatus?.state === "active" && this.buildStarted_ms !== undefined) {
      this.builds.push({ kind: this.buildKind, ms: performance.now() - this.buildStarted_ms,
        label: "Active sparse presentation submitted" });
      this.buildStarted_ms = undefined;
    }
    if (message.type !== "status") return;
    const status = message.status;
    const previous = this.statuses.at(-1);
    if (process.env.FLUID_EDIT_PROBE_PROGRESS === "1"
      && (previous?.state !== status.state || previous.label !== status.label)) {
      console.error(`probe status ${status.state}: ${status.label}`);
    }
    this.statuses.push(status);
    // Only a solver or scene build carries a kind; renderer start-up does not.
    const kind = (status as { kind?: string }).kind;
    if (status.state === "initializing" && kind && this.buildStarted_ms === undefined) {
      this.buildStarted_ms = performance.now();
      this.buildKind = kind;
    } else if (status.state === "ready" && this.buildStarted_ms !== undefined) {
      this.builds.push({ kind: this.buildKind, ms: performance.now() - this.buildStarted_ms, label: status.label });
      this.buildStarted_ms = undefined;
    }
    if (status.state === "unavailable" || status.state === "lost" || status.state === "blocked") {
      this.failure ??= status.label;
      for (const waiter of this.waiters) waiter.reject(new Error(`The renderer stopped: ${status.label}`));
      this.waiters.clear();
    }
  }

  get buildInFlight(): boolean { return this.buildStarted_ms !== undefined; }

  /** The size the renderer gave the canvas it draws to. */
  get canvasSize(): string { return `${this.offscreen.width}x${this.offscreen.height}`; }

  /** The next answer from the worker that passes `test`, once the client has handled it. */
  waitFor<T extends WebGPURenderWorkerResponse = WebGPURenderWorkerResponse>(test: (message: WebGPURenderWorkerResponse) => boolean): Promise<T> {
    if (this.failure) return Promise.reject(new Error(`The renderer stopped: ${this.failure}`));
    return new Promise<T>((resolve, reject) => {
      this.waiters.add({ test, resolve: (message) => resolve(message as T), reject });
    });
  }

  /** What crossed since the last call. */
  takeTraffic(): Traffic {
    const taken = this.traffic;
    this.traffic = { sent: [], clone_ms: {}, handler_ms: {} };
    return taken;
  }

  /** Let promise continuations and timers the frame left behind run before the next one. */
  async idle(): Promise<void> {
    if (this.real) await this.rawDevice?.queue.onSubmittedWorkDone();
    await new Promise<void>((resolve) => setImmediate(resolve));
  }

  /** The canvas as presented, or nothing on the stand-in device. */
  async readCanvas(): Promise<FrameImage | undefined> {
    const device = this.rawDevice;
    if (!this.real || !this.swapTexture || !device) return undefined;
    const { width, height } = this.swapTexture;
    const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
    if (!this.readback || this.readback.size !== bytesPerRow * height) {
      this.readback?.destroy();
      this.readback = device.createBuffer({ label: "Probe canvas readback", size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const encoder = device.createCommandEncoder({ label: "Probe canvas readback" });
    encoder.copyTextureToBuffer({ texture: this.swapTexture }, { buffer: this.readback, bytesPerRow, rowsPerImage: height }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(this.readback.getMappedRange());
    const pixels = new Uint8Array(width * height * 4);
    for (let row = 0; row < height; row += 1) pixels.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + width * 4), row * width * 4);
    this.readback.unmap();
    return { width, height, pixels };
  }
}

// ---------------------------------------------------------------------------
// Optional internals
// ---------------------------------------------------------------------------

/** An export that may have been renamed or removed since this was written. */
async function optionalExport(path: string, name: string): Promise<unknown> {
  try {
    const module = await import(new URL(path, import.meta.url).href) as Record<string, unknown>;
    return module[name];
  } catch {
    return undefined;
  }
}

const LIVE_SCENE_PATH = "../lib/svo/features/scene-publication/webgpu-live-svo-scene.ts";
const STAGES: readonly { owner: string; path: string; methods: readonly string[] }[] = [
  { owner: "SceneInteractionHost", path: "../lib/interaction/runtime/host.ts", methods: ["validateSolidEdit", "replaceScene"] },
  { owner: "Box3DWorld", path: "../lib/interaction/adapters/box3d/world.ts", methods: ["validateSolidEdit", "updateSolids"] },
  { owner: "FluidLabRenderer", path: "../lib/core/webgpu-renderer.ts",
    methods: ["acceptLiveSolidEdit", "validateLiveSolidEdit", "previewLiveSolidEdit", "publishRenderScene", "draw"] },
  { owner: "WebGPULiveSvoScene", path: LIVE_SCENE_PATH,
    methods: ["validateLiveSolidEdit", "stageSceneUpdate", "stageLivePrimitiveUpdates", "encodeSceneMaintenance"] },
  { owner: "OctreeSparseBrickWorld", path: "../lib/svo/features/construction/webgpu-svo-sparse-bricks.ts",
    methods: ["validateLiveSolidEdit", "stageSceneUpdate", "planLiveDerivedGrowth", "growLiveDerivedAddressPlan",
      "assertLiveTopologyCapacity", "terrainResamplingBounds", "encodeSceneMaintenance",
      "liveBricksToActivate", "livePrimitiveBoundsNear", "reclaimableTopologyBricks", "renderResidualFor", "solidChangesBetween",
      "stagePrimitivePublication", "releaseUnclaimedTopologyBricks", "listLivePrimitives", "stageAppearanceTables", "raiseBackdropContentCeiling"] },
  { owner: "SparseSceneProxyVoxelizer", path: "../lib/core/webgpu-sparse-scene-proxies.ts",
    methods: ["validateSolidWorld", "setSolidWorld", "publish", "encodeMaintenance"] },
  { owner: "SparseVoxelDrySceneRenderer", path: "../lib/svo/pipeline/webgpu-svo-dry-scene.ts",
    methods: ["publishScene", "encode"] },
];
const MAINTENANCE_STAGE = "WebGPULiveSvoScene.encodeSceneMaintenance";
const VOXELIZATION_STAGE = "SparseSceneProxyVoxelizer.encodeMaintenance";

/** Read a property that may not exist, or whose getter may throw before the owner is ready. */
function peek(owner: unknown, ...path: string[]): unknown {
  let value = owner;
  for (const key of path) {
    if (value === null || typeof value !== "object") return undefined;
    try { value = (value as Record<string, unknown>)[key]; } catch { return undefined; }
  }
  return value;
}

// ---------------------------------------------------------------------------
// Scenes, sites and edits
// ---------------------------------------------------------------------------

function createScene(id: string): { scene: SceneDescription; camera: CameraState; background: string; presentationMode: string } {
  if (id === "hall") {
    return { scene: createEmptyScene({ sceneId: "hall", extents_m: { x: 3.2, y: 1.6, z: 3.2 } }),
      camera: { ...defaultCamera }, background: "default", presentationMode: "full-scene" };
  }
  const preset = getScenePreset(id);
  if (preset.id !== id) throw new Error(`No scene preset is called "${id}".`);
  return { scene: preset.create(), camera: cameraForPreset(preset), background: preset.background, presentationMode: preset.presentationMode };
}

type Cell = [number, number, number];
interface SolidEdit { patches: SolidWorldVoxelPatch[]; note: string; bounds: { minimum: Cell; maximumExclusive: Cell } }

/** A reproducible pointer sweep on actual imported scenery, not fabricated patches. */
function planPenStroke(scene: SceneDescription, site: readonly [number, number], carve: boolean, diameter = 3, requestedNode?: string, samples = 16) {
  const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
  const owner = new Map<number, string>();
  for (const span of catalog.spans) for (let i = span.from; i < span.to; i++) owner.set(i, span.nodeId);
  const candidates = catalog.primitives.flatMap((primitive, index) => {
    if (primitive.kind !== "field-program" || !primitive.program.sampledVolume || primitive.tags.includes("shell")) return [];
    if (requestedNode && !owner.get(index)?.includes(requestedNode) && !primitive.program.sampledVolume.asset.includes(requestedNode)) return [];
    const box = primitive.aabb_m;
    const x = .5 * (box.min.x + box.max.x), z = .5 * (box.min.z + box.max.z);
    return [{ primitive, nodeId: owner.get(index), distance: Math.hypot(x - site[0], z - site[1]),
      preferred: /paver|lounger/.test(primitive.program.sampledVolume.asset) }];
  }).sort((a, b) => Number(b.preferred) - Number(a.preferred) || a.distance - b.distance);
  const values = toolValues(penTool, { size: diameter, mode: carve ? 1 : 0 });
  const cells = sceneCellSizes_m(scene);
  for (const candidate of candidates.slice(0, 64)) {
    if (!candidate.nodeId) continue;
    const box = candidate.primitive.aabb_m;
    const axis = box.max.x - box.min.x >= box.max.z - box.min.z ? "x" : "z";
    const extent = box.max[axis] - box.min[axis];
    const width = Math.min(extent * .6, (samples - 1) * cells[axis === "x" ? 0 : 2]);
    const rays: EditorRay[] = Array.from({ length: samples }, (_, i) => ({ origin: {
      x: .5 * (box.min.x + box.max.x) + (axis === "x" ? (i / (samples - 1) - .5) * width : 0),
      y: box.max.y + .25,
      z: .5 * (box.min.z + box.max.z) + (axis === "z" ? (i / (samples - 1) - .5) * width : 0),
    }, direction: { x: 0, y: -1, z: 0 } }));
    if (!rays.every(ray => scenerySurfaceAt(scene, ray)?.nodeId === candidate.nodeId)) continue;
    const gesture = penTool.begin({ scene, ray: rays[0]!, values });
    if (!gesture) continue;
    let patches: readonly SolidWorldVoxelPatch[] = [];
    for (const ray of rays) patches = gesture.update(ray)?.patches ?? patches;
    if (!patches.length) continue;
    const bounds = {
      minimum: [0, 1, 2].map(a => Math.min(...patches.map(p => p.minimum[a]!))) as Cell,
      maximumExclusive: [0, 1, 2].map(a => Math.max(...patches.map(p => p.maximumExclusive[a]!))) as Cell,
    };
    return { rays, values, nodeId: candidate.nodeId, solid: { patches: [...patches], bounds,
      note: `${samples} live Pen ${carve ? "Carve" : "Draw"} samples on ${candidate.nodeId}, diameter ${diameter} voxels, then one undo` } };
  }
  return undefined;
}

/**
 * Where the solid edits land, in the signed patch lattice.
 *
 * `brickBase` is the world's brick grid origin in that lattice when the world
 * says where it is; without it the bricks are taken to start at the lattice
 * origin and `aligned` is false, so the report can say the placement of
 * `voxel-inside` and `slab` against brick boundaries was not checked.
 */
class EditSites {
  readonly cell: readonly [number, number, number];
  readonly origin: readonly [number, number, number];
  readonly brick: number;
  readonly brickBase: Cell;
  readonly aligned: boolean;

  constructor(private readonly scene: SceneDescription, private readonly site: readonly [number, number], world: unknown) {
    this.cell = sceneCellSizes_m(scene);
    this.origin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
    this.brick = scene.voxelDomain.brickSize_cells === 4 ? 4 : 8;
    const worldOrigin = peek(world, "sceneWorldOrigin"), worldCell = peek(world, "cellSize");
    const known = Array.isArray(worldOrigin) && Array.isArray(worldCell)
      && [0, 1, 2].every((axis) => Math.abs(worldCell[axis] - this.cell[axis]!) < 1e-9);
    this.aligned = known;
    this.brickBase = known
      ? [0, 1, 2].map((axis) => Math.round(((worldOrigin as number[])[axis]! - this.origin[axis]!) / this.cell[axis]!)) as Cell
      : [0, 0, 0];
  }

  private column(dx = 0, dz = 0): { x: number; z: number; air: number } {
    const x = Math.floor((this.site[0] - this.origin[0]) / this.cell[0]) + dx;
    const z = Math.floor((this.site[1] - this.origin[2]) / this.cell[2]) + dz;
    const height = terrainHeightAt(this.scene.terrain, this.origin[0] + (x + 0.5) * this.cell[0], this.origin[2] + (z + 0.5) * this.cell[2]);
    // The first cell with no ground in it; the one beneath holds the surface.
    return { x, z, air: Math.ceil(height / this.cell[1] - 1e-6) };
  }

  private local(cell: Cell): Cell {
    return cell.map((value, axis) => ((value - this.brickBase[axis]!) % this.brick + this.brick) % this.brick) as Cell;
  }

  private describe(cell: Cell): string {
    return `cell ${cell.join(",")}, brick-local ${this.local(cell).join(",")}${this.aligned ? "" : " (brick grid assumed)"}`;
  }

  private box(operation: SolidWorldVoxelPatch["operation"], minimum: Cell, size: Cell, kind?: string): SolidEdit {
    const maximumExclusive = minimum.map((value, axis) => value + size[axis]!) as Cell;
    const patch: SolidWorldVoxelPatch = { operation, minimum, maximumExclusive, ...(operation === "clear" ? {} : { materialId: 2 }), ...(kind ? { kind } : {}) };
    return { patches: [patch], note: "", bounds: { minimum, maximumExclusive } };
  }

  voxel(operation: "fill" | "clear"): SolidEdit {
    const column = this.column();
    const cell: Cell = [column.x, column.air, column.z];
    return { ...this.box(operation, cell, [1, 1, 1]), note: `${operation} ${this.describe(cell)}` };
  }

  /** A surface cell at least two cells from every face of its brick, so carving it cannot leave the brick. */
  voxelInside(): SolidEdit {
    const interior = (value: number) => value >= 2 && value <= this.brick - 3;
    let chosen: Cell | undefined, fallback: Cell | undefined;
    for (let radius = 2 * this.brick; radius <= 12 * this.brick && !chosen; radius += 1) {
      for (let step = -radius; step <= radius && !chosen; step += 1) {
        for (const [dx, dz] of [[step, -radius], [step, radius], [-radius, step], [radius, step]] as const) {
          const column = this.column(dx, dz);
          if (column.air < 1) continue;
          const cell: Cell = [column.x, column.air - 1, column.z];
          fallback ??= cell;
          if (this.local(cell).every(interior)) { chosen = cell; break; }
        }
      }
    }
    if (!chosen && !fallback) {
      // No ground in the lattice to carve (a bare floor): fill a brick-interior cell above it instead.
      const column = this.column(2 * this.brick, 2 * this.brick);
      const snap = (value: number, axis: number) => value - (((value - this.brickBase[axis]!) % this.brick) + this.brick) % this.brick + 2;
      const cell: Cell = [snap(column.x, 0), snap(column.air, 1), snap(column.z, 2)];
      return { ...this.box("fill", cell, [1, 1, 1]), note: `fill ${this.describe(cell)}; the site has no ground cell to carve` };
    }
    const cell = chosen ?? fallback!;
    return { ...this.box("clear", cell, [1, 1, 1]),
      note: `clear ${this.describe(cell)}${chosen ? "" : "; no surface cell near the site sits inside a brick, so this one may touch a neighbour"}` };
  }

  /** `start`, moved so `size` cells from it stay inside the lattice where the lattice is that large. */
  private inside(start: number, size: number, axis: number): number {
    return Math.max(0, Math.min(start, sceneLatticeDimensions(this.scene)[axis]! - size));
  }

  /** East of the site by its own width, so it lands on bricks the single-voxel edits never touched. */
  sphere(): SolidEdit {
    const column = this.column();
    const x = this.inside(column.x + 32, 64, 0), z = this.inside(column.z - 32, 64, 2);
    const centre = this.column(x + 32 - column.x, z + 32 - column.z);
    const minimum: Cell = [x, this.inside(centre.air - 8, 64, 1), z];
    return { ...this.box("fill", minimum, [64, 64, 64], "ellipsoid"), note: `ellipsoid 64 cells across, from ${this.describe(minimum)}` };
  }

  /** West of the site, one brick thick and brick-aligned, floating two bricks clear of the ground beneath it. */
  slab(): SolidEdit {
    const dimensions = sceneLatticeDimensions(this.scene);
    const span = (axis: 0 | 2) => Math.min(32 * this.brick, Math.floor((dimensions[axis] - 2 * this.brick) / this.brick) * this.brick);
    const size: Cell = [span(0), this.brick, span(2)];
    const column = this.column();
    const snap = (value: number, axis: number) => value - (((value - this.brickBase[axis]!) % this.brick) + this.brick) % this.brick;
    const x = snap(this.inside(column.x - 48 - size[0], size[0] + this.brick, 0) + this.brick, 0);
    const z = snap(this.inside(column.z - size[2] / 2, size[2] + this.brick, 2) + this.brick, 2);
    let air = column.air;
    for (let i = 0; i <= 16; i += 1) for (let j = 0; j <= 16; j += 1) {
      air = Math.max(air, this.column(x - column.x + Math.round(i * size[0] / 16), z - column.z + Math.round(j * size[2] / 16)).air);
    }
    const minimum: Cell = [x, snap(air + 2 * this.brick - 1, 1), z];
    const edit = this.box("fill", minimum, size);
    const sphere = this.sphere().bounds;
    const overlaps = [0, 1, 2].every((axis) => minimum[axis]! < sphere.maximumExclusive[axis]! && sphere.minimum[axis]! < edit.bounds.maximumExclusive[axis]!);
    return { ...edit, note: `${size[0] / this.brick} x 1 x ${size[2] / this.brick} bricks (${size.join(" x ")} cells), from ${this.describe(minimum)}`
      + (overlaps ? "; the lattice is too small to keep it clear of the sphere's bricks" : "") };
  }

  /** Bricks the edit's cells fall in, and the same with a one-brick halo. */
  dirtyBricks(bounds: SolidEdit["bounds"]): { dirty: number; withHalo: number } {
    let dirty = 1, withHalo = 1;
    for (let axis = 0; axis < 3; axis += 1) {
      const first = Math.floor((bounds.minimum[axis]! - this.brickBase[axis]!) / this.brick);
      const last = Math.floor((bounds.maximumExclusive[axis]! - 1 - this.brickBase[axis]!) / this.brick);
      dirty *= last - first + 1;
      withHalo *= last - first + 3;
    }
    return { dirty, withHalo };
  }

  /** Metre-space centre and bounding radius of an edit, for aiming the camera at it. */
  sphereOf(bounds: SolidEdit["bounds"]): { centre: [number, number, number]; radius: number } {
    const centre = [0, 1, 2].map((axis) => this.origin[axis]! + 0.5 * (bounds.minimum[axis]! + bounds.maximumExclusive[axis]!) * this.cell[axis]!) as [number, number, number];
    const radius = 0.5 * Math.hypot(...[0, 1, 2].map((axis) => (bounds.maximumExclusive[axis]! - bounds.minimum[axis]!) * this.cell[axis]!));
    return { centre, radius };
  }
}

type Mutable = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** The four property edits, each on a clone; `undefined` when the scene has nothing of that kind to change. */
const PROPERTY_EDITS: Partial<Record<EditName, (scene: Mutable) => string | undefined>> = {
  "palette-tint": (scene) => {
    const palettes = scene.scenery?.palettes as Record<string, Mutable> | undefined;
    const name = palettes && Object.keys(palettes).find((key) => Array.isArray(palettes[key]?.tint));
    if (!name) return undefined;
    palettes![name]!.tint[0] *= 0.9;
    return `scenery.palettes.${name}.tint[0] x 0.9`;
  },
  "node-material": (scene) => {
    const nodes = scene.scenery?.nodes as Mutable[] | undefined;
    const index = nodes?.findIndex((node) => node.material && typeof node.material === "object") ?? -1;
    if (index < 0) return undefined;
    const material = nodes![index]!.material as Mutable;
    const colour = material.colorLinear;
    if (Array.isArray(colour) && typeof colour[0] === "number") { colour[0] *= 0.9; return `scenery.nodes[${index}].material.colorLinear[0] x 0.9`; }
    const field = Object.keys(material).find((key) => typeof material[key] === "number");
    if (!field) return undefined;
    material[field] *= 0.9;
    return `scenery.nodes[${index}].material.${field} x 0.9`;
  },
  "fluid-optics": (scene) => {
    const absorption = scene.fluid?.optics?.absorption_mInv;
    if (!Array.isArray(absorption)) return undefined;
    absorption[0] *= 0.9;
    return "fluid.optics.absorption_mInv[0] x 0.9";
  },
  "material-model": (scene) => {
    const node = (scene.scenery?.nodes as Mutable[] | undefined)?.[0];
    if (!node) return undefined;
    node.materialModel = node.materialModel === "porcelain" ? "granite" : "porcelain";
    return `scenery.nodes[0].materialModel -> ${node.materialModel}`;
  },
};

/** The first scenery node, at any depth, the editor shows light controls for. */
function firstLightFixture(scene: SceneDescription): string | undefined {
  const pending = [...((scene.scenery?.nodes ?? []) as unknown as Mutable[])];
  for (let node = pending.shift(); node; node = pending.shift()) {
    if (typeof node.id === "string" && sceneryLightFixture(scene, node.id)) return node.id;
    if (Array.isArray(node.children)) pending.push(...node.children);
  }
  return undefined;
}

/** Edits made by the functions the editor's own controls call, so the document they produce is the one the app sends. */
const CONTROL_EDITS: Partial<Record<EditName, (scene: SceneDescription) => { next: SceneDescription; described: string } | undefined>> = {
  "lamp-colour": (scene) => {
    const id = firstLightFixture(scene);
    return id === undefined ? undefined
      : { next: withSceneryLightColor(scene, id, [1, 0.55, 0.3]), described: `the light colour control on scenery node ${id}` };
  },
  "lamp-strength": (scene) => {
    const id = firstLightFixture(scene);
    return id === undefined ? undefined
      : { next: withSceneryLightStrength(scene, id, sceneryLightFixture(scene, id)!.intensity * 0.5 + 1), described: `the light strength control on scenery node ${id}` };
  },
  "sun-colour": (scene) => {
    const [red, green, blue] = resolvedSun(scene).colorLinear;
    return { next: { ...scene, ...withSun(scene, { colorLinear: [red * 0.8, green, blue] }) }, described: "the sun colour control, red x 0.8" };
  },
  "sun-strength": (scene) => {
    const intensity = resolvedSun(scene).intensity;
    return { next: { ...scene, ...withSun(scene, { intensity: intensity * 0.8 }) }, described: `the sun strength control, ${intensity} x 0.8` };
  },
  "sun-direction": (scene) => {
    const { azimuth_deg, elevation_deg } = sunAngles(resolvedSun(scene).direction);
    return { next: { ...scene, ...withSun(scene, { direction: sunDirection(azimuth_deg + 20, elevation_deg) }) }, described: "the sun azimuth control, +20 degrees" };
  },
};

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

interface FrameRecord {
  /** Time on the thread from posting `draw` until its `frame` answer. */
  host_ms: number;
  /** Time inside the client's `draw` call itself. */
  caller_ms: number;
  traffic: Traffic;
  /** Until the GPU reported the frame's work done. Dawn only. */
  wall_ms?: number;
  /** GPU time inside each pass of the frame, by label. Dawn only. */
  gpuPasses_ms?: Record<string, number>;
  gpuTimeline?: GpuPassTimeline;
  submitted: boolean;
  gpu: GpuTally;
  stages: Record<string, StageTally>;
  maintenance?: boolean;
  voxelizationDispatches?: number;
  revision?: number;
  buildInFlight: boolean;
  /** The surface mesh says a build is under way. Only a real device ever answers it. */
  meshPending: boolean;
  image?: FrameImage;
}

interface BudgetRow { measure: string; value: string; budget: string; verdict: "pass" | "FAIL" | "not reported" | "" }

interface EditReport {
  edit: EditName;
  description: string;
  pen?: {
    nodeId: string;
    planning_ms: number;
    samples: { geometry_ms: number; acceptance_ms: number; update_ms: number; patches: number;
      published: boolean; frameHost_ms: number; frameWall_ms?: number; gpuPasses_ms?: Record<string, number>; gpuTimeline?: GpuPassTimeline;
      dirtyBuildPlan?: Record<string, number | boolean>;
      /** Inclusive host stages across acceptance and the immediately following frame. */
      stages: Record<string, StageTally> }[];
    publications: number;
    transactionFinishes: number;
    pipelineWarmup_ms?: number;
    images?: { baseline: string; edited: string; undo: string; changedPixels: number; undoChangedPixels: number };
    pipelineVariants?: { beforeWarmup: { cached: string[]; pending: string[] };
      before: { cached: string[]; pending: string[] }; after: { cached: string[]; pending: string[] } };
    undo?: { acceptance_ms: number; frames: number; restoredPatchCount: number; mesh?: Mutable };
  };
  refused?: string;
  skipped?: string;
  accept?: { reply_ms: number; gpu: GpuTally; stages: Record<string, StageTally> };
  /** Over the accept and the edit's frames. */
  traffic?: Traffic;
  /** Time the calling thread was held inside the client: the accept call and the first `draw`, clones included. */
  callerBlocked_ms?: { accept: number; firstDraw: number };
  /** What a frame with nothing to do encodes, for scale. */
  idleFrame?: { uploadBytes: number; directDispatches: number; indirectDispatches: number };
  frames: number;
  frameCapReached: boolean;
  firstFrameHost_ms?: number;
  idleFrameHost_ms?: number;
  hostFrames_ms: number[];
  wallFrames_ms?: number[];
  /** GPU time inside each pass, by label, for each frame. Dawn only. */
  gpuPassFrames_ms?: Record<string, number>[];
  editHost_ms?: number;
  uploadBytes?: { accept: number; framesOverIdle: number; byLabel: Record<string, number> };
  stages: Record<string, StageTally>;
  passes: Record<string, PassTally>;
  directDispatches: number;
  indirectDispatches: number;
  indirectPasses: string[];
  /** Render passes and the draws in them; indirect draws are counted, not budgeted. */
  draws: { renderPasses: number; direct: number; indirect: number };
  creations: Record<string, number>;
  created: string[];
  maintenanceFrames?: number;
  voxelizationFrames?: number;
  /** Times the published scene's revision word moved, sampled after the accept and after every frame. */
  revisions?: number;
  revisionSeries?: (number | undefined)[];
  /** Calls that publish the scene to the voxelizer and to the dry-scene renderer. */
  publications?: { voxelizer?: number; renderer?: number };
  dirtyBricks?: { dirty: number; withHalo: number };
  rebuildKeysChanged?: string[];
  uniformKeyChanged?: boolean;
  rebuilds: BuildRecord[];
  firstVisibleFrame?: number;
  convergedFrame?: number;
  /** Pixels two settled frames differ by with no edit between them. */
  noiseFloor?: { rect: number; frame: number };
  /** The edit's screen rectangle, in pixels. */
  rect?: { x0: number; y0: number; x1: number; y1: number };
  surfaceMesh?: { before: unknown; after: unknown };
  lighting?: unknown;
  budget: BudgetRow[];
}

function emptyReport(edit: EditName): EditReport {
  return { edit, description: "", frames: 0, frameCapReached: false, hostFrames_ms: [], stages: {}, passes: {},
    directDispatches: 0, indirectDispatches: 0, indirectPasses: [], draws: { renderPasses: 0, direct: 0, indirect: 0 },
    creations: {}, created: [], rebuilds: [], budget: [] };
}

const IDLE_FRAMES = 3;
const PIXEL_THRESHOLD = 3;
/** Share of the frame that may keep flickering in an image still called settled. */
const STILL_FRACTION = 0.001;
/** A real device is not asked for more frames than this to settle before an edit. */
const REAL_SETTLE_FRAMES = 120;

/** Pixels of `rect` whose colour moved by more than the threshold between two frames. */
function changedPixels(a: FrameImage, b: FrameImage, rect?: { x0: number; y0: number; x1: number; y1: number }): number {
  if (a.width !== b.width || a.height !== b.height) return a.width * a.height;
  const { x0, y0, x1, y1 } = rect ?? { x0: 0, y0: 0, x1: a.width, y1: a.height };
  let changed = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = x0; x < x1; x += 1) {
      const at = (y * a.width + x) * 4;
      if (Math.abs(a.pixels[at]! - b.pixels[at]!) > PIXEL_THRESHOLD || Math.abs(a.pixels[at + 1]! - b.pixels[at + 1]!) > PIXEL_THRESHOLD
        || Math.abs(a.pixels[at + 2]! - b.pixels[at + 2]!) > PIXEL_THRESHOLD) changed += 1;
    }
  }
  return changed;
}

const median = (values: readonly number[]) => values.length === 0 ? 0 : [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
const ms = (value: number | undefined) => value === undefined ? "not reported" : `${value.toFixed(value < 10 ? 2 : 1)} ms`;
const bytes = (value: number) => value < 1024 ? `${value} B` : value < 1_048_576 ? `${(value / 1024).toFixed(1)} KiB` : `${(value / 1_048_576).toFixed(2)} MiB`;

type DrawTail = Parameters<WebGPURenderWorkerClient["draw"]> extends [number, SceneDescription, CameraState, ...infer Rest] ? Rest : never;

class Probe {
  private readonly client: WebGPURenderWorkerClient;
  /** The document the viewport would be drawing: what `draw` is given every frame. */
  private document!: SceneDescription;
  private baseDocument!: SceneDescription;
  private camera: CameraState;
  private readonly baseCamera: CameraState;
  private sites!: EditSites;
  private keyFunctions: Record<string, (scene: SceneDescription) => string> = {};
  private drawTail!: DrawTail;
  private methodId = "";
  private runConfig: Mutable = {};
  private idle = { host_ms: 0, uploadBytes: 0, voxelizationDispatches: 0 };
  private voxelDocument?: SceneDescription;
  private phase = "";
  private penStrokeIndex = 0;
  private simulationTime_s = 0;
  private interactionHost?: import("../lib/interaction/runtime/host").SceneInteractionHost;
  private readonly externalCompletions_s: number[] = [];
  readonly simulation = {
    mode: "advancing fluid with fixed rigid-body poses",
    bodyIds: [] as string[],
    methodConfig: undefined as Mutable | undefined,
    checkpoints: [] as { phase: string; requestedSteps: number; completedReceipts: number; before_s: number; after_s: number }[],
    error: undefined as string | undefined,
  };
  /** How the world build chose its bricks. */
  brickSelection = "the GPU's selection, read back";
  worldBuild?: { ms: number; frames: number; builds: BuildRecord[]; creations: Record<string, number>; uploadBytes: number; settleFrames: number };

  constructor(private readonly options: Options, private readonly recorder: Recorder, private readonly host: BrowserHost,
    private readonly authored: ReturnType<typeof createScene>) {
    this.baseCamera = authored.camera;
    this.camera = authored.camera;
    this.client = new WebGPURenderWorkerClient(host.canvas, { onStatus() {} });
  }

  /** Hand `next` to the client as the edit of the drawn document, and draw it from the next frame on. */
  private async accept(next: SceneDescription): Promise<number> {
    const started = performance.now();
    this.interactionHost?.validateSolidEdit(next);
    const answered = this.client.validateLiveSolidEdit(next, this.document);
    const caller_ms = performance.now() - started;
    await answered;
    await this.interactionHost?.replaceScene(next);
    this.document = next;
    return caller_ms;
  }

  /** One animation frame: `draw` the document, and wait for the worker's answer to it. */
  private async frame(readImage: boolean): Promise<FrameRecord> {
    const answered = this.host.waitFor<FrameResponse>((message) => message.type === "frame");
    const started = performance.now();
    this.client.draw(this.simulationTime_s, this.document, this.camera, ...this.drawTail);
    const caller_ms = performance.now() - started;
    const response = await this.boundedSimulationWait(answered, "worker frame");
    const host_ms = performance.now() - started;
    await this.boundedSimulationWait(this.host.idle(), "GPU frame completion");
    const wall_ms = this.options.device === "dawn" ? performance.now() - started : undefined;
    const gpuPasses_ms = await this.recorder.takePassTimes();
    const image = readImage ? await this.host.readCanvas() : undefined;
    const taken = this.recorder.take();
    const revision = this.sourceRevision();
    return { host_ms, caller_ms, traffic: this.host.takeTraffic(), wall_ms, gpuPasses_ms, gpuTimeline: this.recorder.lastPassTimeline,
      submitted: response.metrics.presentationSubmitted, gpu: taken.gpu, stages: taken.stages,
      maintenance: taken.stages[MAINTENANCE_STAGE] && Boolean(taken.stages[MAINTENANCE_STAGE]!.returnedTrue),
      voxelizationDispatches: taken.stages[VOXELIZATION_STAGE]?.dispatches,
      revision,
      buildInFlight: this.host.buildInFlight,
      meshPending: this.options.device === "dawn" && this.surfaceMesh()?.state === "pending", image };
  }

  private async boundedSimulationWait<T>(promise: Promise<T>, operation: string): Promise<T> {
    if (!this.options.simulationSteps) return promise;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([promise, new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`Simulation ${operation} stalled for 120 seconds.`)), 120_000);
      })]);
    } finally { if (timer !== undefined) clearTimeout(timer); }
  }

  private assertSimulationHealthy(): void {
    if (this.host.failure || this.host.simulationErrors.length || this.host.validationErrors.length) {
      throw new Error(`Advancing simulation failed: ${this.host.failure ?? [...this.host.simulationErrors, ...this.host.validationErrors].join("; ")}`);
    }
  }

  private simulationCompletedTime(): number { return this.externalCompletions_s.at(-1) ?? this.host.advanceCompletions_s.at(-1) ?? 0; }
  private simulationReceiptCount(): number { return this.externalCompletions_s.length + this.host.advanceCompletions_s.length; }
  simulationReport() {
    return { ...this.simulation, completedReceipts: this.simulationReceiptCount(), completedTime_s: this.simulationCompletedTime(),
      clockAdvanceReceipts: this.host.advanceCompletions_s.length, externalFluidResponses: this.externalCompletions_s.length,
      interaction: this.interactionHost ? { readiness: this.interactionHost.getView().readiness,
        snapshot: this.interactionHost.getView().snapshot } : undefined,
      finalGpuInfo: this.host.gpuInfo, errors: this.host.simulationErrors };
  }

  /** Wait for the production GPU completion receipt, not merely an encoded frame. */
  private async advancingFrame(): Promise<FrameRecord> {
    if (this.interactionHost) {
      const before = this.simulationCompletedTime(), tick = this.interactionHost.getView().snapshot?.tick ?? 0;
      await this.interactionHost.advance(1 / 60);
      const view = this.interactionHost.getView();
      if (view.readiness.state !== "ready" || view.snapshot?.tick !== tick + 1 || this.simulationCompletedTime() <= before) {
        throw new Error("Box3D/fluid shared-clock step did not complete with a ready host and advancing tick.");
      }
      this.drawTail[0] = this.interactionHost.currentBodies();
      for (const body of this.drawTail[0]) if (![body.position_m, body.orientation, body.linearVelocity_m_s, body.angularVelocity_rad_s]
        .every(value => Object.values(value).every(Number.isFinite))) throw new Error(`Non-finite coupled body state: ${body.description.id}`);
      this.simulationTime_s = view.snapshot.time_s;
      const frame = await this.frame(false);
      this.assertSimulationHealthy();
      return frame;
    }
    const before = this.host.advanceCompletions_s.at(-1) ?? 0;
    this.simulationTime_s = Math.max(this.simulationTime_s, this.host.gpuInfo?.submittedTime_s ?? 0) + 1 / 30;
    const deadline = performance.now() + 120_000;
    do {
      const frame = await this.frame(false);
      this.assertSimulationHealthy();
      const completed = this.host.advanceCompletions_s.at(-1) ?? 0;
      if (completed > before) {
        const info = this.host.gpuInfo;
        if (!info || info.simulationPipelinesReady === false || (info.completedTime_s ?? 0) < completed) {
          throw new Error("Advance receipt lacks a ready solver and matching gpu-info completed time.");
        }
        // Do not leave an owed target-clock step running during the paused
        // image/mesh settling that follows a smoke checkpoint.
        this.simulationTime_s = info.submittedTime_s ?? completed;
        return frame;
      }
      await new Promise<void>(resolve => setTimeout(resolve, 8));
    } while (performance.now() < deadline);
    throw new Error(`Simulation made no completed progress in 120 seconds (solver ${this.host.gpuInfo ? "present" : "missing"}, pipelines ready ${this.host.gpuInfo?.simulationPipelinesReady}, submitted ${this.host.gpuInfo?.submittedTime_s}, completed ${this.host.gpuInfo?.completedTime_s}).`);
  }

  private async simulationCheckpoint(phase: string): Promise<void> {
    if (!this.options.simulationSteps) return;
    const before_s = this.simulationCompletedTime(), beforeCount = this.simulationReceiptCount();
    await this.client.setSimulationRunning(true);
    this.interactionHost?.setRunning(true);
    try {
      for (let i = 0; i < this.options.simulationSteps; i++) await this.advancingFrame();
      const after_s = this.simulationCompletedTime();
      this.simulation.checkpoints.push({ phase, requestedSteps: this.options.simulationSteps,
        completedReceipts: this.simulationReceiptCount() - beforeCount, before_s, after_s });
      console.log(`simulation ${phase}: completed ${before_s.toFixed(6)} → ${after_s.toFixed(6)} s; healthy GPU receipts`);
    } catch (error) {
      this.simulation.error = error instanceof Error ? error.message : String(error);
      throw error;
    } finally { this.interactionHost?.setRunning(false); await this.client.setSimulationRunning(false); }
  }

  /** The published sparse scene's revision word, which moves when its topology is republished. */
  private sourceRevision(): number | undefined {
    const revision = peek(this.recorder.instances.get("WebGPULiveSvoScene"), "sparseVoxelSceneSource", "revision")
      ?? peek(this.recorder.instances.get("OctreeSparseBrickWorld"), "sceneSource", "revision");
    return typeof revision === "number" ? revision : undefined;
  }

  /** Frames whose scene maintenance reports nothing left; without that report, frames that encode the same work twice running. */
  private quiet(frame: FrameRecord, previous: FrameRecord | undefined): boolean {
    if (frame.buildInFlight || frame.meshPending || !frame.submitted) return false;
    if (frame.maintenance !== undefined) return !frame.maintenance;
    return previous !== undefined && JSON.stringify(frame.gpu.passes) === JSON.stringify(previous.gpu.passes)
      && frame.gpu.uploadBytes === previous.gpu.uploadBytes;
  }

  /** Draw until the scene is quiescent (and, on a real device, the image has stopped changing). */
  private async settle(limit: number, readImages: boolean): Promise<FrameRecord[]> {
    const frames: FrameRecord[] = [];
    let quietRun = 0;
    // The limit is on frames a built scene is given to settle, not on the build's own.
    let settling = 0;
    while (settling < limit || this.host.buildInFlight) {
      const frame = await this.frame(readImages);
      const previous = frames[frames.length - 1];
      frames.push(frame);
      settling = frame.buildInFlight ? 0 : settling + 1;
      const still = !frame.image || !previous?.image
        || changedPixels(frame.image, previous.image) <= STILL_FRACTION * frame.image.width * frame.image.height;
      quietRun = this.quiet(frame, previous) && still ? quietRun + 1 : 0;
      if (quietRun >= IDLE_FRAMES) break;
      // A build yields to the event loop between slices; give it the thread rather than a frame.
      if (frame.buildInFlight) await new Promise<void>((resolve) => setTimeout(resolve, 8));
      if (this.host.buildInFlight && frames.length > 200_000) throw new Error("The world build did not finish.");
    }
    return frames;
  }

  // -- setup ------------------------------------------------------------------

  async start(): Promise<void> {
    const options = this.options;
    for (const stage of STAGES) {
      const owner = await optionalExport(stage.path, stage.owner) as { prototype?: unknown } | undefined;
      this.recorder.instrument(stage.owner, owner?.prototype, stage.methods);
    }
    if (options.device === "fake") await this.selectBricksOnTheHost();
    await this.client.initialize();
    if (this.options.simulationSteps) await this.client.setSimulationRunning(false);
    if (this.host.failure) throw new Error(`The renderer did not start: ${this.host.failure}`);
    await this.loadDrawArguments();
    await this.loadKeyFunctions();

    const scene = options.fluid === "off"
      ? { ...this.authored.scene, systems: { ...this.authored.scene.systems, fluid: false } } as SceneDescription
      : this.authored.scene;
    if (options.simulationSteps) {
      if (scene.interaction?.entities.length) {
        const { SceneInteractionHost } = await import("../lib/interaction/runtime/host");
        this.interactionHost = new SceneInteractionHost(scene, { fluid: {
          failureReason: () => this.host.failure ?? this.host.simulationErrors[0],
          step: async request => {
          const result = await this.boundedSimulationWait(this.client.advanceInteractionFluid(request), "external fluid response");
          this.assertSimulationHealthy();
          const info = this.host.gpuInfo;
          if (!info || !Number.isFinite(info.completedTime_s) || Math.abs((info.completedTime_s ?? 0) - request.time_s) > 1e-7
            || request.time_s <= this.simulationCompletedTime()) {
            throw new Error(`External fluid response lacks matching gpu-info completion: requested ${request.time_s}, completed ${info?.completedTime_s}.`);
          }
          this.externalCompletions_s.push(request.time_s);
          return result;
        } } });
        await this.interactionHost.ready;
        this.drawTail[0] = this.interactionHost.currentBodies();
        this.simulation.mode = "authored SceneInteractionHost Box3D + worker NB-FLIP external steps";
      }
      this.simulation.bodyIds = this.drawTail[0].map(body => body.description.id);
    }
    this.baseDocument = this.document = scene;
    this.recorder.take();
    const started = performance.now();
    const frames = await this.settle(options.maxFrames, false);
    const built = frames.findIndex((frame) => frame.submitted && !frame.buildInFlight);
    if (built < 0) {
      const last = frames[frames.length - 1];
      console.error(JSON.stringify({
        frames: frames.length,
        submitted: frames.filter((frame) => frame.submitted).length,
        buildInFlight: frames.filter((frame) => frame.buildInFlight).length,
        meshPending: frames.filter((frame) => frame.meshPending).length,
        maintenance: frames.filter((frame) => frame.maintenance).length,
        statuses: this.host.statuses.map((status) => `${status.state}: ${status.label}`),
        surfaceMesh: this.surfaceMesh(),
        lastFramePasses: last ? Object.keys(last.gpu.passes) : [],
        lastFrameStages: last ? Object.fromEntries(Object.entries(last.stages).map(([name, tally]) => [name, tally.calls])) : {},
        wall_ms: frames.slice(-5).map((frame) => frame.wall_ms),
      }, undefined, 1));
      throw new Error(`The scene never presented a frame. Last status: ${this.host.statuses[this.host.statuses.length - 1]?.label ?? "none"}`);
    }
    const creations: Record<string, number> = {};
    let uploadBytes = 0;
    for (const frame of frames) {
      uploadBytes += frame.gpu.uploadBytes;
      for (const [kind, count] of Object.entries(frame.gpu.creations)) creations[kind] = (creations[kind] ?? 0) + count;
    }
    this.worldBuild = { ms: this.host.builds.reduce((sum, build) => sum + build.ms, 0) || performance.now() - started,
      frames: built + 1, builds: this.host.builds.splice(0), creations, uploadBytes, settleFrames: frames.length - built - 1 };
    this.sites = new EditSites(scene, options.site, this.recorder.instances.get("OctreeSparseBrickWorld"));
    this.host.takeTraffic();
    await this.simulationCheckpoint("baseline");
  }

  /**
   * The world build asks the GPU which bricks the scene occupies and reads the
   * answer back. A stand-in device answers zeros, which would build a world
   * with none of the scene's bricks in it, so the stand-in build is switched to
   * the product's own host-side selection (`cpuBrickSelection`, its paired
   * oracle for the GPU one). Everything after the build is unchanged.
   */
  private async selectBricksOnTheHost(): Promise<void> {
    const live = await optionalExport(LIVE_SCENE_PATH, "WebGPULiveSvoScene") as Mutable | undefined;
    const create = live?.create as ((...args: unknown[]) => unknown) | undefined;
    if (typeof create !== "function" || create.length > 6) { this.brickSelection = "zeroed GPU readbacks (WebGPULiveSvoScene.create has changed shape); the world's bricks are not the scene's"; return; }
    live!.create = function createWithHostSelection(this: unknown, ...args: unknown[]) {
      while (args.length < 5) args.push(undefined);
      args[5] = { ...(args[5] as object | undefined), cpuBrickSelection: true };
      return Reflect.apply(create, this, args);
    };
    this.brickSelection = "the product's host-side selection (cpuBrickSelection), since a stand-in device cannot answer the GPU's";
  }

  /** The arguments `WebGPUViewport` passes to `draw` after the scene and camera, from the product's own stores. */
  private async loadDrawArguments(): Promise<void> {
    const methodStore = await optionalExport("../lib/core/stores/method-store.ts", "createMethodStore") as (() => { getState(): Mutable }) | undefined;
    const resolved = await optionalExport("../lib/core/stores/method-store.ts", "resolvedMethodValues") as ((state: Mutable) => unknown) | undefined;
    const uiStore = await optionalExport("../lib/core/stores/ui-store.ts", "createUIStore") as (() => { getState(): Mutable }) | undefined;
    if (!methodStore || !resolved) throw new Error("lib/core/stores/method-store no longer exports createMethodStore and resolvedMethodValues; the probe cannot form the product's run configuration.");
    const store = methodStore();
    if (this.options.simulationSteps && this.options.scene !== "hall") {
      const { getSceneDefinition } = await import("../lib/core/scenes");
      const profile = getSceneDefinition(this.options.scene).methodProfile;
      if (profile) {
        const applyProfile = store.getState().applyProfile;
        if (typeof applyProfile !== "function") throw new Error("The method store cannot apply the authored scene profile.");
        applyProfile(profile);
      }
    }
    const method = store.getState();
    const ui = uiStore?.().getState();
    this.methodId = method.methodId;
    this.runConfig = { methodId: method.methodId, quality: method.quality, values: resolved(method), simulationEpoch: 0, inFlightDepth: 1 };
    if (this.options.simulationSteps) {
      if (this.options.scene === "backyard-pool-party" && this.methodId !== "uniform-narrow-band-flip") {
        throw new Error(`Pool Party smoke requires its NB-FLIP method; resolved ${this.methodId}.`);
      }
      this.simulation.methodConfig = structuredClone(this.runConfig);
    }
    const lighting = ui && {
      shadowsEnabled: ui.svoShadowsEnabled, ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
      silhouetteRefinementEnabled: ui.silhouetteRefinementEnabled, coneTracingMode: ui.svoConeTracingMode,
      globalIlluminationEnabled: ui.svoGlobalIlluminationEnabled, worldGiCacheEnabled: ui.svoWorldGiCacheEnabled,
      latticeVisibilityEnabled: ui.svoLatticeVisibilityEnabled, primaryTraversal: ui.svoPrimaryTraversal,
      disabledStages: ui.disabledRenderStages,
    };
    const diagnostics = ui && { stageView: ui.svoStageView, lightSlot: ui.svoStageLightSlot,
      maximumTraversalDepth: ui.svoMaximumTraversalDepth, maximumNodeVisits: ui.svoMaximumNodeVisits };
    const overlayConfig = ui && { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode,
      layers: ui.visualLayers, particles: ui.fluidParticleDisplay };
    // Undefined arguments take `draw`'s own defaults, which is what a missing store leaves.
    this.drawTail = [initializeRigidBodies(this.authored.scene.rigidBodies), undefined, this.runConfig, overlayConfig,
      this.authored.background, this.authored.presentationMode, ui?.fluidSurfaceRenderMode, lighting, diagnostics, ui?.svoRenderTuning, undefined] as unknown as DrawTail;
  }

  /** The renderer's rebuild keys, by the names it exports them under today. */
  private async loadKeyFunctions(): Promise<void> {
    const renderer = "../lib/core/webgpu-renderer.ts";
    const candidates: [string, string, string, (key: (...args: any[]) => string) => (scene: SceneDescription) => string][] = [ // eslint-disable-line @typescript-eslint/no-explicit-any
      ["sceneryConstruction", "../lib/core/scenery-construction-key.ts", "sceneryConstructionKey", (key) => (scene) => key(scene)],
      ["sceneStructural", renderer, "sceneStructuralKey", (key) => (scene) => key(scene)],
      ["gpuSceneStructural", renderer, "gpuSceneStructuralKey", (key) => (scene) => key(scene, this.runConfig)],
      ["gpuSceneSeed", renderer, "gpuSceneSeedKey", (key) => (scene) => key(scene, this.methodId)],
      ["gpuSceneSolver", renderer, "gpuSceneSolverKey", (key) => (scene) => key(scene, this.runConfig)],
      ["gpuSceneUniform", renderer, "gpuSceneUniformKey", (key) => (scene) => key(scene)],
    ];
    for (const [name, path, exported, bind] of candidates) {
      const key = await optionalExport(path, exported);
      if (typeof key === "function") this.keyFunctions[name] = bind(key as (...args: unknown[]) => string);
    }
  }

  private keys(scene: SceneDescription): Record<string, string | undefined> {
    const keys: Record<string, string | undefined> = {};
    for (const [name, key] of Object.entries(this.keyFunctions)) {
      try { keys[name] = String(key(scene)); } catch { keys[name] = undefined; }
    }
    return keys;
  }

  // -- edits ------------------------------------------------------------------

  /** Aim at a solid edit from close enough that it fills a known share of the frame. */
  private aim(edit: SolidEdit | undefined): { x0: number; y0: number; x1: number; y1: number } | undefined {
    if (!edit) { this.camera = this.baseCamera; return undefined; }
    const { centre, radius } = this.sites.sphereOf(edit.bounds);
    const tanHalfFov = this.baseCamera.tanHalfFov ?? 0.72;
    const distance_m = Math.max(0.12, 5 * radius / tanHalfFov);
    this.camera = { ...this.baseCamera, azimuth_rad: 0.7, elevation_rad: 0.6, distance_m,
      target_m: { x: centre[0], y: centre[1], z: centre[2] } };
    // The target projects to the centre of the frame; the edit's bounding
    // sphere spans radius / (distance * tanHalfFov) of the half height, padded
    // by half again for perspective and by four pixels for filtering.
    const { width, height } = this.options;
    const half = Math.ceil(1.5 * radius / (distance_m * tanHalfFov) * height / 2) + 4;
    return { x0: Math.max(0, Math.floor(width / 2) - half), y0: Math.max(0, Math.floor(height / 2) - half),
      x1: Math.min(width, Math.ceil(width / 2) + half), y1: Math.min(height, Math.ceil(height / 2) + half) };
  }

  /** Whether two documents differ in their solid patches and nothing else: what an undo can cross as a patch count. */
  private solidOnly(a: SceneDescription, b: SceneDescription): boolean {
    return Object.keys(a).every((field) => field === "solidVoxels" || (a as unknown as Mutable)[field] === (b as unknown as Mutable)[field]);
  }

  /** Return to the base document the way the product would, without measuring it. */
  private async restoreBase(): Promise<void> {
    if (this.document === this.baseDocument) return;
    if (this.solidOnly(this.baseDocument, this.document)) await this.accept(this.baseDocument);
    else this.document = this.baseDocument;
    await this.settle(this.options.maxFrames, false);
    this.host.builds.length = 0;
    this.host.takeTraffic();
    this.recorder.take();
  }

  /** One edit, start to quiescence. A renderer that stops part-way is reported against the edit, not thrown. */
  async measure(edit: EditName): Promise<EditReport> {
    const report = emptyReport(edit);
    if (this.host.failure) { report.skipped = `not run: the renderer had stopped (${this.host.failure}).`; return report; }
    this.phase = "preparing the edit";
    try {
      if (edit === "pen-draw" || edit === "pen-carve") await this.runPen(report, edit === "pen-carve");
      else await this.run(report, edit);
    } catch (error) {
      report.refused = `${error instanceof Error ? error.message : String(error)} — while ${this.phase}`;
      if (this.options.simulationSteps) this.simulation.error = report.refused;
      this.recorder.take();
    }
    return report;
  }

  /** Real plugin + live transaction + worker acceptance, one presented frame per pointer sample. */
  private async runPen(report: EditReport, carve: boolean): Promise<void> {
    await this.restoreBase();
    await this.simulationCheckpoint(`before ${report.edit}`);
    const base = this.document, planningStart = performance.now();
    const requestedNodes = this.options.penNode?.split(",");
    const requestedNode = requestedNodes?.[this.penStrokeIndex % requestedNodes.length];
    const plan = planPenStroke(base, this.options.site, carve, this.options.penSize, requestedNode, this.options.penSamples);
    if (!plan) { this.penStrokeIndex++; report.skipped = `No accessible imported surface ${requestedNode ? `matching ${requestedNode}` : "near the requested site"}.`; return; }
    const pen = report.pen = { nodeId: plan.nodeId, planning_ms: performance.now() - planningStart,
      samples: [] as NonNullable<EditReport["pen"]>["samples"], publications: 0, transactionFinishes: 0 } as NonNullable<EditReport["pen"]>;
    report.description = plan.solid.note;
    this.aim(plan.solid);
    if (this.options.penCamera === "close") this.camera = { ...this.camera, distance_m: this.camera.distance_m * .45, elevation_rad: 1.0 };
    await this.settle(this.options.maxFrames, false);
    const renderer = this.recorder.instances.get("SparseVoxelDrySceneRenderer");
    const pipelineVariants = () => {
      const keys = (field: string) => { const map = peek(renderer, field); return map instanceof Map ? [...map.keys()].map(String) : []; };
      return { cached: keys("splitPipelineBundles"), pending: keys("splitPipelineCompiles") };
    };
    const beforeWarmup = pipelineVariants(), warmupStart = performance.now();
    // Startup can present a compatible generic bundle while the scene's
    // specialized variant compiles (and can prewarm a body-capable twin).
    // Exclude those async creates from the first edit that runs beside them.
    const pending = peek(renderer, "splitPipelineCompiles");
    if (pending instanceof Map && pending.size) {
      this.phase = "waiting for background pipeline warmup";
      console.log(`pen startup: waiting for ${pending.size} background pipeline variant(s); excluded from edit timings`);
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([Promise.all([...pending.values()]), new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error("Background pipeline warmup exceeded 120 seconds.")), 120_000);
        })]);
      } finally { if (timer !== undefined) clearTimeout(timer); }
    }
    pen.pipelineWarmup_ms = performance.now() - warmupStart;
    const idle = await this.settle(this.options.maxFrames, false);
    const idleHost = median(idle.slice(-IDLE_FRAMES).map(frame => frame.host_ms));
    const meshBefore = this.surfaceMesh();
    const variantsBefore = pipelineVariants();
    if (variantsBefore.pending.length) throw new Error("Background pipeline variants are still compiling after warmup.");
    this.penStrokeIndex++;
    const imageDirectory = this.options.penImages ? resolve(this.options.penImages) : undefined;
    if (imageDirectory) mkdirSync(imageDirectory, { recursive: true });
    const imagePath = (phase: string) => join(imageDirectory!, `${this.penStrokeIndex}-${report.edit}-${phase}.png`);
    const baselineImage = imageDirectory ? await this.host.readCanvas() : undefined;
    if (baselineImage) saveCanvasPng(imagePath("baseline"), baselineImage);
    this.host.builds.length = 0; this.host.takeTraffic(); this.recorder.take();
    const frames: FrameRecord[] = [];
    let acceptance_ms = 0;
    const transaction = beginToolTransaction(penTool, {
      scene: () => this.document,
      begin: () => {},
      publish: async (next, previous) => {
        if (this.document !== previous) throw new Error("Live probe received a stale pen document.");
        const start = performance.now(); await this.accept(next); acceptance_ms += performance.now() - start;
        pen.publications++;
      },
      finish: () => { pen.transactionFinishes++; },
      cancel: async () => { if (this.document !== base) await this.accept(base); },
    }, plan.rays[0]!, plan.values);
    if (!transaction) throw new Error("The pen lost its planned surface before the stroke began.");
    const simulationBefore = this.simulationCompletedTime();
    const simulationReceiptCount = this.simulationReceiptCount();
    if (this.options.simulationSteps) await this.client.setSimulationRunning(true);
    if (this.options.simulationSteps) this.interactionHost?.setRunning(true);
    try {
      for (const ray of plan.rays) {
        this.phase = `accepting pen sample ${pen.samples.length + 1}`;
        acceptance_ms = 0;
        const publications = pen.publications, start = performance.now();
        const accepted = transaction.update(ray), geometry_ms = performance.now() - start;
        const update = await accepted, update_ms = performance.now() - start;
        const frame = this.options.simulationSteps ? await this.advancingFrame() : await this.frame(false); frames.push(frame);
        // Publication occurs inside draw; inspect the plan after that frame,
        // otherwise this reports the preceding pointer sample's dirty region.
        const proxy = this.recorder.instances.get("SparseSceneProxyVoxelizer");
        const dirtyBuildPlan = Object.fromEntries(["pending", "drainChunks", "plannedChunks", "chunksRemaining",
          "plannedChunkBricks", "lastRegionCellCount", "primitiveCount"].flatMap(key => {
          const value = peek(proxy, key); return typeof value === "number" || typeof value === "boolean" ? [[key, value]] : [];
        })) as Record<string, number | boolean>;
        pen.samples.push({ geometry_ms, acceptance_ms, update_ms, patches: update?.patches.length ?? 0,
          published: pen.publications !== publications, frameHost_ms: frame.host_ms, frameWall_ms: frame.wall_ms,
          gpuPasses_ms: frame.gpuPasses_ms, gpuTimeline: frame.gpuTimeline, dirtyBuildPlan, stages: frame.stages });
      }
      await transaction.finish();
    } catch (error) { await transaction.finish(true); throw error; }
    finally { if (this.options.simulationSteps) { this.interactionHost?.setRunning(false); await this.client.setSimulationRunning(false); } }
    if (this.options.simulationSteps) this.simulation.checkpoints.push({ phase: `during ${report.edit}`, requestedSteps: plan.rays.length,
      completedReceipts: this.simulationReceiptCount() - simulationReceiptCount,
      before_s: simulationBefore, after_s: this.simulationCompletedTime() });
    await this.simulationCheckpoint(`after ${report.edit}`);
    this.phase = "settling the completed pen stroke";
    const settled = await this.settle(this.options.maxFrames, false); frames.push(...settled);
    report.surfaceMesh = { before: meshBefore, after: this.surfaceMesh() };
    pen.pipelineVariants = { beforeWarmup, before: variantsBefore, after: pipelineVariants() };
    const editedImage = imageDirectory ? await this.host.readCanvas() : undefined;
    if (editedImage) saveCanvasPng(imagePath("edited"), editedImage);
    report.frames = frames.length;
    report.hostFrames_ms = frames.map(frame => frame.host_ms);
    report.wallFrames_ms = this.options.device === "dawn" ? frames.map(frame => frame.wall_ms ?? 0) : undefined;
    report.gpuPassFrames_ms = this.options.device === "dawn" ? frames.map(frame => frame.gpuPasses_ms ?? {}) : undefined;
    report.frameCapReached = !this.quiet(settled[settled.length - 1]!, settled[settled.length - 2]);
    // Duplicate pointer hits do not publish an edit. Including their zero
    // acceptance cost would hide slow edits on densely sampled strokes.
    report.editHost_ms = median(pen.samples.filter(sample => sample.published).map(sample => sample.acceptance_ms));
    report.idleFrameHost_ms = idleHost;
    report.dirtyBricks = this.sites.dirtyBricks(plan.solid.bounds);
    let uploads = 0;
    for (const frame of frames) {
      uploads += frame.gpu.uploadBytes;
      for (const [name, stage] of Object.entries(frame.stages)) {
        const total = report.stages[name] ??= { calls: 0, ms: 0, dispatches: 0, uploadBytes: 0 };
        total.calls += stage.calls; total.ms += stage.ms; total.dispatches += stage.dispatches; total.uploadBytes += stage.uploadBytes;
        if (stage.settled_ms !== undefined) total.settled_ms = (total.settled_ms ?? 0) + stage.settled_ms;
      }
      for (const [kind, count] of Object.entries(frame.gpu.creations)) report.creations[kind] = (report.creations[kind] ?? 0) + count;
      for (const label of frame.gpu.created) if (report.created.length < CREATED_LABEL_CAP) report.created.push(label);
      for (const [name, pass] of Object.entries(frame.gpu.passes)) {
        const total = report.passes[name] ??= { direct: 0, indirect: 0, workgroups: 0 };
        total.direct += pass.direct; total.indirect += pass.indirect; total.workgroups += pass.workgroups;
      }
      report.draws.renderPasses += frame.gpu.renderPasses;
      report.draws.direct += frame.gpu.directDraws; report.draws.indirect += frame.gpu.indirectDraws;
      report.directDispatches += frame.gpu.directDispatches; report.indirectDispatches += frame.gpu.indirectDispatches;
    }
    this.phase = "undoing the live pen stroke";
    const undoStart = performance.now(); await this.accept(base);
    const undoAccept = performance.now() - undoStart, undone = await this.settle(this.options.maxFrames, false);
    pen.undo = { acceptance_ms: undoAccept, frames: undone.length, restoredPatchCount: this.document.solidVoxels.length, mesh: this.surfaceMesh() };
    await this.simulationCheckpoint(`after ${report.edit} undo`);
    const undoImage = imageDirectory ? await this.host.readCanvas() : undefined;
    if (undoImage) saveCanvasPng(imagePath("undo"), undoImage);
    if (baselineImage && editedImage && undoImage) pen.images = {
      baseline: imagePath("baseline"), edited: imagePath("edited"), undo: imagePath("undo"),
      changedPixels: changedPixels(baselineImage, editedImage), undoChangedPixels: changedPixels(baselineImage, undoImage),
    };
    report.rebuilds = this.host.builds.splice(0);
    report.budget = [
      { measure: "host accept", value: `${ms(report.editHost_ms)} median/accepted prefix`, budget: "< 5 ms/accepted prefix", verdict: report.editHost_ms < 5 ? "pass" : "FAIL" },
      { measure: "uploads", value: `${bytes(uploads)} including frame traffic`, budget: "reported for whole stroke", verdict: "" },
      { measure: "live transaction", value: `${pen.samples.length} samples, ${pen.publications} accepted suffixes, ${pen.transactionFinishes} finish`,
        budget: `${this.options.penSamples} samples, one finish`, verdict: pen.samples.length === this.options.penSamples && pen.transactionFinishes === 1 ? "pass" : "FAIL" },
      { measure: "rebuild", value: report.rebuilds.length ? "world rebuilt" : "none", budget: "none", verdict: report.rebuilds.length ? "FAIL" : "pass" },
      { measure: "undo", value: `${ms(undoAccept)}, ${undone.length} settling frames`, budget: "restore base patches", verdict: this.document === base ? "pass" : "FAIL" },
    ];
  }

  private async run(report: EditReport, edit: EditName): Promise<void> {
    const real = this.options.device === "dawn";

    // What the edit is, before anything is sent.
    let solid: SolidEdit | undefined;
    let target: SceneDescription | undefined;
    try {
      if (edit === "erase") {
        if (!this.voxelDocument) { report.skipped = "erase removes what voxel added; run it after voxel (--edit=voxel,erase)."; return; }
        if (this.document !== this.voxelDocument) { report.skipped = "the voxel's document is no longer the one being drawn."; return; }
        solid = this.sites.voxel("clear");
      } else {
        await this.restoreBase();
        if (edit === "voxel") solid = this.sites.voxel("fill");
        else if (edit === "voxel-inside") solid = this.sites.voxelInside();
        else if (edit === "sphere") solid = this.sites.sphere();
        else if (edit === "slab") solid = this.sites.slab();
      }
      const property = PROPERTY_EDITS[edit];
      if (solid) {
        report.description = solid.note;
        // What `voxel-editor/transaction.ts` hands the viewport for a stroke.
        target = sceneWithSolidStroke(this.document, solid.patches);
      } else if (edit === "none") {
        report.description = "an equal document, published again";
        // New identity, every part shared: what a store hands on for a write that changed nothing.
        target = { ...this.document };
      } else if (edit === "undo") {
        // Something to undo: the voxel, accepted and settled, unmeasured.
        solid = this.sites.voxel("fill");
        await this.accept(sceneWithSolidStroke(this.document, solid.patches));
        report.description = `back to the base document from ${solid.note}`;
        target = this.baseDocument;
      } else if (CONTROL_EDITS[edit]) {
        const made = CONTROL_EDITS[edit]!(this.document);
        if (!made) { report.skipped = "this scene has nothing of that kind to edit."; return; }
        report.description = made.described;
        target = made.next;
      } else if (property) {
        const clone = structuredClone(this.document) as unknown as Mutable;
        const described = property(clone);
        if (!described) { report.skipped = "this scene has nothing of that kind to edit."; return; }
        report.description = described;
        // The editor's stores write persistently: only the path to the edited
        // part is new, and every other part is the object it was.
        const entries = documentDelta(this.document, clone);
        target = entries ? applyDocumentDelta(this.document, entries) : clone as unknown as SceneDescription;
      }
    } catch (error) {
      report.refused = `while preparing the edit: ${error instanceof Error ? error.message : String(error)}`;
      return;
    }
    if (!target) { report.skipped = "not implemented."; return; }
    if (solid) report.dirtyBricks = this.sites.dirtyBricks(solid.bounds);

    // A settled frame from the edit's camera to compare against.
    this.phase = "settling before the edit was sent";
    const rect = this.aim(solid);
    const settled = await this.settle(real ? Math.min(REAL_SETTLE_FRAMES, this.options.maxFrames) : this.options.maxFrames, real);
    const idleFrames = settled.slice(-IDLE_FRAMES);
    this.idle = { host_ms: median(idleFrames.map((frame) => frame.host_ms)),
      uploadBytes: Math.min(...idleFrames.map((frame) => frame.gpu.uploadBytes)),
      voxelizationDispatches: Math.min(...idleFrames.map((frame) => frame.voxelizationDispatches ?? 0)) };
    const before = settled[settled.length - 1]!;
    report.idleFrame = { uploadBytes: before.gpu.uploadBytes, directDispatches: before.gpu.directDispatches, indirectDispatches: before.gpu.indirectDispatches };
    const revisionBefore = before.revision;
    const meshBefore = this.surfaceMesh();
    const keysBefore = this.keys(this.document);
    // What two settled frames differ by with no edit between them: in the edit's rectangle, and over the frame.
    const noise = { rect: 0, frame: 0 };
    for (let index = Math.max(1, settled.length - IDLE_FRAMES); index < settled.length; index += 1) {
      const a = settled[index - 1]!.image, b = settled[index]!.image;
      if (!a || !b) continue;
      noise.rect = Math.max(noise.rect, changedPixels(a, b, rect));
      noise.frame = Math.max(noise.frame, changedPixels(a, b));
    }
    if (real) { report.noiseFloor = noise; report.rect = rect ?? { x0: 0, y0: 0, x1: this.options.width, y1: this.options.height }; }
    this.host.builds.length = 0;
    this.host.takeTraffic();
    this.recorder.take();

    // The edit. A solid edit is accepted before it is drawn; any other document is simply drawn.
    this.phase = "the edit was being accepted";
    const started = performance.now();
    const solidOnly = this.solidOnly(target, this.document) && target !== this.document && edit !== "none";
    let acceptCaller_ms = 0;
    try {
      if (solidOnly) acceptCaller_ms = await this.accept(target);
      else this.document = target;
    } catch (error) {
      report.refused = error instanceof Error ? error.message : String(error);
      this.recorder.take();
      return;
    }
    const reply_ms = solidOnly ? performance.now() - started : 0;
    const accepted = this.recorder.take();
    const traffic = this.host.takeTraffic();
    const revisionSeries = [revisionBefore, this.sourceRevision()];
    if (edit === "none") this.baseDocument = target;
    if (edit === "voxel") this.voxelDocument = target;
    const keysAfter = this.keys(target);
    const changedKeys = Object.keys(keysBefore).filter((name) => keysBefore[name] !== keysAfter[name]);
    report.uniformKeyChanged = changedKeys.includes("gpuSceneUniform");
    report.rebuildKeysChanged = changedKeys.filter((name) => name !== "gpuSceneUniform");

    // Frames until nothing is left to do.
    this.phase = "drawing the frames after the edit";
    const frames: FrameRecord[] = [];
    let quietRun = 0, lastBusy = -1, lastMoving = -1;
    while (frames.length < this.options.maxFrames || this.host.buildInFlight) {
      const frame = await this.frame(real);
      const previous = frames[frames.length - 1] ?? before;
      frames.push(frame);
      const index = frames.length - 1;
      if (!this.quiet(frame, previous)) lastBusy = index;
      if (frame.image && before.image) {
        if (report.firstVisibleFrame === undefined && changedPixels(frame.image, before.image, rect) > 2 * noise.rect) report.firstVisibleFrame = index + 1;
        if (previous.image && changedPixels(frame.image, previous.image) > 2 * noise.frame) lastMoving = index;
      }
      quietRun = lastBusy === index || lastMoving === index ? 0 : quietRun + 1;
      if (quietRun >= IDLE_FRAMES) break;
      if (frame.buildInFlight) await new Promise<void>((resolve) => setTimeout(resolve, 8));
    }
    report.accept = { reply_ms, gpu: accepted.gpu, stages: accepted.stages };
    report.frameCapReached = quietRun < IDLE_FRAMES;
    report.rebuilds = this.host.builds.splice(0);
    // The edit's own frames: through the last one with work in it, and never fewer than the one that shows it.
    report.frames = Math.max(1, lastBusy + 1, lastMoving + 1);
    frames.length = report.frames;

    // Totals.
    const add = (into: Record<string, StageTally>, from: Record<string, StageTally>) => {
      for (const [name, stage] of Object.entries(from)) {
        const total = into[name] ??= { calls: 0, ms: 0, dispatches: 0, uploadBytes: 0 };
        total.calls += stage.calls; total.ms += stage.ms; total.dispatches += stage.dispatches; total.uploadBytes += stage.uploadBytes;
        if (stage.settled_ms !== undefined) total.settled_ms = (total.settled_ms ?? 0) + stage.settled_ms;
      }
    };
    const byLabel: Record<string, number> = {};
    const tally = (gpu: GpuTally) => {
      for (const [label, upload] of Object.entries(gpu.uploads)) byLabel[label] = (byLabel[label] ?? 0) + upload.bytes;
      for (const [name, pass] of Object.entries(gpu.passes)) {
        const total = report.passes[name] ??= { direct: 0, indirect: 0, workgroups: 0 };
        total.direct += pass.direct; total.indirect += pass.indirect; total.workgroups += pass.workgroups;
      }
      report.directDispatches += gpu.directDispatches;
      report.indirectDispatches += gpu.indirectDispatches;
      report.draws.renderPasses += gpu.renderPasses; report.draws.direct += gpu.directDraws; report.draws.indirect += gpu.indirectDraws;
      for (const [kind, count] of Object.entries(gpu.creations)) report.creations[kind] = (report.creations[kind] ?? 0) + count;
      for (const label of gpu.created) if (report.created.length < CREATED_LABEL_CAP) report.created.push(label);
    };
    tally(report.accept.gpu);
    add(report.stages, report.accept.stages);
    let framesOverIdle = 0;
    for (const frame of frames) {
      tally(frame.gpu);
      add(report.stages, frame.stages);
      traffic.sent.push(...frame.traffic.sent);
      for (const [type, value] of Object.entries(frame.traffic.clone_ms)) addTo(traffic.clone_ms, type, value);
      for (const [type, value] of Object.entries(frame.traffic.handler_ms)) addTo(traffic.handler_ms, type, value);
      framesOverIdle += Math.max(0, frame.gpu.uploadBytes - this.idle.uploadBytes);
    }
    report.indirectPasses = Object.entries(report.passes).filter(([, pass]) => pass.indirect > 0).map(([name, pass]) => `${name} x${pass.indirect}`);
    report.uploadBytes = { accept: report.accept.gpu.uploadBytes, framesOverIdle, byLabel };
    report.hostFrames_ms = frames.map((frame) => frame.host_ms);
    if (real) report.wallFrames_ms = frames.map((frame) => frame.wall_ms ?? 0);
    if (real) report.gpuPassFrames_ms = frames.map((frame) => frame.gpuPasses_ms ?? {});
    report.traffic = traffic;
    report.callerBlocked_ms = { accept: acceptCaller_ms, firstDraw: frames[0]?.caller_ms ?? 0 };
    report.firstFrameHost_ms = frames[0]?.host_ms;
    report.idleFrameHost_ms = this.idle.host_ms;
    report.editHost_ms = reply_ms + Math.max(0, (frames[0]?.host_ms ?? 0) - this.idle.host_ms);
    if (frames.some((frame) => frame.maintenance !== undefined)) report.maintenanceFrames = frames.filter((frame) => frame.maintenance).length;
    if (frames.some((frame) => frame.voxelizationDispatches !== undefined)) {
      report.voxelizationFrames = frames.filter((frame) => (frame.voxelizationDispatches ?? 0) > this.idle.voxelizationDispatches).length;
    }
    for (const frame of frames) revisionSeries.push(frame.revision);
    report.revisionSeries = revisionSeries;
    if (revisionSeries.every((value) => value !== undefined) && report.rebuilds.length === 0) {
      report.revisions = revisionSeries.filter((value, index) => index > 0 && value !== revisionSeries[index - 1]).length;
    }
    const calls = (stage: string) => this.recorder.instrumented.includes(stage) ? report.stages[stage]?.calls ?? 0 : undefined;
    report.publications = { voxelizer: calls("SparseSceneProxyVoxelizer.publish"), renderer: calls("SparseVoxelDrySceneRenderer.publishScene") };
    if (real) {
      report.convergedFrame = lastMoving + 2;
      // The mesh reports by readback, a few frames behind what it drew. These frames are not the edit's.
      for (let extra = 0; extra < 30 && this.surfaceMesh()?.state === "pending"; extra += 1) await this.frame(false);
      this.recorder.take();
      report.surfaceMesh = { before: meshBefore, after: this.surfaceMesh() };
      report.lighting = peek(this.recorder.instances.get("SparseVoxelDrySceneRenderer"), "lightingVisibilityStatus") ?? this.host.effectiveStatus?.lightingVisibility;
    }
    report.budget = this.budget(report, edit);
  }

  private surfaceMesh(): Mutable | undefined {
    const status = peek(this.recorder.instances.get("SparseVoxelDrySceneRenderer"), "surfaceMeshStatus") ?? this.host.effectiveStatus?.surfaceMesh;
    return status && typeof status === "object" ? { ...(status as Mutable) } : undefined;
  }

  private budget(report: EditReport, edit: EditName): BudgetRow[] {
    const rows: BudgetRow[] = [];
    const verdict = (pass: boolean): BudgetRow["verdict"] => pass ? "pass" : "FAIL";
    const solidEdit = ["voxel", "erase", "voxel-inside", "sphere", "slab", "undo"].includes(edit);
    rows.push({ measure: "host accept", value: ms(report.editHost_ms), budget: "< 5 ms", verdict: verdict((report.editHost_ms ?? Infinity) < 5) });

    const uploaded = report.uploadBytes!.accept + report.uploadBytes!.framesOverIdle;
    const uploadLimit = Math.max(65_536, 1024 * (report.dirtyBricks?.withHalo ?? 0));
    rows.push({ measure: "uploads", value: bytes(uploaded), budget: `<= ${bytes(uploadLimit)}`, verdict: verdict(uploaded <= uploadLimit) });

    const published = report.publications ?? {};
    const publications = `voxelizer published ${published.voxelizer ?? "?"}x, renderer ${published.renderer ?? "?"}x`;
    if (report.rebuilds.length > 0) rows.push({ measure: "scene revisions", value: "world rebuilt", budget: "<= 1", verdict: "" });
    else if (report.revisions === undefined) rows.push({ measure: "scene revisions", value: `not reported (${publications})`, budget: "<= 1", verdict: "not reported" });
    else rows.push({ measure: "scene revisions", value: `${report.revisions} (${publications})`, budget: "<= 1",
      verdict: verdict(report.revisions <= 1 && (published.voxelizer ?? 0) <= 1 && (published.renderer ?? 0) <= 1) });

    const voxelizationLimit = edit === "slab" ? 2 : ["voxel", "erase", "voxel-inside", "sphere", "undo"].includes(edit) ? 1 : 0;
    rows.push(report.voxelizationFrames === undefined
      ? { measure: "voxelization frames", value: "not reported", budget: `<= ${voxelizationLimit}`, verdict: "not reported" }
      : report.rebuilds.length > 0 ? { measure: "voxelization frames", value: `${report.voxelizationFrames} (world rebuilt)`, budget: `<= ${voxelizationLimit}`, verdict: "" }
      : { measure: "voxelization frames", value: String(report.voxelizationFrames), budget: `<= ${voxelizationLimit}`, verdict: verdict(report.voxelizationFrames <= voxelizationLimit) });

    const mesh = report.surfaceMesh?.after as Mutable | undefined, meshBefore = report.surfaceMesh?.before as Mutable | undefined;
    const meshBudget = "incremental, <= dirty + halo";
    const moved = (field: string) => typeof mesh?.[field] === "number" && typeof meshBefore?.[field] === "number"
      ? (mesh[field] - meshBefore[field]) >>> 0 : undefined;
    if (this.options.device !== "dawn" || !mesh || mesh.builds === undefined) {
      rows.push({ measure: "mesh build", value: this.options.device === "dawn" ? "not reported" : "needs a GPU readback", budget: meshBudget, verdict: "not reported" });
    } else if (moved("builds") === 0) {
      rows.push({ measure: "mesh build", value: `none (${mesh.state ?? "state not reported"})`, budget: meshBudget, verdict: solidEdit ? "FAIL" : "pass" });
    } else {
      // Counters that run across builds where the status has them; the last build's own numbers otherwise.
      const limit = report.dirtyBricks?.withHalo ?? 0;
      const whole = moved("wholeBuilds");
      const kind = whole !== undefined ? (whole > 0 ? "initial" : "incremental") : mesh.lastBuildKind ?? mesh.buildKind;
      const bricks = moved("rebuiltBricks") ?? (typeof mesh.lastBuildBricks === "number" ? mesh.lastBuildBricks : typeof mesh.completedBricks === "number" ? mesh.completedBricks : undefined);
      const grown = moved("arenaAllocations");
      rows.push({ measure: "mesh build", value: `${moved("builds") ?? "?"} build${moved("builds") === 1 ? "" : "s"}, ${kind ?? "kind not reported"}, ${bricks ?? "?"} bricks`
          + `${mesh.restartReason ? `, restart: ${mesh.restartReason}` : ""}${grown ? `, arena reallocated ${grown}x` : ""}${mesh.state === "ready" ? "" : `, still ${mesh.state}`}`,
        budget: solidEdit ? `incremental, <= ${limit} bricks` : "none",
        verdict: verdict(solidEdit && kind === "incremental" && bricks !== undefined && bricks <= limit && !grown) });
    }

    const created = Object.values(report.creations).reduce((sum, count) => sum + count, 0);
    rows.push({ measure: "GPU objects created", value: created === 0 ? "0" : Object.entries(report.creations).map(([kind, count]) => `${count} ${kind}`).join(", "),
      budget: "0", verdict: verdict(created === 0) });
    rows.push({ measure: "indirect dispatches", value: `${report.indirectDispatches} (of ${report.directDispatches + report.indirectDispatches})`, budget: "0", verdict: verdict(report.indirectDispatches === 0) });

    const rebuilt = report.rebuilds.length > 0 || (report.rebuildKeysChanged?.length ?? 0) > 0;
    const cost = report.rebuilds.length > 0 ? `${report.rebuilds.map((build) => `${build.kind} ${ms(build.ms)}`).join(", ")}` : "no build observed";
    rows.push({ measure: "rebuild", value: rebuilt ? `REBUILD (${cost}; keys: ${report.rebuildKeysChanged?.join(", ") || "none"})` : "none", budget: "none", verdict: verdict(!rebuilt) });

    if (this.options.device === "dawn") {
      rows.push(report.firstVisibleFrame === undefined
        ? { measure: "first visible frame", value: "never differed", budget: "<= 2", verdict: edit === "none" ? "pass" : "FAIL" }
        : { measure: "first visible frame", value: String(report.firstVisibleFrame), budget: "<= 2", verdict: verdict(report.firstVisibleFrame <= 2) });
    } else rows.push({ measure: "first visible frame", value: "needs a GPU", budget: "<= 2", verdict: "not reported" });
    return rows;
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

function table(rows: readonly (readonly string[])[]): string {
  const widths = rows[0]!.map((_, column) => Math.max(...rows.map((row) => (row[column] ?? "").length)));
  return rows.map((row) => row.map((cell, column) => column === row.length - 1 ? cell : cell.padEnd(widths[column]!)).join("  ").trimEnd()).join("\n");
}

/** How an edit that produced no measurements ended. */
function outcome(report: EditReport): string {
  if (report.skipped) return report.skipped.startsWith("not run") ? "not run" : "skipped";
  return report.refused?.startsWith("The renderer stopped") ? "STOPPED" : "REFUSED";
}

function printEdit(report: EditReport): void {
  console.log(`\n== ${report.edit}${report.description ? ` — ${report.description}` : ""}`);
  if (report.skipped) { console.log(`   skipped: ${report.skipped}`); return; }
  if (report.refused) { console.log(`   ${outcome(report)} by the product: ${report.refused}`); return; }
  if (report.pen) {
    const active = report.pen.samples.filter(sample => sample.published);
    const p95 = (values: number[]) => values.sort((a, b) => a - b)[Math.ceil(values.length * .95) - 1];
    console.log(table([["measure", "value", "budget", "verdict"], ...report.budget.map(row => [row.measure, row.value, row.budget, row.verdict])]));
    console.log(`   pointer geometry median ${ms(median(report.pen.samples.map(sample => sample.geometry_ms)))}; accepted-prefix frame host median ${ms(median(active.map(sample => sample.frameHost_ms)))}; planning ${ms(report.pen.planning_ms)} excluded.`);
    console.log(`   background pipeline warmup ${ms(report.pen.pipelineWarmup_ms ?? 0)} excluded; steady edit timing does not include initial startup.`);
    const timed = active.filter(sample => sample.gpuTimeline);
    if (timed.length) console.log(`   accepted-prefix GPU timestamp span median/p95 ${ms(median(timed.map(sample => sample.gpuTimeline!.span_ms)))}/${ms(p95(timed.map(sample => sample.gpuTimeline!.span_ms)))}; interval union median ${ms(median(timed.map(sample => sample.gpuTimeline!.covered_ms)))}; serial frame GPU-complete median/p95 ${ms(median(timed.map(sample => sample.frameWall_ms ?? 0)))}/${ms(p95(timed.map(sample => sample.frameWall_ms ?? 0)))}. No-op, settling and undo frames excluded; not browser presentation latency.`);
    console.log(`   ${report.frameCapReached ? "WARNING: settling frame cap reached" : "stroke settled"}; ${Object.values(report.creations).reduce((sum, count) => sum + count, 0)} GPU objects created during stroke; ${report.pen.images ? `images saved (${report.pen.images.changedPixels} changed pixels; ${report.pen.images.undoChangedPixels} after undo)` : "images not captured"}.`);
    console.log(table([["sample", "geometry ms", "acceptance ms", "update ms", "patches", "published", "frame ms"],
      ...report.pen.samples.map((sample, i) => [String(i + 1), ms(sample.geometry_ms), ms(sample.acceptance_ms), ms(sample.update_ms), String(sample.patches), String(sample.published), ms(sample.frameWall_ms ?? sample.frameHost_ms)])]));
    const stages = Object.entries(report.stages).sort((a, b) => b[1].ms - a[1].ms);
    if (stages.length) {
      console.log("   host stages (inclusive; acceptance and stroke/settling frames):");
      console.log(table(stages.map(([name, stage]) => [`     ${name}`, ms(stage.ms), `x${stage.calls}`])));
    }
    return;
  }
  const accept = report.accept!;
  const rows: string[][] = [["measure", "value", "budget", "verdict"]];
  for (const row of report.budget) rows.push([row.measure, row.value, row.budget, row.verdict]);
  rows.push(["accept (call to answer)", accept.reply_ms === 0 ? "no accept: the document is drawn" : ms(accept.reply_ms), "", ""]);
  const traffic = report.traffic!;
  const counted = [...new Set(traffic.sent)].map((type) => { const count = traffic.sent.filter((other) => other === type).length; return count > 1 ? `${type} x${count}` : type; });
  const costly = (totals: Record<string, number>) => Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([type, value]) => `${type} ${ms(value)}`).join(", ") || "none";
  rows.push(["messages sent", counted.join(", "), "", ""]);
  rows.push(["caller blocked", `${ms(report.callerBlocked_ms!.accept)} in the accept call, ${ms(report.callerBlocked_ms!.firstDraw)} in the first draw call (clones included)`, "", ""]);
  rows.push(["structured clone", costly(traffic.clone_ms), "", ""]);
  rows.push(["worker handlers", costly(traffic.handler_ms), "", ""]);
  rows.push(["first frame (draw to answer)", `${ms(report.firstFrameHost_ms)} (idle frame ${ms(report.idleFrameHost_ms)})`, "", ""]);
  rows.push(["frames until quiescent", `${report.frames}${report.frameCapReached ? " (frame cap reached; not quiescent)" : ""}`, "", ""]);
  rows.push(["maintenance frames", report.maintenanceFrames === undefined ? "not reported" : String(report.maintenanceFrames), "", ""]);
  if (report.dirtyBricks) rows.push(["dirty bricks", `${report.dirtyBricks.dirty} (${report.dirtyBricks.withHalo} with halo)`, "", ""]);
  if (report.uniformKeyChanged) rows.push(["uniform key", "changed (a restage, not a rebuild)", "", ""]);
  if (report.wallFrames_ms) {
    rows.push(["wall per frame", `first ${ms(report.wallFrames_ms[0])}, median ${ms(median(report.wallFrames_ms))}, max ${ms(Math.max(...report.wallFrames_ms))}`, "", ""]);
    const slowest = Object.entries(report.gpuPassFrames_ms?.[0] ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 4);
    if (slowest.length) rows.push(["first frame's slowest passes", slowest.map(([label, time]) => `${label} ${ms(time)}`).join("; "), "", ""]);
    rows.push(["converged frame", `${report.convergedFrame} (settled frames differ by ${report.noiseFloor?.frame ?? 0} px, ${report.noiseFloor?.rect ?? 0} in the edit's rectangle)`, "", ""]);
    rows.push(["lighting", report.lighting ? JSON.stringify(report.lighting) : "not reported", "", ""]);
  }
  console.log(table(rows));
  const stages = Object.entries(report.stages).sort((a, b) => b[1].ms - a[1].ms);
  if (stages.length > 0) {
    console.log("   host stages (inclusive; accept and frames):");
    console.log(table(stages.map(([name, stage]) => [`     ${name}`, `${ms(stage.ms)}`, `x${stage.calls}`,
      stage.settled_ms === undefined ? "" : `settled ${ms(stage.settled_ms)}`, stage.dispatches > 0 ? `${stage.dispatches} dispatches` : "",
      stage.uploadBytes > 0 ? bytes(stage.uploadBytes) : ""])));
  }
  const uploads = Object.entries(report.uploadBytes!.byLabel).sort((a, b) => b[1] - a[1]).slice(0, 8);
  console.log(`   uploads: ${bytes(report.uploadBytes!.accept)} in the accept, ${bytes(report.uploadBytes!.framesOverIdle)} in frames over idle. Largest by label (all frames):`);
  console.log(table(uploads.map(([label, total]) => [`     ${label}`, bytes(total)])));
  const passes = Object.entries(report.passes).sort((a, b) => b[1].direct + b[1].indirect - a[1].direct - a[1].indirect);
  console.log(`   compute passes: ${passes.length} labels, ${report.directDispatches} direct and ${report.indirectDispatches} indirect dispatches over ${report.frames} frame${report.frames === 1 ? "" : "s"}`
    + ` (an idle frame: ${report.idleFrame?.directDispatches ?? "?"} direct, ${report.idleFrame?.indirectDispatches ?? "?"} indirect, ${bytes(report.idleFrame?.uploadBytes ?? 0)} uploaded). Most dispatched:`);
  console.log(table(passes.slice(0, 10).map(([name, pass]) => [`     ${name}`, `${pass.direct} direct`, pass.indirect > 0 ? `${pass.indirect} INDIRECT` : ""])));
  if (report.indirectPasses.length > 0) console.log(`   indirect: ${report.indirectPasses.join("; ")}`);
  console.log(`   render passes: ${report.draws.renderPasses}, with ${report.draws.direct} direct and ${report.draws.indirect} indirect draws`);
  if (report.created.length > 0) console.log(`   created after build:\n${report.created.map((label) => `     ${label}`).join("\n")}`);
}

// ---------------------------------------------------------------------------

/** The command line that runs `edits` on the same scene and device from a fresh world. */
function rerunCommand(options: Options, edits: readonly EditName[]): string {
  const quote = (value: string) => /^[a-zA-Z0-9_./,:=+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;
  const arguments_ = ["node", "--import", "tsx", "tools/probe-edit-latency.ts", `--device=${options.device}`,
    `--scene=${options.scene}`, `--edit=${edits.join(",")}`, `--fluid=${options.fluid}`, `--size=${options.width}x${options.height}`,
    `--site=${options.site.join(",")}`, `--max-frames=${options.maxFrames}`, `--pen-size=${options.penSize}`, `--simulation-steps=${options.simulationSteps}`,
    `--pen-samples=${options.penSamples}`, `--pen-camera=${options.penCamera}`, `--sample-cache=${options.sampleCache}`,
    ...(options.penNode ? [`--pen-node=${options.penNode}`] : []), ...(options.penImages ? [`--pen-images=${options.penImages}`] : [])];
  return `${options.device === "dawn" ? "WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js " : ""}${arguments_.map(quote).join(" ")}`;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  const sourcePaths = ["tools/probe-edit-latency.ts", "lib/core/webgpu-renderer.ts", "lib/core/webgpu-sparse-scene-proxies.ts",
    "lib/core/webgpu-render-worker.ts", "lib/core/webgpu-render-worker-client.ts", "lib/interaction/runtime/host.ts",
    "lib/interaction/adapters/box3d/world.ts", "lib/methods/uniform/webgpu-uniform-reference.ts",
    "lib/interaction/adapters/solid-world/collision.ts", "lib/interaction/adapters/solid-world/overlap.ts",
    "lib/core/imported-assets/mesh-volume.wgsl.ts", "lib/svo/features/scene-publication/svo-field-program.ts",
    "lib/svo/pipeline/webgpu-svo-dry-scene.ts", "lib/svo/features/construction/webgpu-svo-sparse-bricks.ts",
    "lib/svo/features/radiance/webgpu-svo-live-derived-builder.ts", "lib/svo/features/scene-publication/svo-sculpt-cuts.ts",
    "lib/svo/features/scene-publication/svo-sculpt-instances.ts", "lib/svo/features/scene-publication/svo-sculpt-change-bounds.ts",
    "lib/svo/features/primary-visibility/svo-surface-mesh.ts", "lib/svo/features/shading/program.ts"];
  const sourceHashes = () => Object.fromEntries(sourcePaths.map(path => {
    try { return [path, createHash("sha256").update(readFileSync(new URL(`../${path}`, import.meta.url))).digest("hex")]; }
    catch { return [path, "unavailable"]; }
  }));
  const sourcesAtStart = sourceHashes();
  const authored = createScene(options.scene);
  // The worker runs in this Node realm, where browser-relative asset URLs
  // cannot be fetched. Preload the exact scene assets from public/ as the
  // other native rendering harnesses do; report their cost separately.
  const assetStart = performance.now();
  await registerSceneMeshAssets(authored.scene);
  const assetLoadMs = performance.now() - assetStart;
  const recorder = new Recorder();
  recorder.referenceSampleCache = options.sampleCache === "reference";
  let gpu: GPU;
  let adapterLabel = "recording stand-in device (nothing executed)";
  if (options.device === "dawn") {
    // Imported here so the fake mode never loads the Dawn module. The bootstrap
    // is used for its instance and its adapter checks; the device it makes is
    // let go, because the renderer requests its own, as it does in a browser.
    const { createDawnRenderDevice } = await import("./svo-dry-frame-harness");
    const dawn = await createDawnRenderDevice({ modulePath: process.env.WEBGPU_NODE_MODULE || undefined, label: "probe-edit-latency" });
    adapterLabel = [dawn.adapterInfo.vendor, dawn.adapterInfo.architecture, dawn.adapterInfo.description].filter(Boolean).join(" · ");
    dawn.device.destroy();
    gpu = dawn.gpu;
  } else gpu = createFakeGpu();
  const host = new BrowserHost(gpu, options.device === "dawn", options, (raw) => {
    if (options.device === "dawn") recorder.timePasses(raw);
    return recordingDevice(raw, recorder);
  });
  const validationErrors = host.validationErrors;
  const probe = new Probe(options, recorder, host, authored);
  console.log(`probe-edit-latency: scene ${options.scene}, device ${options.device} (${adapterLabel}), fluid ${options.fluid}, ${options.width}x${options.height}`);
  console.log(`scene asset preload: ${ms(assetLoadMs)} (excluded from world build and edit timings)`);
  await probe.start();
  if (recorder.referenceSampleCache && recorder.sampleCacheReplacements === 0) throw new Error("Sampler reference requested but no cached production shader was replaced.");
  const build = probe.worldBuild!;
  console.log(`world build: ${ms(build.ms)} (${build.builds.map((entry) => `${entry.kind}: ${entry.label}`).join("; ") || "no build status seen"}); `
    + `${Object.entries(build.creations).map(([kind, count]) => `${count} ${kind}s`).join(", ")}; ${bytes(build.uploadBytes)} uploaded; `
    + `${build.settleFrames} frames to quiescence after the first presented frame. Not part of any edit below.`);
  const slowestCompiles = [...recorder.compiles].sort((a, b) => b.ms - a.ms).slice(0, 5);
  if (slowestCompiles.length > 0) console.log(`slowest pipeline compiles: ${slowestCompiles.map((entry) => `${entry.label} ${ms(entry.ms)}`).join("; ")}`);
  console.log(`device features: ${[...(host.rawDevice?.features ?? [])].join(", ") || "none"}; canvas ${host.canvasSize}`);
  console.log(`buffer memory: ${bytes(recorder.bufferMemory.liveBytes)} live, ${bytes(recorder.bufferMemory.peakBytes)} peak (excludes textures and driver/compiler allocations)`);
  console.log(`texture memory: ${bytes(recorder.textureMemory.liveBytes)} live, ${bytes(recorder.textureMemory.peakBytes)} peak (texel storage asked for; a driver may pad it)`);
  const largest = [...Object.entries(recorder.bufferMemory.byLabel).map(([label, size]) => ["buffer", label, size] as const),
    ...Object.entries(recorder.textureMemory.byLabel).map(([label, size]) => ["texture", label, size] as const)]
    .sort((a, b) => b[2] - a[2]).slice(0, 16);
  console.log(`largest allocations:\n${largest.map(([kind, label, size]) => `  ${bytes(size).padStart(12)}  ${kind.padEnd(7)}  ${label}`).join("\n")}`);
  console.log(`bricks chosen by: ${probe.brickSelection}`);
  console.log(`stages timed: ${recorder.instrumented.join(", ") || "none"}`);
  if (recorder.notInstrumented.length > 0) console.log(`stages not reported (no such method today): ${recorder.notInstrumented.join(", ")}`);

  const reports: EditReport[] = [];
  for (const edit of options.edits) {
    const report = await probe.measure(edit);
    reports.push(report);
    printEdit(report);
  }

  const summary: string[][] = [["edit", "host accept", "uploads", "frames", "voxelization", "revisions", "created", "indirect", "rebuild", "FAIL"]];
  for (const report of reports) {
    if (report.skipped || report.refused) { summary.push([report.edit, outcome(report), "", "", "", "", "", "", "", ""]); continue; }
    const value = (measure: string) => report.budget.find((row) => row.measure === measure)?.value ?? "";
    summary.push([report.edit, value("host accept"), value("uploads"), String(report.frames), value("voxelization frames").split(" (")[0]!,
      value("scene revisions").split(" (")[0]!, String(Object.values(report.creations).reduce((sum, count) => sum + count, 0)), String(report.indirectDispatches),
      report.rebuilds.length > 0 || (report.rebuildKeysChanged?.length ?? 0) > 0 ? "REBUILD" : "none",
      String(report.budget.filter((row) => row.verdict === "FAIL").length)]);
  }
  console.log(`\n== summary — ${options.scene}, ${options.device}\n${table(summary)}`);
  const notRun = reports.filter((report) => report.skipped?.startsWith("not run")).map((report) => report.edit);
  if (notRun.length > 0) {
    console.log(`\nThe renderer stopped: ${host.failure}. Not run: ${notRun.join(", ")}. A fresh world runs them:\n  ${rerunCommand(options, notRun)}`);
  }
  if (validationErrors.length > 0) console.log(`\nWebGPU validation errors (${validationErrors.length}):\n${validationErrors.slice(0, 12).join("\n")}`);
  const simulation = options.simulationSteps ? probe.simulationReport() : undefined;
  const simulationPassed = simulation && !simulation.error && !simulation.errors.length && !host.failure && !validationErrors.length
    && !reports.some(report => report.refused || report.skipped)
    && simulation.checkpoints.every(checkpoint => checkpoint.after_s > checkpoint.before_s
      && checkpoint.completedReceipts >= checkpoint.requestedSteps);
  if (simulation) console.log(`\nsimulation ${simulationPassed ? "passed" : "FAILED"}: ${simulation.completedReceipts} completed receipts, ${simulation.completedTime_s.toFixed(6)} s; ${simulation.mode}. Edit timing budgets are reported separately.`);

  if (options.json) {
    writeFileSync(options.json, JSON.stringify({ options, sampleCacheReplacements: recorder.sampleCacheReplacements, sourceHashes: { start: sourcesAtStart, end: sourceHashes() }, adapter: adapterLabel, assetLoadMs, worldBuild: build, bufferMemory: recorder.bufferMemory, textureMemory: recorder.textureMemory, brickSelection: probe.brickSelection,
      stagesTimed: recorder.instrumented, stagesNotReported: recorder.notInstrumented, edits: reports, validationErrors,
      simulation: simulation ? { ...simulation, passed: simulationPassed } : undefined }, null, 2));
    console.log(`\nwrote ${options.json}`);
  }
  // The renderer keeps timers and a never-settling device-lost promise alive.
  process.exit(validationErrors.length > 0 || host.failure || (simulation && !simulationPassed) ? 1 : 0);
}

// An await the worker never answers leaves nothing on the event loop, and Node
// would exit 0 with half a report.
process.on("beforeExit", () => {
  console.error("probe-edit-latency stopped with a wait the worker never answered; the report above is incomplete.");
  process.exit(1);
});

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
