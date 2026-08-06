/**
 * One CPU tracing worker.
 *
 * The specimen is sent once and prepared once — canonicalising a descriptor
 * allocates and a tape has to be proved legal before it can be evaluated without
 * proof — and tiles are then asked for against it. That split is the whole
 * protocol: a `scene` message replaces what this worker is looking at, a `tile`
 * message asks for a rectangle of the current view of it.
 *
 * `sceneId` is separate from `generation` for the same reason: dragging a camera
 * changes the view a hundred times against one specimen, and re-sending a stone
 * set's 1 338 descriptors and a megabyte of ground on every pointer-move is the
 * whole cost of the interaction. A tile naming a scene this worker does not hold
 * is dropped rather than guessed at.
 *
 * Pixel buffers travel back as transfers rather than copies, and each is its own
 * tile rather than a frame, so a full pass moves the picture once.
 */
import type { SvoPrimitiveDescriptor } from "../svo-primitive-abi";
import type { TerrainGrid } from "../terrain";
import {
  prepareShapeLabScene,
  traceShapeLabTile,
  type ShapeLabScene,
  type ShapeLabTraceOptions,
} from "./trace";

export interface ShapeLabSceneMessage {
  readonly type: "scene";
  readonly sceneId: string;
  readonly descriptors: readonly SvoPrimitiveDescriptor[];
  readonly colors: readonly (readonly [number, number, number])[];
  readonly ground?: TerrainGrid;
}

export interface ShapeLabTileMessage {
  readonly type: "tile";
  readonly sceneId: string;
  readonly generation: number;
  readonly pass: number;
  readonly options: ShapeLabTraceOptions;
}

export type ShapeLabRequest = ShapeLabSceneMessage | ShapeLabTileMessage;

export interface ShapeLabTileResult {
  readonly type: "tile";
  readonly generation: number;
  readonly pass: number;
  readonly tile: { readonly x: number; readonly y: number; readonly width: number; readonly height: number };
  readonly rgba: ArrayBuffer;
  readonly elapsedMs: number;
}

export interface ShapeLabDropped {
  readonly type: "dropped";
  readonly generation: number;
}

export interface ShapeLabFailure {
  readonly type: "failed";
  readonly generation: number;
  readonly message: string;
}

export type ShapeLabResponse = ShapeLabTileResult | ShapeLabDropped | ShapeLabFailure;

let scene: ShapeLabScene | undefined;
let sceneId = "";

const reply = (message: ShapeLabResponse, transfer: Transferable[] = []): void => {
  (self as unknown as Worker).postMessage(message, transfer);
};

self.addEventListener("message", (event: MessageEvent<ShapeLabRequest>) => {
  const message = event.data;
  try {
    if (message.type === "scene") {
      scene = prepareShapeLabScene(message.descriptors, message.colors, message.ground);
      sceneId = message.sceneId;
      return;
    }
    if (!scene || sceneId !== message.sceneId) {
      reply({ type: "dropped", generation: message.generation });
      return;
    }
    const started = performance.now();
    const options = message.options;
    const tile = options.tile ?? { x: 0, y: 0, width: options.width, height: options.height };
    const traced = traceShapeLabTile(scene, { ...options, tile });
    const buffer = traced.rgba.buffer as ArrayBuffer;
    reply({
      type: "tile",
      generation: message.generation,
      pass: message.pass,
      tile,
      rgba: buffer,
      elapsedMs: performance.now() - started,
    }, [buffer]);
  } catch (error) {
    reply({
      type: "failed",
      generation: message.type === "tile" ? message.generation : -1,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
