/**
 * A pool of CPU tracing workers and the progressive schedule they serve.
 *
 * ## Why progressive
 *
 * A field program is about 6.5 us a sample and a canopy pad is 280 voxels deep
 * at production's leaf, so a full frame is seconds however it is divided. A tool
 * whose control you are *dragging* cannot spend seconds before showing anything,
 * so every render runs twice: a quarter-scale pass that lands in a sixteenth of
 * the pixels and tells you the silhouette, then the full-resolution pass over it
 * in tiles that appear as they finish.
 *
 * ## Why tiles rather than one job per worker
 *
 * Cost per pixel varies by more than an order of magnitude across a frame — sky
 * is a bounding-sphere rejection, a grazing ray through a pad is hundreds of
 * voxel steps — so a static split by rows leaves most of the pool idle behind
 * one unlucky stripe. Tiles are handed out as workers free up, which is the
 * cheapest possible load balance and needs no measurement.
 */
import type { TerrainGrid } from "../terrain";
import type { SvoPrimitiveDescriptor } from "../svo-primitive-abi";
import type { ShapeLabCamera, ShapeLabShading } from "./trace";
import type { ShapeLabRequest, ShapeLabResponse } from "./worker";

export interface ShapeLabRenderRequest {
  /** Identity of the specimen, so a camera move does not re-send it. */
  readonly sceneId: string;
  readonly descriptors: readonly SvoPrimitiveDescriptor[];
  readonly colors: readonly (readonly [number, number, number])[];
  readonly ground?: TerrainGrid;
  readonly width: number;
  readonly height: number;
  readonly camera: ShapeLabCamera;
  readonly leaf_m?: number;
  readonly shading: ShapeLabShading;
  readonly showGround: boolean;
}

export interface ShapeLabTilePaint {
  readonly pass: number;
  /** Scale divisor this pass was traced at: 4 for the preview, 1 for the frame. */
  readonly scale: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8ClampedArray;
}

export interface ShapeLabProgress {
  readonly done: number;
  readonly total: number;
  readonly elapsedMs: number;
  readonly complete: boolean;
}

const TILE = 64;
/** The preview divisor. Four is sixteen times fewer pixels and still reads as a shape. */
const PREVIEW_SCALE = 4;

interface PendingTile {
  readonly pass: number;
  readonly options: {
    readonly width: number;
    readonly height: number;
    readonly camera: ShapeLabCamera;
    readonly leaf_m?: number;
    readonly shading: ShapeLabShading;
    readonly showGround: boolean;
    readonly tile: { x: number; y: number; width: number; height: number };
  };
}

export class ShapeLabPool {
  private readonly workers: Worker[] = [];
  private readonly idle: Worker[] = [];
  private queue: PendingTile[] = [];
  private generation = 0;
  private sceneId = "";
  private started = 0;
  private done = 0;
  private total = 0;

  constructor(
    size: number,
    private readonly onTile: (tile: ShapeLabTilePaint) => void,
    private readonly onProgress: (progress: ShapeLabProgress) => void,
    private readonly onError: (message: string) => void,
  ) {
    for (let index = 0; index < size; index += 1) {
      const worker = new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
        name: `shape-lab-trace-${index}`,
      });
      worker.addEventListener("message", (event: MessageEvent<ShapeLabResponse>) => this.receive(worker, event.data));
      worker.addEventListener("error", (event) => this.onError(event.message || "Shape lab worker failed"));
      this.workers.push(worker);
      this.idle.push(worker);
    }
  }

  dispose(): void {
    for (const worker of this.workers) worker.terminate();
    this.workers.length = 0;
    this.idle.length = 0;
    this.queue = [];
  }

  /**
   * Replace what is being drawn.
   *
   * In-flight tiles are abandoned by bumping the generation rather than by
   * asking a worker to stop: a tile is tens of milliseconds and a worker in the
   * middle of one cannot be interrupted anyway, so the cheap thing is to let it
   * finish into a result nobody reads.
   */
  render(request: ShapeLabRenderRequest): void {
    this.generation += 1;
    this.queue = [];
    this.done = 0;
    this.started = performance.now();
    if (request.sceneId !== this.sceneId) {
      this.sceneId = request.sceneId;
      const scene: ShapeLabRequest = {
        type: "scene",
        sceneId: request.sceneId,
        descriptors: request.descriptors,
        colors: request.colors,
        ...(request.ground ? { ground: request.ground } : {}),
      };
      for (const worker of this.workers) worker.postMessage(scene);
    }
    const shared = {
      camera: request.camera,
      shading: request.shading,
      showGround: request.showGround,
      ...(request.leaf_m === undefined ? {} : { leaf_m: request.leaf_m }),
    };
    const pending: PendingTile[] = [];
    const previewWidth = Math.max(1, Math.ceil(request.width / PREVIEW_SCALE));
    const previewHeight = Math.max(1, Math.ceil(request.height / PREVIEW_SCALE));
    for (let y = 0; y < previewHeight; y += TILE) {
      for (let x = 0; x < previewWidth; x += TILE) {
        pending.push({
          pass: 0,
          options: {
            ...shared, width: previewWidth, height: previewHeight,
            tile: { x, y, width: Math.min(TILE, previewWidth - x), height: Math.min(TILE, previewHeight - y) },
          },
        });
      }
    }
    for (let y = 0; y < request.height; y += TILE) {
      for (let x = 0; x < request.width; x += TILE) {
        pending.push({
          pass: 1,
          options: {
            ...shared, width: request.width, height: request.height,
            tile: { x, y, width: Math.min(TILE, request.width - x), height: Math.min(TILE, request.height - y) },
          },
        });
      }
    }
    this.queue = pending;
    this.total = pending.length;
    this.pump();
  }

  private pump(): void {
    while (this.idle.length > 0 && this.queue.length > 0) {
      const worker = this.idle.pop()!;
      const next = this.queue.shift()!;
      worker.postMessage({
        type: "tile",
        sceneId: this.sceneId,
        generation: this.generation,
        pass: next.pass,
        options: next.options,
      } satisfies ShapeLabRequest);
    }
  }

  private receive(worker: Worker, response: ShapeLabResponse): void {
    this.idle.push(worker);
    if (response.generation === this.generation) {
      if (response.type === "failed") {
        this.onError(response.message);
      } else if (response.type === "tile") {
        this.done += 1;
        this.onTile({
          pass: response.pass,
          scale: response.pass === 0 ? PREVIEW_SCALE : 1,
          x: response.tile.x,
          y: response.tile.y,
          width: response.tile.width,
          height: response.tile.height,
          rgba: new Uint8ClampedArray(response.rgba),
        });
        this.onProgress({
          done: this.done,
          total: this.total,
          elapsedMs: performance.now() - this.started,
          complete: this.done >= this.total,
        });
      }
    }
    this.pump();
  }
}

export const shapeLabPoolSize = (): number => Math.max(1, Math.min(8, (globalThis.navigator?.hardwareConcurrency ?? 4) - 1));
