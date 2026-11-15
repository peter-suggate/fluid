/**
 * Host-side decoding of the brick engine's state, for tests and harnesses.
 * It never runs on the step cadence. It reads the directory, the lists and a
 * pool's channels, and resolves a cell to its slot as the kernels do.
 */
import { BRICK_STRIDE, H, LIST, NONE, REC, SEAM, brickKeyCoord, cellLocal, type PoolName } from "./brick-abi";
import type { BrickEngine, Vec3 } from "./brick-engine";

export interface BrickSnapshot {
  readonly header: Uint32Array;
  readonly set: Uint32Array;
  readonly dir: Uint32Array;
  /** Live F bricks: key -> slot (from the directory, page word then child word). */
  readonly bricks: Map<number, number>;
  /** Live C pages: page key -> slot. */
  readonly pages: Map<number, number>;
  list(k: number): number[];
}

export async function snapshot(e: BrickEngine): Promise<BrickSnapshot> {
  await e.fence();
  const L = e.layout, set = await e.read("set"), dir = await e.read("dir");
  const bricks = new Map<number, number>(), pages = new Map<number, number>();
  const [wx, wy, wz] = L.window;
  for (let pz = 0; pz < wz; pz++) for (let py = 0; py < wy; py++) for (let px = 0; px < wx; px++) {
    const cs = dir[L.dir.page + L.pageIndex(px, py, pz)]!;
    if (cs === NONE || cs >= SEAM) continue;
    pages.set(dir[L.dir.crec + 32 * cs + REC.KEY]!, cs);
    for (let bl = 0; bl < 64; bl++) {
      const fs = dir[L.dir.child + 64 * cs + bl]!;
      if (fs === NONE || fs >= SEAM) continue;
      bricks.set(dir[L.dir.frec + 32 * fs + REC.KEY]!, fs);
    }
  }
  return {
    header: set.slice(0, 256), set, dir, bricks, pages,
    list: (k: number) => Array.from(set.subarray(L.set.lists[k]!, L.set.lists[k]! + Math.min(set[H.LIST_COUNT + k]!, L.set.listCap[k]!))),
  };
}

/** A channel of a pool for every live F brick: slot -> 512 values (f32 view unless u32). */
export async function readChannel(e: BrickEngine, pool: PoolName, channel: string, u32 = false): Promise<(slot: number) => Float32Array | Uint32Array> {
  const base = e.layout.base(pool, "F", channel), words = await e.read(pool, base, e.layout.cap.f * BRICK_STRIDE);
  const view = u32 ? words : new Float32Array(words.buffer);
  return (slot: number) => view.subarray(BRICK_STRIDE * slot, BRICK_STRIDE * slot + 512);
}

/** The slot and local word of cell (or vertex) c, or undefined where no F brick holds it. */
export function cellSlot(s: BrickSnapshot, c: Vec3): { slot: number; local: number } | undefined {
  if (c.some(v => v < 0)) return undefined;
  const key = ((c[0] >> 3) & 1023) | (((c[1] >> 3) & 1023) << 10) | (((c[2] >> 3) & 1023) << 20);
  const slot = s.bricks.get(key >>> 0);
  return slot === undefined ? undefined : { slot, local: cellLocal(c[0], c[1], c[2]) };
}

export const keyOf = brickKeyCoord;
export const LISTS = LIST;
