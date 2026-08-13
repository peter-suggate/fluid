import { editorFluidLattice, fluidBrickCenter, fluidBrickIndexAt } from "./editor-fluid";
import { cloneScene, type SceneDescription, type Vec3 } from "./model";

/**
 * Painted water rides its own key, not a `scene.*` path.
 *
 * A seed's metres are already thrown away by everything that reads them:
 * `seedCell` floors the point to a finest cell, every consumer floors that to a
 * brick, and `seedBrickCoordinates` deduplicates the result. The brick index is
 * the value; the position is one of the many points inside it. Serializing the
 * document's own array therefore spent seventeen significant digits per axis on
 * a small integer — a 256-brick paint measured 81 characters per seed, ~20.7 kB
 * in one key, which reloads as an HTTP 431 once the request line passes Node's
 * 16 kB header ceiling. The same paint is 172 characters here.
 *
 * The encoding is the brick occupancy itself, in one of two modes chosen by
 * whichever is shorter:
 *
 *  - `b` a bitset over the seeded bricks' own bounding box, base64. Costs
 *    one character per six bricks of *box*, so it wins on the dense blobs a
 *    brush actually paints and loses on a few bricks flung across the tank.
 *  - `d` ascending gaps between linear brick indices, base36. Costs a few
 *    characters per *seed* regardless of how far apart they are, which is the
 *    sparse case the bitset cannot cover.
 *
 * Both are prefixed by the brick-grid dimensions the paint was authored
 * against, so a link that also re-authors the lattice — `scene.voxelDomain` is
 * a query key, and a finer cell makes every brick smaller — puts the water back
 * in the same *place* rather than at the same index. Like `refinementRegionsToQuery`,
 * this makes the key an instruction about the domain rather than a document value.
 */

const FIELD = "_";
const RECORD = "*";

/**
 * Beyond this the key is dropped rather than written, on the same contract a
 * sculpted terrain grid already takes: some authoring is a scene-library
 * document, not a link. Generous enough for any brush stroke — roughly 48,000
 * bricks of bounding box, a solid 36³ — and far below the header ceiling that
 * turns an over-long link into a reload that fails.
 */
const SEEDS_QUERY_BUDGET = 8000;

interface SeedBricks {
  /** Brick-grid dimensions the indices below are addressed in. */
  readonly bricks: readonly [number, number, number];
  /** Distinct seeded bricks, ascending by linear index. */
  readonly linear: readonly number[];
}

/**
 * Base64 over the two extra characters `x-www-form-urlencoded` leaves alone
 * that this format has not already spent.
 *
 * Not base64url's own `-_` pair: `_` is the field separator, and a payload that
 * can contain it makes the record it sits in ambiguous to read back. `.` is
 * untouched by `URLSearchParams.toString()` for the same reason `*` and `-`
 * are, so the value still reaches the address bar unescaped.
 */
function base64Compact(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, ".").replace(/=+$/, "");
}

function fromBase64Compact(text: string): Uint8Array | undefined {
  if (text === "" || !/^[A-Za-z0-9\-.]+$/.test(text)) return undefined;
  try {
    const binary = atob(text.replace(/-/g, "+").replace(/\./g, "/"));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
  } catch { return undefined; }
}

/** The seeded bricks of a document, deduplicated, in the lattice it was painted on. */
function seedBricks(scene: SceneDescription): SeedBricks | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds) return undefined;
  const lattice = editorFluidLattice(scene);
  const { x: bx, y: by, z: bz } = lattice.bricks;
  const unique = new Set<number>();
  for (const seed of seeds) {
    const index = fluidBrickIndexAt(lattice, seed);
    // A seed outside the tank wets nothing and cannot be painted back; the
    // editor's own `seedsByBrick` drops it on exactly this test.
    if (index) unique.add((index.z * by + index.y) * bx + index.x);
  }
  return { bricks: [bx, by, bz], linear: [...unique].sort((left, right) => left - right) };
}

function bitsetPayload(bricks: SeedBricks): string {
  const [bx, by] = bricks.bricks;
  const cells = bricks.linear.map((linear) => {
    const x = linear % bx;
    const y = Math.floor(linear / bx) % by;
    return [x, y, Math.floor(linear / (bx * by))] as const;
  });
  const low = [0, 1, 2].map((axis) => Math.min(...cells.map((cell) => cell[axis]))) as [number, number, number];
  const extent = [0, 1, 2].map((axis) =>
    Math.max(...cells.map((cell) => cell[axis])) - low[axis] + 1) as [number, number, number];
  const bits = new Uint8Array(Math.ceil((extent[0] * extent[1] * extent[2]) / 8));
  for (const cell of cells) {
    const n = ((cell[2] - low[2]) * extent[1] + (cell[1] - low[1])) * extent[0] + (cell[0] - low[0]);
    bits[n >> 3]! |= 1 << (n & 7);
  }
  return ["b", ...low, ...extent, base64Compact(bits)].join(FIELD);
}

function deltaPayload(bricks: SeedBricks): string {
  let previous = 0;
  const gaps = bricks.linear.map((linear) => {
    const gap = linear - previous;
    previous = linear;
    return gap.toString(36);
  });
  return ["d", ...gaps].join(FIELD);
}

/**
 * The seeded bricks as a query value, or `""` for a document with none.
 *
 * The empty string is meaningful and is still written when it differs from the
 * preset's own paint: hydration must be able to tell "this link removed the
 * authored water" from "this link says nothing about water", and only a present
 * key can say the first.
 */
export function sceneSeedsQuery(scene: SceneDescription): string {
  const bricks = seedBricks(scene);
  if (!bricks || bricks.linear.length === 0) return "";
  const payloads = [bitsetPayload(bricks), deltaPayload(bricks)];
  const shortest = payloads.reduce((left, right) => (right.length < left.length ? right : left));
  const encoded = [bricks.bricks.join(FIELD), shortest].join(RECORD);
  return encoded.length > SEEDS_QUERY_BUDGET ? "" : encoded;
}

function integers(text: string, count: number): number[] | undefined {
  const parts = text.split(FIELD);
  if (parts.length !== count) return undefined;
  const values = parts.map((part) => (/^-?\d+$/.test(part) ? Number(part) : Number.NaN));
  return values.some((value) => !Number.isFinite(value)) ? undefined : values;
}

function decodeBitset(payload: readonly string[], bricks: readonly [number, number, number]): number[] | undefined {
  if (payload.length !== 7) return undefined;
  const box = integers(payload.slice(0, 6).join(FIELD), 6);
  if (!box || box.some((value) => value < 0)) return undefined;
  const [ox, oy, oz, ex, ey, ez] = box as [number, number, number, number, number, number];
  if (ex <= 0 || ey <= 0 || ez <= 0) return undefined;
  const bits = fromBase64Compact(payload[6] ?? "");
  if (!bits || bits.length < Math.ceil((ex * ey * ez) / 8)) return undefined;
  const linear: number[] = [];
  for (let n = 0; n < ex * ey * ez; n += 1) {
    if ((bits[n >> 3]! & (1 << (n & 7))) === 0) continue;
    const x = ox + (n % ex);
    const y = oy + (Math.floor(n / ex) % ey);
    const z = oz + Math.floor(n / (ex * ey));
    linear.push((z * bricks[1] + y) * bricks[0] + x);
  }
  return linear;
}

function decodeDeltas(payload: readonly string[]): number[] | undefined {
  const linear: number[] = [];
  let previous = 0;
  for (const gap of payload) {
    if (!/^[0-9a-z]+$/.test(gap)) return undefined;
    const value = Number.parseInt(gap, 36);
    if (!Number.isFinite(value) || value < 0) return undefined;
    previous += value;
    linear.push(previous);
  }
  return linear;
}

/**
 * Re-address a brick index onto this scene's own brick grid.
 *
 * Through the brick's centre rather than its corner, so an index survives a
 * lattice that halves the cell size as the two bricks that cover the same
 * space's middle rather than drifting to one edge. On a link opened at the
 * lattice it was written from this is the identity.
 */
function rescale(index: number, from: number, to: number): number {
  if (from === to) return index;
  return Math.min(to - 1, Math.max(0, Math.floor(((index + 0.5) * to) / from)));
}

/**
 * Read painted water back against *this* scene's container and lattice.
 *
 * Seeds are emitted at brick centres, which is where the brush authors them, so
 * a link opened on its own scene round-trips to the same document. Anything
 * malformed leaves the scene untouched: a link is external input, and a bad
 * payload must not cost the reader the water the preset authored.
 */
export function withSceneSeedsFromQuery(scene: SceneDescription, raw: string): SceneDescription {
  const next = cloneScene(scene);
  if (raw === "") {
    // Not `[]` — `validateScene` rejects an empty seed array, and the paint
    // tool's own erase-to-nothing drops the field for the same reason.
    delete next.fluid.initialBrickSeeds_m;
    return next;
  }
  const [dimensions, body, ...extra] = raw.split(RECORD);
  if (extra.length > 0) return scene;
  const [mode, ...payload] = (body ?? "").split(FIELD);
  const source = integers(dimensions ?? "", 3);
  if (!source || source.some((value) => value <= 0)) return scene;
  const lattice = editorFluidLattice(next);
  const target: [number, number, number] = [lattice.bricks.x, lattice.bricks.y, lattice.bricks.z];
  const linear = mode === "b" ? decodeBitset(payload, source as [number, number, number])
    : mode === "d" ? decodeDeltas(payload)
      : undefined;
  if (!linear) return scene;

  const seeds = new Map<string, Vec3>();
  for (const value of linear) {
    const x = value % source[0]!;
    const y = Math.floor(value / source[0]!) % source[1]!;
    const z = Math.floor(value / (source[0]! * source[1]!));
    if (z >= source[2]!) continue;
    const index = {
      x: rescale(x, source[0]!, target[0]),
      y: rescale(y, source[1]!, target[1]),
      z: rescale(z, source[2]!, target[2]),
    };
    const centre = fluidBrickCenter(lattice, index);
    if (!insideSolverBounds(next, centre)) continue;
    seeds.set(`${index.x}:${index.y}:${index.z}`, centre);
  }
  if (seeds.size === 0) return scene;
  next.fluid.initialBrickSeeds_m = [...seeds.values()];
  return next;
}

/**
 * `editorFluidLattice` rounds its brick count up, so the last brick on an axis
 * whose cell count is not a multiple of eight has its centre outside the tank —
 * the bound `validateScene` holds a seed to. Decoding one would produce a
 * document that will not save, so it is dropped instead.
 */
function insideSolverBounds(scene: SceneDescription, point: Vec3): boolean {
  const c = scene.container;
  return point.x >= -c.width_m / 2 && point.x < c.width_m / 2
    && point.y >= 0 && point.y < c.height_m
    && point.z >= -c.depth_m / 2 && point.z < c.depth_m / 2;
}
