/** Surface dent probe for Narrow-band FLIP on a corner dam break: advances
 * the scene headless and, at the listed steps, reads the stored surface
 * (vertex phi, tile words, volume) back through the pool's page table of that
 * moment and reports how far each free face of the column is from smooth.
 * Simulation only, no rendering. Browser GPU must be off.
 *
 * WEBGPU_NODE_MODULE=<repo>/node_modules/webgpu/index.js node --import tsx tools/probe-surface-dents.ts <scene> <steps> <read,read,...> [--dt=0.016666666666666666] [--maps] [--worst=12] [--variants=<json list of method values>] [--stages=<step,step,...>] [--dump=x,y,z] [--vtrace=x,y,z] [--lines=<face>/<i|j>/<at>/<from>/<to>,...] [--vslab] [--spectra]
 *
 * A face is a height map of the outermost phi crossing along one axis over
 * the lattice's vertices: `top` along y, `sideX` along x, `sideZ` along z,
 * each over the interior of the column's own face (ten cells in from its
 * edges as the column stood at step 0, and ten under its top as it is now;
 * a side face starts twelve cells over the floor), `bodyX` and `bodyZ` the
 * side faces from 32 cells over the floor (above the foot, where the column
 * turns into the sheet and a height map along x or z is ill-conditioned),
 * `rim` the top map three to ten cells in from the column's free edges, and
 * `sheet` the top map outside the column's footprint. A side face means
 * nothing once the column has slumped past its box (about step 12 of
 * high-resolution-dam-break). Per face, in cells:
 *   rough  |m - mean of the four neighbours two vertices away|: rms and max,
 *          and `dents`, the vertices where it passes 0.2 cells. About zero on
 *          a plane or a gently curved face; a dent of depth a and one tile
 *          across gives about a/2.
 *   fit    residual of the face against its own least-squares quadratic: rms
 *          and max. The spread of a face that should still be flat.
 * Then the worst vertices by rough, at least six apart: the vertex, the cell
 * below it on the face's axis, its tile, the tile's place in its page, the
 * page and its block, the tile's width (h or 4h) and those of its six
 * neighbours. --maps prints each face as characters (a row a vertex line):
 * '.' within 0.15 cells of the 9x9 median, digits a dent in tenths of a
 * cell, letters a bump (a = 0.1 ... ), and beside it the width of the tile
 * each crossing lies in (h: an h tile, C: a 4h tile).
 * --variants runs one solver per entry, its values over the method's
 * defaults. --stages names steps whose phi is also read as each surface
 * stage of the frame leaves it (head, advect, reconstruct, retire,
 * redistance, coarsePhi): the same numbers per stage, and the end's worst
 * vertices followed through the stages as crossing/rough. With it, --dump
 * prints a slab about one cell (tile widths before and after the relayout,
 * phi at head and after the advect, the velocity the advect read) and
 * --vtrace the velocity each velocity stage leaves at a column of cells
 * (head, momentum, p2g, forces, toSimulation, band, extended), in cells a
 * step; a cell of a 4h tile reads zero there.
 * --lines=<face>/<i|j>/<at>/<from>/<to>,... prints height profiles at every
 * read and stage: the face's map along the other index at the fixed one
 * (top: i x, j z; sideX: i y, j z; sideZ: i x, j y), and the width of the
 * tile at each crossing. --vslab (with --stages and --dump) prints the slab's
 * raw velocity texels as each velocity stage leaves them (forces,
 * toSimulation, band, extended), the extension the next step starts from
 * (extension, and headExtension as the relayout finds it) and what the
 * relayout's remap makes of it (remapHolder, remapExtension) and of the
 * projected field (remapPhysical). A 4h tile holds each face once, at its
 * anchor. --spectra prints, for the sheet (the top map outside the column's
 * footprint), the foot of the side face and the whole top map, along each
 * index: the amplitude by period of 32-sample windows (quadratic detrend,
 * Hann), their rms, and the peak to peak of the detrended mean folded on the
 * tile (4) and page (32) lattices. A stripe locked to a lattice shows in its
 * fold; one that is not shows in the spectrum alone.
 * --criteria=<x0>-<x1>/<y0>-<y1>/<z> (tile coordinates) turns the layout
 * views on (the census then records why each tile is band) and prints, at
 * every read, what the head census of that step scored: over every stored
 * tile, the crossing tiles by the width the build gave them, the tiles each
 * criterion triggered, required, held and dropped by the budget, the
 * budget's cutoff, the h tiles by band reason, for the crossing tiles left
 * at 4h each criterion's scores, and the requests the budget kept and
 * dropped (by the tile's width, by whether it lies under the column as it
 * stood, and by its top criterion); then the slab's tiles one by one. A
 * score is over its threshold: 1 triggers.
 * --foot=<z>/<x0>-<x1>/<y0>-<y1> (cells) prints, at every printed read, the
 * slab one cell thick at z: each tile's width, target heat, the samples'
 * support heat (theta) and their own (surface heat, over 1 is authority to
 * erase) and whether it was activated this step; then per cell the liquid
 * fraction, the samples and the samples of the outer layer (depth over -1);
 * then per cell row the front (the last cell at least half liquid), the
 * farthest sample, and over the four cells behind the front the samples, the
 * outer ones, their mean heat, their own x velocity and the grid's (cells a
 * step) and the rms of the difference.
 * --ledger prints, for every stage capture and every printed read, the
 * liquid the phi holds in cells (a cell's share is 0.5 less the mean of its
 * eight vertices' phi in cells, held in 0 to 1): the total, the part within
 * 16 cells of the far walls, the part under y = 2, the cells partly full, and
 * the total by 8-cell bands of x, y and z.
 * A variant's `_flags` key is not a method value: it sets experiment bits
 * the probe patches into the sample kernels of every arm (1: samples take
 * the grid velocity, no FLIP part; 2: the outer layer is replenished at every
 * step; 4: an activated tile's outer layer is not seeded; 8: the collar's
 * samples erase as a hot tile's do; 16: the outer layer seeds on a 4h owner
 * too; 32: the collar takes its neighbour's full heat). */
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { readMixedTexture, readMixedTileWords } from "../tests/helpers/uniform-mixed-native-fields";

// A tree from before the page pool has no lattice address: its fields are dense boxes and its tile words are by box index.
const lattice = await import("../lib/methods/uniform/uniform-lattice-address").catch(() => undefined);
const method = uniformNarrowBandMethod;
const [sceneId, stepsText, readsText] = process.argv.slice(2).filter(a => !a.startsWith("--"));
const steps = Number(stepsText), reads = new Set((readsText ?? "").split(",").filter(Boolean).map(Number));
assert.ok(sceneId && Number.isInteger(steps) && steps >= 0, "usage: probe-surface-dents.ts <scene> <steps> <read,read,...> [--dt=] [--maps] [--worst=]");
const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.split("=")[1];
const dt = Number(option("dt") ?? 1 / 60), maps = process.argv.includes("--maps"), worstCount = Number(option("worst") ?? 12);
// Method values over the app's defaults, one solver per entry: --variants='[{},{"adaptiveBudgetPercent":100}]'.
const variants = JSON.parse(option("variants") ?? "[{}]") as Record<string, string | number>[];
// Steps whose phi is also read as each surface stage leaves it: --stages=5,6.
const stageSteps = new Set((option("stages") ?? "").split(",").filter(Boolean).map(Number));
// With --stages, a slab of the fields about one cell: --dump=81,27,65.
const dump = option("dump")?.split(",").map(Number);
// With --stages, the velocity each velocity stage leaves at a column of cells: --vtrace=80,27,65.
const vtrace = option("vtrace")?.split(",").map(Number);
// Height profiles at every read and stage: --lines=top/i/10/60/84 is face `top` at i = 10 along j = 60..84 (top: i x, j z; sideX: i y, j z; sideZ: i x, j y).
const lines = (option("lines") ?? "").split(",").filter(Boolean).map(text => { const [face, fixed, at, from, to] = text.split("/"); return { face: face!, fixed: fixed!, at: Number(at), from: Number(from), to: Number(to) }; });
// With --stages and --dump, the slab's velocity as each velocity stage leaves it.
const vslab = process.argv.includes("--vslab");
// The amplitude spectrum of the sheet and of the foot of the side face, and their means folded on the tile and page lattices.
const spectra = process.argv.includes("--spectra");
// What the head census scored: --criteria=17-31/0-3/10 is tiles x 17..31, y 0..3 at z 10.
const criteria = option("criteria")?.split("/").map(part => part.split("-").map(Number));
const CRITERIA = ["shape", "thin", "strain", "rotation", "impact", "approach"], REASONS = ["skipped", "traced", "crossing", "closure", "travel", "join", "source", "solid", "bulk"];
// The samples and the heat on a slab one cell thick: --foot=40/84-127/0-27 is z 40, x 84..127, y 0..27.
const foot = option("foot")?.split("/").map(part => part.split("-").map(Number));
// The liquid each stage's phi holds.
const ledger = process.argv.includes("--ledger");
// Experiment bits read from a spare parameter word (activity.y), patched into the sample kernels: each [old, new, modules it must be found in at least].
const FLAG = "u32(params.activity.y)";
const PATCHES: [string, string][] = [
  ["v=mix(pic,flip,params.settings.x);", `v=mix(pic,flip,select(params.settings.x,0.0,(${FLAG}&1u)!=0u));`],
  ["if(!initial&&!activated&&distance>-1.0&&nbSourcePhi(q,1.0)>=0.0){return vec4f(-1,0,0,0);}", `if(!initial&&!activated&&distance>-1.0&&nbSourcePhi(q,1.0)>=0.0&&(${FLAG}&2u)==0u){return vec4f(-1,0,0,0);}`],
  [",activated);if(site.x<0.0)", `,activated&&(${FLAG}&4u)==0u);if(site.x<0.0)`],
  ["nbSurfaceThetaFrom(home,vec3f(q)),particle>bulk);", `nbSurfaceThetaFrom(home,vec3f(q)),particle>bulk&&(${FLAG}&8u)==0u);`],
  ["if(distance>-1.0&&umOwnerAt(vec3i(c)).width!=1u){", `if(distance>-1.0&&umOwnerAt(vec3i(c)).width!=1u&&(${FLAG}&16u)==0u){`],
  ["value*select(0.5,1.0,x==0&&y==0&&z==0)", `value*select(select(0.5,1.0,(${FLAG}&32u)!=0u||((${FLAG}&64u)!=0u&&max(abs(x),max(abs(y),abs(z)))<=1)),1.0,x==0&&y==0&&z==0)`],
  ["var heat=0.0;\n  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){\n   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}", `var heat=0.0;\n  let wide=i32((${FLAG}&64u)!=0u);for(var z=-1-wide;z<=1+wide;z++){for(var y=-1-wide;y<=1+wide;y++){for(var x=-1-wide;x<=1+wide;x++){\n   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}`],
];
// Experiment bits compiled into an arm's own kernels (the module has no spare word): each [bit, old, new].
const COMPILED: [number, string, string][] = [
  [128, "if(!(abs(value)<2.0*w)){return umSurfaceTrace(p);}", "if(!(abs(value)<max(2.0*w,8.0))){return umSurfaceTrace(p);}"],
  [128, "if(reach>2.0*w){offset*=2.0*w/reach;}", "if(reach>max(2.0*w,8.0)){offset*=max(2.0*w,8.0)/reach;}"],
];
let armFlags = 0;
const patched = PATCHES.map(() => 0), compiled = COMPILED.map(() => 0), flagged = variants.some(v => v._flags);
const scene = structuredClone(sceneDocument(getSceneDefinition(sceneId)));
scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };

type Map2 = { w: number; h: number; v: Float32Array };
const median = (values: number[]) => { values.sort((a, b) => a - b); return values[values.length >> 1]!; };

let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  // With experiment bits asked for, every arm runs the same patched kernels: bit clear is the kernel as it stands.
  const source = !flagged ? raw : new Proxy(raw, { get: (target, key) => {
    if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => {
      let code = d.code; PATCHES.forEach(([from, to], k) => { const parts = code.split(from); if (parts.length > 1) { patched[k]! += parts.length - 1; code = parts.join(to); } });
      COMPILED.forEach(([bit, from, to], k) => { const parts = code.split(from); if ((armFlags & bit) && parts.length > 1) { compiled[k]! += parts.length - 1; code = parts.join(to); } });
      return target.createShaderModule({ ...d, code });
    };
    const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
  device = managedGPUDevice(source, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  for (const { _flags: flags = 0, ...variant } of variants) {
  const started = performance.now();
  if (variants.length > 1 || Object.keys(variant).length || flags) console.log(`VARIANT ${JSON.stringify({ ...variant, _flags: flags })}`);
  armFlags = Number(flags); compiled.fill(0);
  solver = await method.createSolverAsync!(device, scene, "balanced", { ...method.appDefaults, timeStep: "scene", ...variant }, undefined, () => {}) as WebGPUUniformReferenceSolver;
  if (flagged) {
    assert.ok(patched.every(n => n > 0), `experiment patches found ${patched.join(",")} times`);
    assert.ok(COMPILED.every(([bit], k) => !(armFlags & bit) || compiled[k]! > 0), `compiled experiment patches found ${compiled.join(",")} times`);
    // The head's activity writes its parameter words every step; the bits follow it into the spare one.
    const nb = (solver as unknown as { mixedFrame: { narrowBandFlip: { activity: (...a: unknown[]) => void; params: GPUBuffer } } }).mixedFrame.narrowBandFlip, activity = nb.activity.bind(nb), word = Float32Array.of(Number(flags));
    nb.activity = (...a) => { activity(...a); device!.queue.writeBuffer(nb.params, 36, word); };
    device.queue.writeBuffer(nb.params, 36, word);
  }
  const { nx, ny, nz } = solver.info, H = scene.container.width_m / nx;
  const address = lattice ? (solver as unknown as { address: UniformLatticeAddress }).address : undefined;
  const frame = (solver as unknown as { mixedFrame: { pagePool: { state: GPUBuffer; layout: { state: { words: number } } }; ownership: { capacity: { fineTiles: number } } } }).mixedFrame;
  console.log(`${sceneId}: ${nx}x${ny}x${nz}, h ${H}, dt ${dt}, created in ${((performance.now() - started) / 1000).toFixed(1)} s, ${address ? `pool of ${address.blocks} blocks` : "dense fields"}, h capacity ${frame.ownership.capacity.fineTiles}`);

  if (criteria) solver.setLayoutViewsEnabled(true);
  /** Words of a census buffer as they are on the GPU now. */
  const censusWords = async (view: { buffer: GPUBuffer; offset: number }, words: number) => {
    const read = device!.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(view.buffer, view.offset, read, 0, 4 * words); device!.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice();
    } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
  };
  /** The pool's state as it is on the GPU now. */
  const poolState = async () => {
    const words = frame.pagePool.layout.state.words, read = device!.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(frame.pagePool.state, 0, read, 0, 4 * words); device!.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice();
    } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
  };

  let column: [number, number, number] | undefined;
  const report = async (step: number, print: boolean, stages: Stage[] = []) => {
    await solver!.awaitFrameCompletion();
    // Host readback places texels by the host's page table: the build's. Hold the GPU's in its place for the reads.
    const built = address?.seed.state.slice(), live = address ? await poolState() : Uint32Array.of(0, 0);
    if (address) { assert.equal(live.length, built!.length); address.seed.state.set(live); }
    let last: Float32Array, tiles: Uint32Array, volume: Float32Array; const staged: Float32Array[] = [];
    try {
      [last, tiles, volume] = await Promise.all([readMixedTexture(device!, solver!.vertexPhiTexture!), readMixedTileWords(device!, solver), readMixedTexture(device!, solver!.volumeTexture)]);
      for (const stage of stages) staged.push(await stage.done());
    } catch (error) { address?.seed.state.set(built!); throw error; }
    try {
    const wordAt = (x: number, y: number, z: number) => tiles[address ? lattice!.uniformTileKey(address, x >> 2, y >> 2, z >> 2) : (x >> 2) + (nx / 4) * ((y >> 2) + (ny / 4) * (z >> 2))]!;
    const inBox = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
    const pageBlock = (x: number, y: number, z: number) => address ? address.seed.table[(x >> 5) | ((y >> 5) << 6) | ((z >> 5) << 12)]! : 1;
    const stored = (x: number, y: number, z: number) => { const b = pageBlock(x, y, z); return !address || (b >= 1 && b < address.blocks - 1); };
    const fineAt = (x: number, y: number, z: number) => inBox(x, y, z) && stored(x, y, z) && (wordAt(x, y, z) & 0x80000000) !== 0;
    const kindAt = (x: number, y: number, z: number) => !inBox(x, y, z) ? "#" : !stored(x, y, z) ? "_" : (wordAt(x, y, z) & 0x80000000) ? "h" : "C";
    /** The dent numbers of one phi field; returns the crossing and roughness at a map entry of a face. */
    const analyse = (label: string, phi: Float32Array, print: boolean) => {
    const raw = (a: number, b: number, c: number) => phi[a + (nx + 1) * (b + (ny + 1) * c)]!;
    /** Logical vertex phi: a vertex of a 4h tile is its corners' trilinear interpolant. */
    const vertex = (x: number, y: number, z: number) => {
      let best = Infinity, origin: [number, number, number] | undefined;
      for (let k = 0; k < 8; k++) {
        const c = [x - (k & 1), y - ((k >> 1) & 1), z - ((k >> 2) & 1)];
        if (!inBox(c[0]!, c[1]!, c[2]!) || !stored(c[0]!, c[1]!, c[2]!)) continue;
        const word = wordAt(c[0]!, c[1]!, c[2]!);
        if (!(word & 0x80000000) && (word & 0x3fffffff) < best) { best = word & 0x3fffffff; origin = c.map(v => v - v % 4) as [number, number, number]; }
      }
      if (!origin) return raw(x, y, z);
      const f = [(x - origin[0]) / 4, (y - origin[1]) / 4, (z - origin[2]) / 4]; let value = 0;
      for (let k = 0; k < 8; k++) { const c = [k & 1, (k >> 1) & 1, (k >> 2) & 1], w = c.reduce((s, v, a) => s * (v ? f[a]! : 1 - f[a]!), 1); if (w) value += w * raw(origin[0] + 4 * c[0]!, origin[1] + 4 * c[1]!, origin[2] + 4 * c[2]!); }
      return value;
    };
    const logical = new Float32Array((nx + 1) * (ny + 1) * (nz + 1));
    for (let z = 0; z <= nz; z++) for (let y = 0; y <= ny; y++) for (let x = 0; x <= nx; x++) logical[x + (nx + 1) * (y + (ny + 1) * z)] = vertex(x, y, z);
    const at = (p: readonly number[]) => logical[p[0]! + (nx + 1) * (p[1]! + (ny + 1) * p[2]!)]!;
    if (ledger && print) {
      let total = 0, far = 0, low = 0, partial = 0; const bands = [new Float64Array(nx >> 3), new Float64Array(ny >> 3), new Float64Array(nz >> 3)], W = nx + 1, S = W * (ny + 1);
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        if (!stored(x, y, z)) continue;
        const i = x + W * y + S * z, mean = (logical[i]! + logical[i + 1]! + logical[i + W]! + logical[i + W + 1]! + logical[i + S]! + logical[i + S + 1]! + logical[i + S + W]! + logical[i + S + W + 1]!) / (8 * H);
        const share = Math.min(1, Math.max(0, 0.5 - mean)); if (share <= 0) continue;
        total += share; if (share < 1) partial++; if (x >= nx - 16 || z >= nz - 16) far += share; if (y < 2) low += share;
        bands[0]![x >> 3]! += share; bands[1]![y >> 3]! += share; bands[2]![z >> 3]! += share;
      }
      console.log(`LEDGER ${label}: liquid ${total.toFixed(0)} cells, by the far walls ${far.toFixed(0)}, under y 2 ${low.toFixed(0)}, cells partly full ${partial}`);
      "xyz".split("").forEach((a, k) => console.log(`LEDGER ${label} by ${a}: ${Array.from(bands[k]!, v => v.toFixed(0)).join(" ")}`));
    }
    /** The outermost crossing along `axis` at every vertex of the other two axes (u: the lower, v: the higher); NaN: none. */
    const crossing = (axis: number): Map2 => {
      const n = [nx, ny, nz], u = axis === 0 ? 1 : 0, v = axis === 2 ? 1 : 2, w = n[u]! + 1, h = n[v]! + 1, out = new Float32Array(w * h).fill(NaN), p = [0, 0, 0];
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        p[u] = i; p[v] = j; p[axis] = n[axis]!;
        let above = at(p);
        if (above < 0) { out[i + w * j] = n[axis]!; continue; }
        for (let k = n[axis]! - 1; k >= 0; k--) { p[axis] = k; const below = at(p); if (below < 0 && above >= 0) { out[i + w * j] = k + below / (below - above); break; } above = below; }
      }
      return { w, h, v: out };
    };
    const top = crossing(1), sideX = crossing(0), sideZ = crossing(2);
    if (!column) {
      // The column as it stands at the first read: its faces at the vertex nearest the corner.
      column = [Math.round(sideX.v[2 + sideX.w * 2]!), Math.round(top.v[2 + top.w * 2]!), Math.round(sideZ.v[2 + sideZ.w * 2]!)];
      console.log(`column ${column.join(" x ")} cells in the corner at the origin`);
    }
    const [X0, Y0, Z0] = column;
    const lookup = (name: string, i: number, j: number) => {
      const map = name === "top" || name === "rim" ? top : name.endsWith("X") ? sideX : sideZ, m = (a: number, b: number) => map.v[a + map.w * b]!;
      return { m: m(i, j), r: m(i, j) - (m(i - 2, j) + m(i + 2, j) + m(i, j - 2) + m(i, j + 2)) / 4 };
    };
    for (const line of lines) {
      const map = line.face === "top" ? top : line.face === "sideX" ? sideX : sideZ, axis = line.face === "top" ? 1 : line.face === "sideX" ? 0 : 2, u = axis === 0 ? 1 : 0, v = axis === 2 ? 1 : 2;
      let values = "", kinds = "";
      for (let k = line.from; k <= line.to; k++) {
        const i = line.fixed === "i" ? line.at : k, j = line.fixed === "i" ? k : line.at, value = map.v[i + map.w * j]!;
        values += ` ${Number.isFinite(value) ? value.toFixed(2).padStart(6) : "     ."}`;
        const p = [0, 0, 0]; p[u] = Math.min(i, [nx, ny, nz][u]! - 1); p[v] = Math.min(j, [nx, ny, nz][v]! - 1); p[axis] = Math.min([nx, ny, nz][axis]! - 1, Math.max(0, Math.floor(Number.isFinite(value) ? value : 0)));
        kinds += kindAt(p[0]!, p[1]!, p[2]!);
      }
      console.log(`LINE ${line.face} ${line.fixed}=${line.at} ${line.fixed === "i" ? "j" : "i"}=${line.from}..${line.to} ${label}:${values} | ${kinds}`);
    }
    if (spectra && print) {
      // Windows of 32 samples along one index of a map, quadratic trend removed, Hann weighted: the mean amplitude by period, and the
      // detrended samples' mean by their coordinate modulo 4 and 32 (peak to peak), the part locked to the tile and the page lattice.
      const spectrum = (name: string, map: Map2, inside: (i: number, j: number) => boolean) => {
        for (const along of ["i", "j"] as const) {
          const power = new Float64Array(17), fold4 = new Float64Array(4), count4 = new Float64Array(4), fold32 = new Float64Array(32), count32 = new Float64Array(32); let windows = 0, square = 0;
          const outer = along === "i" ? map.h : map.w, inner = along === "i" ? map.w : map.h, value = (a: number, b: number) => along === "i" ? map.v[a + map.w * b]! : map.v[b + map.w * a]!, ok = (a: number, b: number) => along === "i" ? inside(a, b) : inside(b, a);
          for (let b = 0; b < outer; b++) for (let a0 = 0; a0 + 32 <= inner; a0 += 16) {
            let whole = true; for (let k = 0; k < 32 && whole; k++) whole = ok(a0 + k, b) && Number.isFinite(value(a0 + k, b));
            if (!whole) continue;
            // Least-squares quadratic in t = k - 15.5 (the odd and even moments separate).
            let s0 = 0, s1 = 0, s2 = 0, t2 = 0, t4 = 0;
            for (let k = 0; k < 32; k++) { const t = k - 15.5, y = value(a0 + k, b); s0 += y; s1 += t * y; s2 += t * t * y; t2 += t * t; t4 += t * t * t * t; }
            const c1 = s1 / t2, c2 = (32 * s2 - t2 * s0) / (32 * t4 - t2 * t2), c0 = (s0 - c2 * t2) / 32, e = new Float64Array(32);
            for (let k = 0; k < 32; k++) { const t = k - 15.5; e[k] = value(a0 + k, b) - (c0 + c1 * t + c2 * t * t); square += e[k]! * e[k]!; fold4[(a0 + k) & 3]! += e[k]!; count4[(a0 + k) & 3]!++; fold32[(a0 + k) & 31]! += e[k]!; count32[(a0 + k) & 31]!++; }
            for (let f = 1; f <= 16; f++) { let re = 0, im = 0; for (let k = 0; k < 32; k++) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * (k + 0.5) / 32), angle = 2 * Math.PI * f * k / 32; re += w * e[k]! * Math.cos(angle); im += w * e[k]! * Math.sin(angle); } power[f]! += (re * re + im * im) / 64; }
            windows++;
          }
          if (!windows) { console.log(`SPECTRUM ${name} along ${along} ${label}: no window`); continue; }
          const spread = (fold: Float64Array, count: Float64Array) => { let low = Infinity, high = -Infinity; fold.forEach((sum, k) => { if (count[k]) { low = Math.min(low, sum / count[k]!); high = Math.max(high, sum / count[k]!); } }); return (high - low).toFixed(3); };
          console.log(`SPECTRUM ${name} along ${along} ${label}: windows ${windows} rms ${Math.sqrt(square / (32 * windows)).toFixed(3)} locked4 ${spread(fold4, count4)} locked32 ${spread(fold32, count32)} amplitude by period ${[2, 3, 4, 5, 6, 7, 8, 10, 12, 16].map(f => `${(32 / f).toFixed(1)}:${Math.sqrt(power[f]! / windows).toFixed(3)}`).join(" ")}`);
        }
      };
      spectrum("sheet", top, (i, j) => i >= 2 && j >= 2 && i <= nx - 2 && j <= nz - 2 && (i > X0 + 4 || j > Z0 + 4) && top.v[i + top.w * j]! > 0.5);
      spectrum("footX", sideX, (i, j) => i >= 2 && i <= 40 && j >= 2 && j <= Z0 - 10);
      spectrum("topAll", top, (i, j) => i >= 2 && j >= 2 && i <= X0 - 3 && j <= Z0 - 3);
    }
    if (!print) return { lookup, picks: [] as { face: string; i: number; j: number }[] };
    const picks: { face: string; i: number; j: number }[] = [];
    let liquid = 0, fine = 0, coarse = 0; for (let i = 0; i < volume.length; i++) liquid += 0; void liquid;
    for (let z = 0; z < nz; z += 4) for (let y = 0; y < ny; y += 4) for (let x = 0; x < nx; x += 4) { if (!stored(x, y, z)) continue; if (wordAt(x, y, z) & 0x80000000) fine++; else coarse++; }
    console.log(`${label}: particles ${solver!.narrowBandFlipInfo?.particles ?? 0}, stored tiles h ${fine} 4h ${coarse}, resident ${live[1]}, pool generation ${live[0]}`);
    // A face: its map, the vertex box of the map it covers, and the lattice coordinate of a map entry.
    // The side faces end ten cells under the lowest point of the top's interior, which sinks as the column slumps.
    let crown = Y0; for (let j = 2; j <= Z0 - 10; j++) for (let i = 2; i <= X0 - 10; i++) { const value = top.v[i + top.w * j]!; if (Number.isFinite(value)) crown = Math.min(crown, Math.floor(value)); }
    const faces: { name: string; map: Map2; axis: number; box: [number, number, number, number]; keep?: (i: number, j: number) => boolean }[] = [
      { name: "top", map: top, axis: 1, box: [2, X0 - 10, 2, Z0 - 10] },
      { name: "sideX", map: sideX, axis: 0, box: [12, crown - 10, 2, Z0 - 10] },
      { name: "sideZ", map: sideZ, axis: 2, box: [2, X0 - 10, 12, crown - 10] },
      // The side faces above the foot, where the column turns into the sheet and its side is no longer a height over y and z.
      { name: "bodyX", map: sideX, axis: 0, box: [32, crown - 10, 2, Z0 - 10] },
      { name: "bodyZ", map: sideZ, axis: 2, box: [2, X0 - 10, 32, crown - 10] },
      // The top's rim, three to ten cells in from the column's free edges.
      { name: "rim", map: top, axis: 1, box: [2, X0 - 3, 2, Z0 - 3], keep: (i, j) => i > X0 - 10 || j > Z0 - 10 },
      { name: "sheet", map: top, axis: 1, box: [2, nx - 2, 2, nz - 2], keep: (i, j) => i > X0 + 4 || j > Z0 + 4 },
    ];
    const summary: Record<string, unknown> = { step };
    for (const face of faces) {
      const { map, axis, box: [i0, i1, j0, j1] } = face, m = (i: number, j: number) => map.v[i + map.w * j]!;
      const inFace = (i: number, j: number) => i >= i0 && i <= i1 && j >= j0 && j <= j1 && (!face.keep || face.keep(i, j)) && Number.isFinite(m(i, j)) && (face.name !== "sheet" || m(i, j) > 0.5);
      const rough: { i: number; j: number; r: number }[] = []; let sum = 0;
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        if (!(inFace(i, j) && inFace(i - 2, j) && inFace(i + 2, j) && inFace(i, j - 2) && inFace(i, j + 2))) continue;
        const r = m(i, j) - (m(i - 2, j) + m(i + 2, j) + m(i, j - 2) + m(i, j + 2)) / 4; rough.push({ i, j, r }); sum += r * r;
      }
      // Least-squares quadratic over the face (normal equations, six terms).
      const A = Array.from({ length: 6 }, () => new Float64Array(7)), points: [number, number, number][] = [];
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (inFace(i, j)) points.push([i - i0, j - j0, m(i, j)]);
      const basis = (a: number, b: number) => [1, a, b, a * a, a * b, b * b];
      for (const [a, b, value] of points) { const q = basis(a, b); for (let r = 0; r < 6; r++) { for (let c = 0; c < 6; c++) A[r]![c]! += q[r]! * q[c]!; A[r]![6]! += q[r]! * value; } }
      let fitRms = NaN, fitMax = NaN, mean = NaN, spread = NaN;
      if (points.length >= 12 && face.name !== "sheet" && face.name !== "rim") {
        for (let c = 0; c < 6; c++) {
          let pivot = c; for (let r = c + 1; r < 6; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[pivot]![c]!)) pivot = r;
          [A[c], A[pivot]] = [A[pivot]!, A[c]!];
          for (let r = 0; r < 6; r++) { if (r === c || !A[c]![c]) continue; const f = A[r]![c]! / A[c]![c]!; for (let k = c; k < 7; k++) A[r]![k]! -= f * A[c]![k]!; }
        }
        const coefficient = A.map((row, r) => row[6]! / row[r]!); let s = 0; fitMax = 0; mean = 0; let low = Infinity, high = -Infinity;
        for (const [a, b, value] of points) { const e = value - basis(a, b).reduce((t, q, r) => t + q * coefficient[r]!, 0); s += e * e; fitMax = Math.max(fitMax, Math.abs(e)); mean += value / points.length; low = Math.min(low, value); high = Math.max(high, value); }
        fitRms = Math.sqrt(s / points.length); spread = high - low;
      }
      const sorted = rough.slice().sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
      summary[face.name] = { points: points.length, rough_rms: +Math.sqrt(sum / Math.max(1, rough.length)).toFixed(4), rough_max: +Math.abs(sorted[0]?.r ?? NaN).toFixed(3), dents: rough.filter(q => Math.abs(q.r) > 0.2).length, fit_rms: +fitRms.toFixed(4), fit_max: +fitMax.toFixed(3), mean: +mean.toFixed(3), spread: +spread.toFixed(3) };
      const picked: typeof rough = [];
      for (const candidate of sorted) { if (picked.length >= worstCount || Math.abs(candidate.r) < 0.1) break; if (picked.every(p => Math.max(Math.abs(p.i - candidate.i), Math.abs(p.j - candidate.j)) >= 6)) picked.push(candidate); }
      const u = axis === 0 ? 1 : 0, v = axis === 2 ? 1 : 2;
      for (const { i, j } of picked.slice(0, 4)) if (face.name !== "sheet") picks.push({ face: face.name, i, j });
      for (const { i, j, r } of picked) {
        const p = [0, 0, 0]; p[u] = i; p[v] = j; p[axis] = Math.min([nx, ny, nz][axis]! - 1, Math.max(0, Math.floor(m(i, j))));
        const c = [Math.min(p[0]!, nx - 1), Math.min(p[1]!, ny - 1), Math.min(p[2]!, nz - 1)], t = c.map(q => q >> 2), around = [[-4, 0, 0], [4, 0, 0], [0, -4, 0], [0, 4, 0], [0, 0, -4], [0, 0, 4]].map(d => kindAt(c[0]! + d[0]!, c[1]! + d[1]!, c[2]! + d[2]!)).join("");
        console.log(`  worst ${face.name} ${label}: rough ${r.toFixed(3)} at map (${i},${j}) crossing ${m(i, j).toFixed(2)} cell (${c.join(",")}) local (${c.map(q => q & 31).join(",")}) tile (${t.join(",")}) in page (${t.map(q => q & 7).join(",")}) page (${c.map(q => q >> 5).join(",")}) block ${pageBlock(c[0]!, c[1]!, c[2]!)} width ${kindAt(c[0]!, c[1]!, c[2]!)} around -x+x-y+y-z+z ${around} volume ${volume[c[0]! + nx * (c[1]! + ny * c[2]!)]!.toFixed(3)}`);
      }
      if (maps && (face.name === "top" || face.name.startsWith("side"))) {
        console.log(`  map ${face.name} ${label}: rows ${axis === 1 ? "z" : axis === 0 ? "z" : "y"} = ${j0 - 2}.. (down the page), columns ${axis === 2 ? "x" : axis === 0 ? "y" : "x"} = ${i0 - 2}..${i1 + 12}; '|' the tile kind of each crossing`);
        for (let j = j0 - 2; j <= j1 + 12 && j < map.h; j++) {
          let line = "", kinds = "";
          for (let i = i0 - 2; i <= i1 + 12 && i < map.w; i++) {
            const value = m(i, j);
            if (!Number.isFinite(value)) { line += " "; kinds += " "; continue; }
            const near: number[] = []; for (let b = -4; b <= 4; b++) for (let a = -4; a <= 4; a++) { const q = map.v[i + a + map.w * (j + b)]; if (i + a >= 0 && i + a < map.w && j + b >= 0 && j + b < map.h && Number.isFinite(q)) near.push(q!); }
            const d = value - median(near), tenths = Math.min(9, Math.round(Math.abs(d) * 10));
            line += Math.abs(d) < 0.15 ? "." : d < 0 ? String(tenths) : String.fromCharCode(96 + Math.min(26, tenths));
            const p = [0, 0, 0]; p[u] = Math.min(i, [nx, ny, nz][u]! - 1); p[v] = Math.min(j, [nx, ny, nz][v]! - 1); p[axis] = Math.min([nx, ny, nz][axis]! - 1, Math.max(0, Math.floor(value)));
            kinds += kindAt(p[0]!, p[1]!, p[2]!);
          }
          console.log(`  ${String(j).padStart(3)} ${line} | ${kinds}`);
        }
      }
    }
    console.log(`DENTS ${JSON.stringify(summary)}`);
    void fineAt;
    return { lookup, picks };
    };
    const end = analyse(`step ${step}`, last, print);
    if (ledger && print) { let sum = 0; for (let i = 0; i < volume.length; i++) sum += volume[i]!; console.log(`LEDGER step ${step}: the volume field holds ${sum.toFixed(0)} cells, ${solver!.narrowBandFlipInfo?.particles ?? 0} samples`); }
    if (foot && print && address) {
      const nb = (frame as unknown as { narrowBandFlip: { activeParticles: GPUBuffer; bins: GPUBuffer; activityWord: number; tiles: number } }).narrowBandFlip, T = nb.tiles;
      const [[Z], [x0, x1], [y0, y1]] = foot as [number[], number[], number[]], cell = dt / H, count = solver!.narrowBandFlipInfo?.particles ?? 0;
      // Heat, target, activated, theta and surface heat, a word a tile key each.
      const [heatWords, sampleWords] = await Promise.all([censusWords({ buffer: nb.bins, offset: 4 * (nb.activityWord + 2 * T) }, 5 * T), censusWords({ buffer: nb.activeParticles, offset: 0 }, 12 * Math.max(1, count))]);
      const heat = new Float32Array(heatWords.buffer), samples = new Float32Array(sampleWords.buffer), key = (x: number, y: number, z: number) => lattice!.uniformTileKey(address, x, y, z);
      const tx = Array.from({ length: (x1! >> 2) - (x0! >> 2) + 1 }, (_, k) => (x0! >> 2) + k);
      const tileTable = (title: string, text: (t: number) => string) => {
        console.log(`  foot step ${step} tile z ${Z! >> 2}: ${title}, columns tile x = ${tx[0]}..${tx[tx.length - 1]}`);
        for (let y = y1! >> 2; y >= y0! >> 2; y--) console.log(`   y ${String(y).padStart(2)} ${tx.map(x => (stored(4 * x, 4 * y, Z!) ? text(key(x, y, Z! >> 2)) : "_").padStart(4)).join(" ")}`);
      };
      tileTable("width, * activated this step", t => `${(tiles[t]! & 0x80000000) ? "h" : "C"}${heatWords[2 * T + t] ? "*" : ""}`);
      tileTable("target heat", t => heat[T + t]!.toFixed(1)); tileTable("theta (the samples' support heat)", t => heat[3 * T + t]!.toFixed(1)); tileTable("surface heat (the samples' own)", t => heat[4 * T + t]!.toFixed(1));
      const w = x1! - x0! + 1, hgt = y1! - y0! + 1, all = new Uint16Array(w * hgt), outer = new Uint16Array(w * hgt), rows = Array.from({ length: hgt }, () => [] as number[]);
      for (let i = 0; i < count; i++) {
        const o = 12 * i, x = Math.floor(samples[o]!), y = Math.floor(samples[o + 1]!);
        if (Math.floor(samples[o + 2]!) !== Z || x < x0! || x > x1! || y < y0! || y > y1!) continue;
        all[x - x0! + w * (y - y0!)]!++; if (samples[o + 7]! > -1) outer[x - x0! + w * (y - y0!)]!++; rows[y - y0!]!.push(i);
      }
      const digit = (n: number) => n === 0 ? "." : n < 10 ? String(n) : "+", fraction = (x: number, y: number) => volume[x + nx * (y + ny * Z!)]!;
      for (const [title, char] of [["liquid fraction (' ' under 0.02, - under 0.5, = under 0.98, # full)", (x: number, y: number) => { const f = fraction(x, y); return f < 0.02 ? " " : f < 0.5 ? "-" : f < 0.98 ? "=" : "#"; }],
        ["samples a cell", (x: number, y: number) => digit(all[x - x0! + w * (y - y0!)]!)], ["samples of the outer layer a cell", (x: number, y: number) => digit(outer[x - x0! + w * (y - y0!)]!)]] as const) {
        console.log(`  foot step ${step} z ${Z}: ${title}, columns x = ${x0}..${x1}`);
        for (let y = y1!; y >= y0!; y--) console.log(`   y ${String(y).padStart(3)} ${Array.from({ length: w }, (_, k) => char(x0! + k, y)).join("")}`);
      }
      console.log(`  foot step ${step} z ${Z}: per row the front, the farthest sample, then over the four cells behind the front: samples, outer samples, mean heat, mean own vx, mean grid vx (cells a step), rms |own - grid|`);
      for (let y = y1!; y >= y0!; y--) {
        let front = -1; for (let x = x1!; x >= x0!; x--) if (fraction(x, y) >= 0.5) { front = x; break; }
        let farthest = -1, n = 0, out = 0, hot = 0, own = 0, grid = 0, square = 0;
        for (const i of rows[y - y0!]!) {
          const o = 12 * i, x = samples[o]!; farthest = Math.max(farthest, x);
          if (x < front - 3 || x >= front + 1) continue;
          n++; if (samples[o + 7]! > -1) out++; hot += samples[o + 3]!; own += samples[o + 4]! * cell; grid += samples[o + 8]! * cell;
          square += ((samples[o + 4]! - samples[o + 8]!) ** 2 + (samples[o + 5]! - samples[o + 9]!) ** 2 + (samples[o + 6]! - samples[o + 10]!) ** 2) * cell * cell;
        }
        console.log(`   y ${String(y).padStart(3)} front ${String(front).padStart(4)} farthest ${farthest.toFixed(2).padStart(7)} samples ${String(n).padStart(3)} outer ${String(out).padStart(3)} heat ${(hot / Math.max(1, n)).toFixed(2)} own ${(own / Math.max(1, n)).toFixed(2).padStart(6)} grid ${(grid / Math.max(1, n)).toFixed(2).padStart(6)} rms ${Math.sqrt(square / Math.max(1, n)).toFixed(3)}`);
      }
    }
    if (criteria && print && address && step > 0) {
      // The census's words per tile key: two importance words, the hold, the budget's histogram, state (cutoff, ties, allowed) and rank; the band reasons.
      const census = (solver as unknown as { mixedDynamic: Record<"importance" | "reasons", { buffer: GPUBuffer; offset: number }> }).mixedDynamic, T = tiles.length;
      const [words, reasons] = await Promise.all([censusWords(census.importance, 4 * T + 259), censusWords(census.reasons, T)]);
      const key = (x: number, y: number, z: number) => lattice!.uniformTileKey(address, x, y, z), fine = (t: number) => (tiles[t]! & 0x80000000) !== 0;
      const score = (t: number, k: number) => ((k < 4 ? words[2 * t]! >>> (8 * k) : words[2 * t + 1]! >>> (8 * (k - 4))) & 255) / 64, flag = (t: number, bit: number) => (words[2 * t + 1]! & (1 << bit)) !== 0;
      const tally = { crossing: [0, 0], required: 0, held: 0, dropped: 0, flight: 0, triggered: CRITERIA.map(() => 0), reason: REASONS.map(() => 0), left: CRITERIA.map(() => [0, 0, 0, 0]) };
      // The budget's verdict on the requests: kept and dropped, by the width the tile already has, whether it lies under the column as it stood, and the highest of its scores (rotation, off, aside).
      const verdict = () => ({ h: 0, coarse: 0, column: 0, out: 0, top: CRITERIA.map(() => 0) }), verdicts = { kept: verdict(), dropped: verdict() };
      for (let z = 0; z < nz / 4; z++) for (let y = 0; y < ny / 4; y++) for (let x = 0; x < nx / 4; x++) {
        if (!stored(4 * x, 4 * y, 4 * z)) continue;
        const t = key(x, y, z), h = fine(t);
        if (h) tally.reason[Math.min(reasons[t]!, REASONS.length - 1)]!++;
        if (flag(t, 25)) tally.required++; if (flag(t, 26)) tally.held++; if (flag(t, 27)) tally.dropped++; if (words[3 * T + 259 + t] === 0xfffffffd) tally.flight++;
        CRITERIA.forEach((_, k) => { if (flag(t, 16 + k)) tally.triggered[k]!++; });
        const v = flag(t, 27) ? verdicts.dropped : flag(t, 25) && !flag(t, 26) ? verdicts.kept : undefined;
        if (v) { v[h ? "h" : "coarse"]++; v[column && 4 * x < column[0] && 4 * z < column[2] ? "column" : "out"]++; let top = 0; CRITERIA.forEach((_, k) => { if (k !== 3 && score(t, k) > score(t, top)) top = k; }); v.top[top]!++; }
        if (!flag(t, 28)) continue;
        tally.crossing[h ? 0 : 1]!++;
        if (!h) CRITERIA.forEach((_, k) => { const v = score(t, k), bins = tally.left[k]!; bins[v >= 1 ? 0 : v >= 0.5 ? 1 : v > 0 ? 2 : 3]!++; });
      }
      console.log(`CRITERIA step ${step}: crossing tiles ${tally.crossing[0]! + tally.crossing[1]!} (h ${tally.crossing[0]}, 4h ${tally.crossing[1]}), required ${tally.required}, held ${tally.held}, dropped by the budget ${tally.dropped}, in flight ${tally.flight}, budget cutoff ${(words[3 * T + 256]! / 64).toFixed(2)} ties ${words[3 * T + 257]} allowed ${words[3 * T + 258]}`);
      console.log(`CRITERIA step ${step}: triggered ${CRITERIA.map((name, k) => `${name} ${tally.triggered[k]}`).join(", ")}; h tiles by reason ${REASONS.map((name, k) => `${name} ${tally.reason[k]}`).filter((_, k) => tally.reason[k]).join(", ")}`);
      console.log(`CRITERIA step ${step}: crossing tiles left 4h by score (1 and over / 0.5 to 1 / under 0.5 / none): ${CRITERIA.map((name, k) => `${name} ${tally.left[k]!.join("/")}`).join(", ")}`);
      for (const [name, v] of Object.entries(verdicts)) console.log(`CRITERIA step ${step}: the budget ${name} ${v.h + v.coarse} requests: ${v.h} of tiles already h, ${v.coarse} of 4h tiles; ${v.column} under the column, ${v.out} outside it; by top criterion ${CRITERIA.map((c, k) => `${c} ${v.top[k]}`).filter((_, k) => v.top[k]).join(", ")}`);
      const [[x0, x1], [y0, y1], [z]] = criteria as [number[], number[], number[]], columns = Array.from({ length: x1! - x0! + 1 }, (_, k) => x0! + k);
      const table = (title: string, cell: (t: number, x: number, y: number) => string) => {
        console.log(`  criteria step ${step} tile z ${z}: ${title}, columns tile x = ${x0}..${x1}`);
        for (let y = y1!; y >= y0!; y--) console.log(`   y ${String(y).padStart(2)} ${columns.map(x => (stored(4 * x, 4 * y, 4 * z!) ? cell(key(x, y, z!), x, y) : "_").padStart(5)).join(" ")}`);
      };
      table("width after the build, crossing (x) and wet (w)", t => `${fine(t) ? "h" : "C"}${flag(t, 28) ? "x" : ""}${flag(t, 29) ? "w" : ""}`);
      CRITERIA.forEach((name, k) => table(`${name} score${k === 3 ? " (off)" : ""}`, t => flag(t, 28) || flag(t, 29) ? `${score(t, k).toFixed(2)}${flag(t, 16 + k) ? "!" : ""}` : "."));
      table("required (R), held (H), dropped by the budget (D), in flight (F)", t => `${flag(t, 25) ? "R" : ""}${flag(t, 26) ? "H" : ""}${flag(t, 27) ? "D" : ""}${words[3 * T + 259 + t] === 0xfffffffd ? "F" : ""}` || ".");
      table("band reason", t => REASONS[Math.min(reasons[t]!, REASONS.length - 1)]!.slice(0, 5));
    }
    if (dump && extras.velocity && stages.length > 1) {
      // A slab at fixed z about the dump cell: tile kinds before and after this step's relayout, phi (in cells) as the head and the advect
      // left it, and the velocity the advect read (cells a step, each cell's upper faces).
      const velocity = await extras.velocity(), [X, Y, Z] = dump, cell = dt / H, row = (y: number, f: (x: number) => string) => `   y ${String(y).padStart(3)} ${Array.from({ length: 15 }, (_, k) => f(X - 7 + k)).join(" ")}`;
      const old = previousTiles, kindOld = (x: number, y: number, z: number) => !old ? "?" : (old[lattice!.uniformTileKey(address!, x >> 2, y >> 2, z >> 2)]! & 0x80000000) ? "h" : "C";
      const phiOf = (field: Float32Array) => (x: number, y: number) => (field[x + (nx + 1) * (y + (ny + 1) * Z)]! / H).toFixed(2).padStart(6);
      const head = staged[0]!, advect = staged[1]!;
      for (const [title, f] of [
        ["kind before>after", (y: number) => (x: number) => `${kindOld(x, y, Z)}>${kindAt(x, y, Z)}`.padStart(6)],
        [`phi ${stages[0]!.name} (stored)`, (y: number) => (x: number) => phiOf(head)(x, y)],
        [`phi ${stages[1]!.name} (stored)`, (y: number) => (x: number) => phiOf(advect)(x, y)],
        ...[0, 1, 2].map(a => [`velocity ${"xyz"[a]} (cells a step)`, (y: number) => (x: number) => (velocity[4 * (x + nx * (y + ny * Z)) + a]! * cell).toFixed(2).padStart(6)] as const),
      ] as const) {
        console.log(`  dump step ${step} z ${Z}: ${title}, columns x = ${X - 7}..${X + 7}`);
        for (let y = Y + 8; y >= Y - 8; y--) console.log(row(y, f(y)));
      }
    }
    if (stages.length) {
      // Stage by stage: the same numbers on the field each stage left, then the end's worst entries followed through the stages.
      const results = stages.map((stage, k) => { const quiet = console.log; console.log = (...a: unknown[]) => { if (/^(DENTS|LINE|LEDGER)/.test(String(a[0]))) quiet(`STAGE ${stage.name} ${a[0]}`); }; try { return analyse(`step ${step} ${stage.name}`, staged[k]!, true); } finally { console.log = quiet; } });
      for (const pick of end.picks) console.log(`  follow ${pick.face} (${pick.i},${pick.j}) step ${step}: ${stages.map((stage, k) => { const q = results[k]!.lookup(pick.face, pick.i, pick.j); return `${stage.name} ${q.m.toFixed(3)}/${q.r.toFixed(3)}`; }).join("  ")}  end ${(() => { const q = end.lookup(pick.face, pick.i, pick.j); return `${q.m.toFixed(3)}/${q.r.toFixed(3)}`; })()}`);
    }
    } finally { address?.seed.state.set(built!); }
  };
  // Phi as each stage of a step leaves it: the advected scratch, then the live field after each resolve.
  type Stage = { name: string; done: () => Promise<Float32Array> };
  let stages: Stage[] | undefined, stageName = "head", stageCount = 0, previousTiles: Uint32Array | undefined;
  const extras: Record<string, () => Promise<Float32Array>> = {}, velocityStages: (() => Promise<void>)[] = [];
  {
    const f = frame as unknown as { fields: { phi: GPUTexture; phiScratch: GPUTexture; velocityScratch: GPUTexture }; phiResolve: { encode: (e: GPUCommandEncoder, g: unknown) => void }; phiResolveGroups: { phi: unknown; scratch: unknown };
      narrowBandFlip: Record<"reconstruct" | "retire" | "redistance", (e: GPUCommandEncoder) => unknown>; coarsePhi: { encode: (e: GPUCommandEncoder) => void } };
    const capture = (encoder: GPUCommandEncoder, name: string, texture: GPUTexture) => { if (stages) stages.push({ name: `${stageCount++}:${name}`, done: uniformDetailField(texture)!.storage.capture(encoder, texture) }); };
    const resolve = f.phiResolve.encode.bind(f.phiResolve);
    f.phiResolve.encode = (encoder, group) => { resolve(encoder, group); if (group === f.phiResolveGroups.scratch) { capture(encoder, "advect", f.fields.phiScratch); if (stages && dump) extras.velocity = uniformDetailField(f.fields.velocityScratch)!.storage.capture(encoder, f.fields.velocityScratch); } else capture(encoder, stageName, f.fields.phi); };
    for (const key of ["reconstruct", "retire", "redistance"] as const) { const original = f.narrowBandFlip[key].bind(f.narrowBandFlip); f.narrowBandFlip[key] = encoder => { stageName = key; return original(encoder); }; }
    if (vtrace) {
      // The velocity fields as each velocity stage of a step leaves them, at a column of cells through the trace cell.
      const g = frame as unknown as { fields: { velocity: GPUTexture; velocityScratch: GPUTexture }; momentum: { encode: (...a: unknown[]) => unknown }; forces: { encode: (...a: unknown[]) => unknown };
        split: { transfer: { encodeToSimulation: (...a: unknown[]) => unknown } }; band: { encodeSolve: (...a: unknown[]) => unknown }; narrowBandFlip: { transfer: (...a: unknown[]) => unknown; update: (...a: unknown[]) => unknown } };
      const [X, Y, Z] = vtrace, cell = dt / H;
      const take = (encoder: GPUCommandEncoder, name: string) => {
        if (!stages) return;
        for (const [label, texture] of [["velocity", g.fields.velocity], ["scratch", g.fields.velocityScratch]] as const) {
          const done = uniformDetailField(texture)!.storage.capture(encoder, texture);
          velocityStages.push(async () => {
            const field = await done(), line = (a: number) => Array.from({ length: 12 }, (_, k) => (field[4 * (X + nx * (Y - 5 + k + ny * Z)) + a]! * cell).toFixed(2).padStart(6)).join(" ");
            console.log(`  vtrace ${name.padEnd(12)} ${label.padEnd(8)} x ${X} z ${Z} y ${Y - 5}..${Y + 6}: vx ${line(0)} | vy ${line(1)}`);
          });
        }
      };
      const after = (object: Record<string, (...a: unknown[]) => unknown>, key: string, name: string, before = false) => {
        const original = object[key]!.bind(object);
        object[key] = (...a: unknown[]) => { if (before) take(a[0] as GPUCommandEncoder, name); const result = original(...a); if (!before) take(a[0] as GPUCommandEncoder, name); return result; };
      };
      after(g.momentum, "encode", "momentum"); after(g.narrowBandFlip, "transfer", "p2g"); after(g.forces, "encode", "forces");
      after(g.split.transfer, "encodeToSimulation", "toSimulation"); after(g.band, "encodeSolve", "band"); after(g.narrowBandFlip, "update", "extended", true);
      const move = f.narrowBandFlip as unknown as Record<string, (...a: unknown[]) => unknown>; after(move, "move", "head", true);
    }
    if (vslab && dump) {
      // The slab of --dump in the velocity each velocity stage leaves: u* after the forces, the root's projection in simulation
      // ownership, the band's, and the extended field the samples and the next advection read. Raw texels: a 4h tile holds each
      // face once, at the cell of its origin on the other two axes.
      const g = frame as unknown as { fields: { velocity: GPUTexture; velocityScratch: GPUTexture }; forces: Record<string, (...a: unknown[]) => unknown>; split: { transfer: Record<string, (...a: unknown[]) => unknown> };
        band: Record<string, (...a: unknown[]) => unknown>; narrowBandFlip: Record<string, (...a: unknown[]) => unknown> };
      const [X, Y, Z] = dump, cell = dt / H;
      const hook = (object: Record<string, (...a: unknown[]) => unknown>, key: string, name: string, texture: () => GPUTexture, before = false) => {
        const original = object[key]!.bind(object);
        const take = (encoder: GPUCommandEncoder) => {
          if (!stages) return;
          const done = uniformDetailField(texture())!.storage.capture(encoder, texture());
          velocityStages.push(async () => {
            const field = await done();
            for (const a of [0, 1, 2]) {
              console.log(`  vslab ${name}: velocity ${"xyz"[a]} (cells a step), z ${Z}, columns x = ${X - 7}..${X + 7}`);
              for (let y = Y + 8; y >= Y - 8; y--) console.log(`   y ${String(y).padStart(3)} ${Array.from({ length: 15 }, (_, k) => (field[4 * (X - 7 + k + nx * (y + ny * Z)) + a]! * cell).toFixed(3).padStart(7)).join(" ")}`);
            }
          });
        };
        object[key] = (...a: unknown[]) => { if (before) take(a[0] as GPUCommandEncoder); const result = original(...a); if (!before) take(a[0] as GPUCommandEncoder); return result; };
      };
      hook(g.forces, "encode", "forces", () => g.fields.velocityScratch); hook(g.split.transfer, "encodeToSimulation", "toSimulation", () => g.fields.velocity);
      hook(g.band, "encodeSolve", "band", () => g.fields.velocity); hook(g.narrowBandFlip, "update", "extended", () => g.fields.velocity, true);
      hook(g.narrowBandFlip, "update", "extension", () => g.fields.velocityScratch, true);
      const relayout = (frame as unknown as { relayout?: Record<string, (...a: unknown[]) => unknown> }).relayout;
      if (relayout) hook(relayout, "encode", "headExtension", () => g.fields.velocityScratch, true);
      const r = frame as unknown as { remap: Record<string, (...a: unknown[]) => unknown>; hanging: { unitVelocity: GPUTexture } };
      hook(r.remap, "applyGpu", "remapHolder", () => r.hanging.unitVelocity); hook(r.remap, "applyGpu", "remapExtension", () => g.fields.velocityScratch); hook(r.remap, "applyGpu", "remapPhysical", () => g.fields.velocity);
    }
    const coarse = f.coarsePhi.encode.bind(f.coarsePhi); f.coarsePhi.encode = encoder => { coarse(encoder); capture(encoder, "coarsePhi", f.fields.phi); };
  }

  await report(0, reads.has(0));
  for (let step = 1; step <= steps; step++) {
    const t0 = performance.now();
    stages = stageSteps.has(step) ? [] : undefined; stageName = "head"; stageCount = 0;
    solver.advanceTo(step * dt, []); await solver.awaitFrameCompletion();
    const taken = stages ?? []; stages = undefined;
    if (solver.info.simulationPipelineError) { console.log(`solver: ${solver.info.simulationPipelineError}`); process.exitCode = 1; break; }
    console.log(`advanced step ${step} in ${(performance.now() - t0).toFixed(0)} ms`);
    if (velocityStages.length) { await solver.awaitFrameCompletion(); console.log(`  vtrace step ${step}`); for (const print of velocityStages.splice(0)) await print(); }
    if (reads.has(step) || taken.length) await report(step, true, taken);
    if (dump && stageSteps.has(step + 1)) previousTiles = await readMixedTileWords(device, solver);
  }
  solver.destroy(); solver = undefined; await device.queue.onSubmittedWorkDone();
  }
  console.log(`errors ${JSON.stringify(errors)}`);
} finally { solver?.destroy(); await device?.queue.onSubmittedWorkDone(); device?.destroy(); }
