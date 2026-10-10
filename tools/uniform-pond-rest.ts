import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { UNIFORM_MIXED_THETA_MIN } from "../lib/methods/uniform/uniform-mixed-pressure-surface.wgsl";
import type { WebGPUUniformPressureMultigrid } from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import type { UniformTexturePages } from "../lib/methods/uniform/uniform-texture-pages";
import { readUniformFields, readUniformSolidFractions } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { uniformFixedStep_s } from "../lib/methods/uniform/uniform-paper";

/** One resting-pond sample. Lengths in mm, speeds in m/s, V in h cells. */
export interface PondRestSample {
  frame: number; missing: number; maxSpeed: number; maxCell: number[]; sum: number; excess: number;
  /** --basin=vessels: each chamber's level from its V, in h cells. */
  chambers?: number[];
  /** --fine-box: the fastest face on the box's boundary planes with a wet side (the h/4h seam), m/s. */
  seamSpeed?: number;
  /** The fastest wet cell of a 4h owner, m/s (maxSpeed covers h cells too). */
  coarseSpeed: number;
  /** Tiles the layout holds at h in this frame. */
  fineTiles: number;
  residual: number | undefined; acceptedResidual: number | undefined; bandResidual: number | undefined; bandTiles: number | undefined; recovery: number; full: number | undefined; vcycles: number | undefined;
  /** Largest |volume correction| any pressure owner carries, 1/s. */
  correction: number;
  /** Open pressure owners whose centre is below the waterline's liquid margin but air, and above it but liquid. */
  airBelow: number; liquidAbove: number;
  surface: Record<"all" | "interior", { count: number; mean_mm: number; rms_mm: number; max_mm: number; range_mm: number }>;
}
export interface PondRestResult {
  arm: string; dimensions: number[]; level: number; h: number; depth_m: number; time_s: number; detail: string; basin: string;
  /** h tiles of the layout the run finished on. */
  fineTiles: number; tiles: number;
  samples: PondRestSample[]; [key: string]: unknown;
}

/** A Dawn device that lives for `run`. */
export async function withPondRestDevice(label: string, run: (device: GPUDevice) => Promise<void>): Promise<void> {
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    await run(device); await device.queue.onSubmittedWorkDone();
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); }
}

/**
 * One resting-pond arm on the hero-garden-hose-x10 lattice. `argv` holds
 * `--key=value` options (the last occurrence wins, so a batch entry overrides
 * the shared ones):
 *   --method=uniform|nb-flip  select the solver used in the app.
 *   --profile=lane|app      app includes UI timestep and detail defaults before
 *                            scene overrides; lane retains the test defaults.
 *   --detail=scene|none      none: Requested policy with no Fine region (all 4h
 *                            except whatever the solver still promotes).
 *   --basin=pond|flat|floor|open|slope|terrace|step|vessels
 *       pond: the authored scene. flat: a flat terrain floor at
 *       12 + --floor-offset-cells (default 3: the pond's median 5.6-cell depth),
 *       so the floor tile has open fraction (4 - offset)/4. floor: the same
 *       floor as solid voxels (binary h cells; the f32 terrain plane sits a
 *       rounding above its lattice row and closes ~1e-6 of it). open: no terrain
 *       and no solid at all (the floor is the container). slope: a planar
 *       terrain height = floor + sx*x + sz*z cells (--slope-x, --slope-z, rise
 *       per cell, zero height at the -x/-z corner). terrace: flat floor, the
 *       +x half raised by --step-cells. step: flat floor with a 16x16-cell
 *       block of --step-cells in the middle.
 *   --fine-box=x0,y0,z0,x1,y1,z1  one Fine region in h cells (with --detail=none:
 *                            the only h tiles besides what the solver promotes).
 *   --toggle-box=x0,y0,z0,x1,y1,z1[:n]  a Fine region drawn before every odd
 *                            frame and removed before every even one, for the
 *                            first n frames (default 10), each of them sampled.
 *   --switch-values=<json>   runtime values applied before --switch-frame=<f>.
 *   --hold=<f,f>             frames that apply their toggle or switch and sample without advancing
 *                            (the remapped state itself; the next frame advances over the gap).
 *   --waterline-cells=<n>    absolute waterline in h cells (else the scene's
 *                            fill, plus --waterline-shift-cells).
 *   --excess-rows=on         per h row, the cells holding more than their capacity (stderr).
 *   --tile-dump=<f,f>        write the waterline tile rows' represented state at
 *                            those frames (artifacts/pond-rest/<arm>-tiles-<f>.json).
 *   --root-dump=on           a fatal frame's worst all-4h root pressure rows (stderr).
 *   --velocity-dump=x0,x1,y0,y1,z:f,f  the + face velocity texels on an x-y window (stderr).
 *   --root-rows=X0,X1,Y0,Y1,Z:f,f  the all-4h root's pressure error, rhs, residual and phi on a tile window (stderr).
 *   --vertex-dump=x0,x1,y0,y1,z:f,f  vertex phi against the authored plane on an x-y window (stderr;
 *                            --vertex-unit=<cells>, default 1e-3).
 *   --volume-dump=x0,x1,y0,y1,z:f,f  owner V against the authored plane's fill on an x-y window (stderr).
 *   --surface-window=<n>    search this many cells above/below the initial level (default 4).
 *   --top=<n>                the n fastest wet owners at each sampled frame (stderr).
 *   --band-dump=x0,x1,y0,y1,z:f,f  the h band's pressure against hydrostatic and its
 *                            row residuals on an x-y window at those frames (stderr).
 *   --tile-rows=<lo,hi>      the tile rows --tile-dump writes (default: two under
 *                            the waterline tile to one over it).
 *   --wgsl-patch='[["regex","replacement"],...]'  A/B candidates patched into
 *       every compiled module of this arm; each must match at least once.
 */
export async function runPondRestArm(device: GPUDevice, argv: readonly string[]): Promise<PondRestResult> {
  const arg = (key: string, fallback: string) => argv.findLast(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
  const arm = arg("arm", "baseline");
  const methodName = arg("method", "uniform");
  assert.ok(methodName === "uniform" || methodName === "nb-flip");
  const method = methodName === "nb-flip" ? uniformNarrowBandMethod : uniformVolumeMethod;
  const frames = Number(arg("frames", "30"));
  const overrides = JSON.parse(arg("values", "{}"));
  const out = arg("out", `artifacts/pond-rest/${arm}.json`);
  const detail = arg("detail", "scene"), basin = arg("basin", "pond");
  assert.ok(detail === "scene" || detail === "none");
  assert.ok(["pond", "flat", "floor", "open", "slope", "terrace", "step", "vessels"].includes(basin));
  const read = (texture: GPUTexture) => readMixedTexture(device, texture);
  let solver: WebGPUUniformReferenceSolver | undefined;
  const compiler = gpuCompilationManagerFor(device), createModule = compiler.createShaderModule;
  try {
    const definition = getSceneDefinition("hero-garden-hose-x10");
    const scene = sceneDocument(definition);
    scene.fluid.inflow = { ...scene.fluid.inflow!, enabled: arg("hose", "off") === "on" };
    const cell = scene.voxelDomain.finestCellSize_m, c = scene.container;
    const columnsX = Math.round(c.width_m / cell), columnsZ = Math.round(c.depth_m / cell);
    const floor = 12 + Number(arg("floor-offset-cells", "3"));
    const flatTerrain = (height: (x: number, z: number) => number) => {
      // Node heights on the cell lattice with a one-node margin: bilinear
      // sampling of a plane is that plane at every column centre.
      const nx = columnsX + 3, nz = columnsZ + 3, heights_m: number[] = [];
      for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) heights_m.push(Math.max(0, height(i - 1, j - 1)) * cell);
      scene.terrain = { baseHeight_m: floor * cell, features: [], grid: { kind: "grid", origin_m: { x: -0.5 * c.width_m - cell, z: -0.5 * c.depth_m - cell }, spacing_m: cell, size: { nx, nz }, heights_m } };
    };
    if (basin !== "pond") { scene.solidVoxels = []; scene.rigidBodies = []; scene.scenery = undefined; }
    if (basin === "flat") {
      // Same domain, resolution and waterline as the pond. Footprint differs,
      // so this isolates a group of geometric effects, not curvature alone.
      scene.terrain = { baseHeight_m: floor * cell, features: [] };
    } else if (basin === "floor") {
      scene.terrain = undefined;
      scene.solidVoxels = [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [columnsX, floor, columnsZ] }];
    } else if (basin === "open") scene.terrain = undefined;
    else if (basin === "slope") {
      const sx = Number(arg("slope-x", "0")), sz = Number(arg("slope-z", "0"));
      flatTerrain((x, z) => floor + sx * x + sz * z);
    } else if (basin === "terrace") {
      const step = Number(arg("step-cells", "4"));
      // Piecewise constant on columns: a node belongs to the column on its +x side.
      scene.terrain = undefined;
      scene.solidVoxels = [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [columnsX, floor, columnsZ] },
        { operation: "fill", minimum: [columnsX / 2, floor, 0], maximumExclusive: [columnsX, floor + step, columnsZ] }];
    } else if (basin === "step") {
      const step = Number(arg("step-cells", "4")), x0 = columnsX / 2 - 8, z0 = columnsZ / 2 - 8;
      scene.terrain = undefined;
      scene.solidVoxels = [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [columnsX, floor, columnsZ] },
        { operation: "fill", minimum: [x0, floor, z0], maximumExclusive: [x0 + 16, floor + step, z0 + 16] }];
    }
    // vessels: two chambers in a solid block over the floor, x0..xw and
    // xw+4..x1 by z0..z1 (--vessel=x0,xw,x1,z0,z1, h cells; xw a tile edge),
    // joined by one window through the dividing tile column
    // (--window=y0,y1,z0,z1). The waterline fills both; the low chamber starts
    // --left-shift-cells higher. Samples carry each chamber's level from its V
    // (the h mask's open area per row), the result the common level the
    // total V settles at.
    let vessel: { xw: number } | undefined;
    if (basin === "vessels") {
      const [x0, xw, x1, z0, z1] = arg("vessel", "52,68,88,40,56").split(",").map(Number) as [number, number, number, number, number];
      const [wy0, wy1, wz0, wz1] = arg("window", "16,18,48,52").split(",").map(Number) as [number, number, number, number];
      const top = Math.round(c.height_m / cell);
      assert.ok(xw % 4 === 0 && x0 < xw && xw + 4 < x1 && wy0 >= floor && wz0 >= z0 && wz1 <= z1);
      scene.terrain = undefined;
      scene.solidVoxels = ([[[0, 0, 0], [columnsX, floor, columnsZ]], [[0, floor, 0], [x0, top, columnsZ]], [[x1, floor, 0], [columnsX, top, columnsZ]],
        [[x0, floor, 0], [x1, top, z0]], [[x0, floor, z1], [x1, top, columnsZ]],
        [[xw, floor, z0], [xw + 4, wy0, z1]], [[xw, wy1, z0], [xw + 4, top, z1]], [[xw, wy0, z0], [xw + 4, wy1, wz0]], [[xw, wy0, wz1], [xw + 4, wy1, z1]]] as [number[], number[]][])
        .filter(([lo, hi]) => lo.every((v, a) => v < hi[a]!)).map(([lo, hi]) => ({ operation: "fill" as const, minimum: lo as [number, number, number], maximumExclusive: hi as [number, number, number] }));
      vessel = { xw };
      const lift = Number(arg("left-shift-cells", "8")), base = Number(arg("waterline-cells", "22"));
      scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: (x0 - 2) * cell - 0.5 * c.width_m, y: 0, z: (z0 - 2) * cell - 0.5 * c.depth_m },
        max_m: { x: (xw + 2) * cell - 0.5 * c.width_m, y: (base + lift) * cell, z: (z1 + 2) * cell - 0.5 * c.depth_m } }];
    }
    if (arg("waterline-cells", "scene") !== "scene") c.fillFraction = Number(arg("waterline-cells", "0")) * cell / c.height_m;
    c.fillFraction += Number(arg("waterline-shift-cells", "0")) * cell / c.height_m;
    if (detail === "none") scene.fluid.refinementRegions = [];
    const fineBox = arg("fine-box", "");
    const box = fineBox ? fineBox.split(",").map(Number) : undefined;
    const fineRegion = (id: string, cells: string) => {
      const [x0, y0, z0, x1, y1, z1] = cells.split(",").map(Number) as [number, number, number, number, number, number];
      return { id, rule: "minimum-cell-size" as const, minimumCellSize_cells: 1, maximumCellSize_cells: 1,
        min_m: { x: x0 * cell - 0.5 * c.width_m, y: y0 * cell, z: z0 * cell - 0.5 * c.depth_m }, max_m: { x: x1 * cell - 0.5 * c.width_m, y: y1 * cell, z: z1 * cell - 0.5 * c.depth_m } };
    };
    if (fineBox) scene.fluid.refinementRegions = [fineRegion("fine", fineBox)];
    const [toggleBox, toggleCount] = arg("toggle-box", "").split(":"), toggleFrames = toggleBox ? Number(toggleCount ?? "10") : 0;
    const switchFrame = Number(arg("switch-frame", "0")), switchValues = JSON.parse(arg("switch-values", "{}"));
    const holds = arg("hold", "").split(",").filter(Boolean).map(Number);
    if (arg("sigma", "scene") !== "scene") scene.fluid.surfaceTension_N_m = Number(arg("sigma", "0"));
    if (arg("gravity", "scene") !== "scene") scene.fluid.gravity_m_s2 = { x: 0, y: Number(arg("gravity", "0")), z: 0 };
    // Keep the historical full-solve control, but also exercise the authored UI
    // profile so a passing reference solve cannot hide an overly loose default.
    const pressureMode = arg("pressure", "full");
    assert.ok(pressureMode === "full" || pressureMode === "scene");
    const profile = arg("profile", "lane");
    assert.ok(profile === "lane" || profile === "app");
    const values = resolveMethodValues(method, "balanced", {
      ...(profile === "app" ? method.appDefaults : {}),
      ...(definition.methodProfile?.methodId === method.id ? definition.methodProfile.overrides : {}),
      ...(pressureMode === "full" ? { pressureResidualTolerance: 0 } : {}),
      ...(detail === "none" ? { detailPolicy: "requested" } : {}), ...overrides,
    });
    const sampleStep = profile === "app" ? uniformFixedStep_s(values.timeStep) ?? scene.numerics.maxDt_s : 1 / 30;
    const weightOverride = arg("jacobi-weight", "production");
    const weight = weightOverride === "production" ? 0.6666667 : Number(weightOverride);
    let dampingReplacements = 0;
    const patches = (JSON.parse(arg("wgsl-patch", "[]")) as [string, string][]).map(([pattern, replacement]) => ({ pattern: new RegExp(pattern, "g"), replacement, count: 0 }));
    if (weightOverride !== "production" || patches.length) {
      assert.ok(weight > 0 && weight <= 1);
      compiler.createShaderModule = descriptor => {
        let code = descriptor.code;
        if (weightOverride !== "production") code = code.replace(/const MG_JACOBI_WEIGHT:f32=[0-9.]+;/g, () => { dampingReplacements++; return `const MG_JACOBI_WEIGHT:f32=${weight.toFixed(8)};`; });
        for (const p of patches) code = code.replace(p.pattern, (...match) => { p.count++; return p.replacement.replace(/\$(\d)/g, (_, n) => String(match[Number(n)] ?? "")); });
        return createModule.call(compiler, { ...descriptor, code });
      };
    }
    solver = await method.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    if (weightOverride !== "production") assert.ok(dampingReplacements > 0, "experimental damping must patch the compiled shader");
    for (const p of patches) assert.ok(p.count > 0, `A/B patch ${p.pattern} matched no compiled shader`);
    const { nx, ny, nz } = solver.info, h = c.height_m / ny, tx = nx / 4, ty = ny / 4, tz = nz / 4;
    // Per-cell open fraction from the solver's solid sampler: the presented
    // open-fraction field is the surface target after frame 1 and, in packed
    // detail storage, one value per 4h tile.
    const open = await readUniformSolidFractions(device, solver, true);
    // Mean open fraction of each tile: a 4h owner's capacity.
    const tileOpen = new Float32Array(tx * ty * tz);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) tileOpen[(x >> 2) + tx * ((y >> 2) + ty * (z >> 2))]! += open[x + nx * (y + ny * z)]! / 64;
    const level = c.height_m * c.fillFraction / h;
    const surfaceY = Math.min(ny - 1, Math.floor(level)), surfaceTile = surfaceY >> 2;
    // Interior at h: the surface cell's 5x5 neighbourhood one cell down is
    // open. At 4h (no detail): the 3x3 tile neighbourhood of the surface tile
    // and the tile below it is uncut. Everything else is shoreline.
    const uncutTiles = (x: number, z: number) => {
      for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++) for (const y of [surfaceTile, surfaceTile - 1]) {
        const qx = (x >> 2) + dx, qz = (z >> 2) + dz;
        if (qx < 0 || qz < 0 || y < 0 || qx >= tx || qz >= tz) continue;
        if (tileOpen[qx + tx * (y + ty * qz)]! < 0.999) return false;
      }
      return true;
    };
    const columns: { x: number; z: number; interior: boolean }[] = [];
    for (let z = 3; z < nz - 3; z++) for (let x = 3; x < nx - 3; x++) {
      if (open[x + nx * (surfaceY - 1 + ny * z)]! < 0.999) continue;
      let interior = true;
      if (detail === "none") interior = uncutTiles(x, z);
      else for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) if (open[x + dx + nx * (surfaceY - 2 + ny * (z + dz))]! < 0.999) interior = false;
      columns.push({ x, z, interior });
    }
    // --interior=optional: a film over a cut floor has no uncut neighbourhood; its columns are all shoreline.
    assert.ok(arg("interior", "required") === "optional" || columns.some(c => c.interior), "the basin has interior columns");
    // Open cells per h row of a column range, and the level a volume fills it to.
    const rowsOpen = (xa: number, xb: number) => Array.from({ length: ny }, (_, y) => { let n = 0; for (let z = 0; z < nz; z++) for (let x = xa; x < xb; x++) n += open[x + nx * (y + ny * z)]!; return n; });
    const levelOf = (rows: number[], amount: number) => { let y = 0; while (y < ny && (rows[y]! <= 0 || amount >= rows[y]!)) { amount -= rows[y]!; y++; } return y < ny ? y + Math.max(0, amount) / rows[y]! : ny; };
    const chamberRows = vessel ? [rowsOpen(0, vessel.xw), rowsOpen(vessel.xw + 4, nx)] : [];
    let vesselFinal: number | undefined;
    // Deepest wet column, for the physical drift bound (residual x depth x time).
    let depthCells = 0;
    for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { let wet = 0; for (let y = 0; y < ny && y < level; y++) if (open[x + nx * (y + ny * z)]! > 1e-5) wet += Math.min(1, level - y); depthCells = Math.max(depthCells, wet); }
    const mg = (solver as unknown as { pressureMultigrid: WebGPUUniformPressureMultigrid }).pressureMultigrid;
    const frameFields = (solver as unknown as { mixedFrame: { fields: { correction: GPUTexture }; presentation: { phi: GPUBufferBinding } } }).mixedFrame;
    const snapshots = new Map<GPUTexture, GPUTexture>();
    const noRecovery = arg("recovery", "on") === "off";
    if (arg("pressure-dump", "off") === "on" || noRecovery) {
      const fields = (mg as unknown as { scratchFields: UniformTexturePages }).scratchFields;
      if (arg("pressure-dump", "off") === "on") for (const l of mg.levels) for (const t of [l.volume[0], l.coefficients, l.pressure[0], l.rhs[0], l.minimum[0]]) snapshots.set(t, fields.snapshotTexture(t));
      const encode = mg.encode.bind(mg);
      mg.encode = (...args: Parameters<typeof mg.encode>) => {
        if (noRecovery) {
          assert.equal(values.pressureResidualTolerance, 0, "no-recovery control requires the fixed schedule");
          const plan = mg as unknown as { finishStart: number; finalStart: number; plan: unknown[] };
          encode(args[0], args[1], args[2], args[3], args[4], { start: 0, end: plan.finishStart, initialize: true, publish: false });
          encode(args[0], args[1], args[2], args[3], args[4], { start: plan.finalStart, end: plan.plan.length, initialize: false, publish: true });
        } else encode(...args);
        // Scratch is reused by later fluid stages. Capture while pressure still
        // owns it, rather than reading the placeholder texture after the frame.
        for (const t of snapshots.keys()) fields.encodeSnapshot(args[0], t);
      };
    }
    const readPressure = (t: GPUTexture) => read(snapshots.get(t) ?? t);
    const samples: PondRestSample[] = [];
    const stepTimes_ms: number[] = [];
    const tileDumps = arg("tile-dump", "").split(",").filter(Boolean).map(Number);
    let pressureDiagnostic: unknown, tiles: Uint32Array = new Uint32Array(0);
    const dumpRootRows = async (frame: number) => {
      const rootRows = arg("root-rows", "").split(":");
      if (rootRows[0] && (rootRows[1] ?? "").split(",").map(Number).includes(frame)) {
        const [X0, X1, Y0, Y1, Z] = rootRows[0].split(",").map(Number) as [number, number, number, number, number];
        type View = { buffer: GPUBuffer; offset?: number };
        const root = (solver as unknown as { mixedFrame: { levels: { pressure: View; rhs: View[]; phi: View; residual: View }[] } }).mixedFrame.levels[0]!;
        const load = async (v: View) => { const all = await readMixedBuffer(device, v.buffer); return (i: number) => all[(v.offset ?? 0) / 4 + i]!; };
        const p = await load(root.pressure), b = await load(root.rhs[0]!), phi = await load(root.phi), r = await load(root.residual);
        const at = (X: number, Y: number) => X + tx * (Y + ty * Z), depth = (Y: number) => level - (4 * Y + 2);
        const ratios: number[] = []; for (let Y = Y0; Y <= Y1; Y++) for (let X = X0; X <= X1; X++) if (p(at(X, Y)) > 0) ratios.push(p(at(X, Y)) / depth(Y));
        ratios.sort((a, c) => a - c); const scale = ratios[ratios.length >> 1] ?? 1;
        const table = (name: string, value: (X: number, Y: number) => string) => console.error(`${arm} frame ${frame} root ${name}, tiles x ${X0}..${X1}, z ${Z}:\n` + Array.from({ length: Y1 - Y0 + 1 }, (_, k) => `${String(Y1 - k).padStart(3)} ` + Array.from({ length: X1 - X0 + 1 }, (_, i) => value(X0 + i, Y1 - k).padStart(10)).join("")).join("\n"));
        table("pressure error (1e-6 cells)", (X, Y) => (1e6 * (p(at(X, Y)) / scale - depth(Y))).toFixed(0));
        table("rhs", (X, Y) => b(at(X, Y)).toPrecision(7));
        table("residual", (X, Y) => r(at(X, Y)).toExponential(2));
        table("phi (cells)", (X, Y) => (phi(at(X, Y)) / h).toFixed(3));
      }
    };
    for (let frame = 0; frame <= frames; frame++) {
      if (frame) {
        const started = performance.now();
        if (frame <= toggleFrames) {
          const drawn = structuredClone(scene);
          drawn.fluid.refinementRegions = [...(scene.fluid.refinementRegions ?? []), ...(frame % 2 ? [fineRegion("toggle", toggleBox!)] : [])];
          solver.applySceneUniforms(drawn); await solver.pipelinesPrepared();
        }
        if (frame === switchFrame) { solver.applyRuntimeValues({ ...values, ...switchValues }); await solver.pipelinesPrepared(); }
        try { if (!holds.includes(frame)) { assert.ok(solver.advanceTo(frame * sampleStep, [])); await solver.awaitFrameCompletion(); } await device.queue.onSubmittedWorkDone(); }
        catch (error) {
          await dumpRootRows(frame);
          // --root-dump=on: a rejected solve's worst all-4h root rows (stderr),
          // read where the withheld projection left them.
          if (arg("root-dump", "off") === "on") {
            type View = { buffer: GPUBuffer; offset?: number };
            const root = (solver as unknown as { mixedFrame: { levels: { pressure: View; rhs: View[]; minimum: View[]; phi: View; residual: View; topology?: { buffer: View } }[] } }).mixedFrame.levels[0]!;
            const arenas = new Map<GPUBuffer, Float32Array>();
            const load = async (v: View) => { const all = arenas.get(v.buffer) ?? await readMixedBuffer(device, v.buffer); arenas.set(v.buffer, all); return (i: number) => all[(v.offset ?? 0) / 4 + i]!; };
            const p = await load(root.pressure), b = await load(root.rhs[0]!), low = await load(root.minimum[0]!), phi = await load(root.phi), r = await load(root.residual);
            const record = root.topology ? await load(root.topology.buffer) : undefined, cells = tx * ty * tz, mid = tz >> 1;
            // Owners of the mid-z slice, then its x and y wall slots and the low z wall's (umBoundaryIndex of the all-4h layout).
            const rows: { row: string; record?: number[]; residual: number; pressure: number; rhs: number; minimum: number; phi_cells?: number }[] = [];
            const push = (row: string, at: number, key?: number) => rows.push({ row, record: record ? [0, 1, 2, 3].map(c => +record(4 * at + c).toFixed(4)) : undefined,
              residual: r(at), pressure: p(at), rhs: b(at), minimum: low(at), phi_cells: key === undefined ? undefined : +(phi(key) / h).toFixed(3) });
            for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) push(`o ${x},${y},${mid}`, x + tx * (y + ty * mid), x + tx * (y + ty * mid));
            for (let side = 0; side < 2; side++) for (let y = 0; y < ty; y++) push(`x${side} ${y},${mid}`, cells + side * ty * tz + y + ty * mid);
            for (let side = 0; side < 2; side++) for (let x = 0; x < tx; x++) push(`y${side} ${x},${mid}`, cells + 2 * ty * tz + side * tx * tz + x + tx * mid);
            for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) push(`z0 ${x},${y}`, cells + 2 * (ty * tz + tx * tz) + x + tx * y);
            const worst = rows.reduce((m, row) => Math.max(m, Math.abs(row.residual)), 0);
            console.error(`${arm} frame ${frame} root rows over a tenth of the worst residual ${worst} (mid-z owners, wall slots):`);
            for (const row of rows.filter(row => Math.abs(row.residual) > 0.1 * worst).slice(0, 160)) console.error(JSON.stringify(row));
          }
          throw error;
        }
        if (frame > 5) stepTimes_ms.push(performance.now() - started);
      }
      const stats = await solver.readStats();
      if (frame === 1 && arg("pressure-dump", "off") === "on") {
        // Diagnostic-only access: inspect the actual solver hierarchy, without
        // adding a production readback or changing any pressure dispatch.
        const hierarchy = [];
        for (const l of mg.levels) {
          const topology = await readPressure(l.volume[0]), coefficients = await readPressure(l.coefficients);
          let liquid = 0, openLiquid = 0, mixedLiquid = 0, solidLiquid = 0;
          for (let i = 0; i < coefficients.length / 4; i++) if ((coefficients[4 * i + 3]! & 1) !== 0) {
            liquid++;
            const [wx, wy] = l.dimensions, x = i % wx - 1, y = Math.floor(i / wx) % wy - 1, z = Math.floor(i / wx / wy) - 1;
            // Finest topology aliases cycle scratch after coefficient baking.
            const capacity = l === mg.levels[0] ? (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz ? open[x + nx * (y + ny * z)]! : 0) : topology[4 * i]!;
            if (capacity <= 1e-5) solidLiquid++; else { openLiquid++; if (capacity < 0.99999) mixedLiquid++; }
          }
          hierarchy.push({ dimensions: l.dimensions.map(v => v - 2), liquid, openLiquid, mixedLiquid, solidLiquid });
        }
        const l = mg.levels[0]!, [wx, wy, wz] = l.dimensions;
        const p = await readPressure(l.pressure[0]), rhs = await readPressure(l.rhs[0]), minimum = await readPressure(l.minimum[0]);
        const c = await readPressure(l.coefficients);
        const worst: { cell: number[]; residual: number; open: number; pressure: number; rhs: number; diagonal: number }[] = [];
        const liquid = (i: number) => i >= 0 && i < p.length && (c[4 * i + 3]! & 1) !== 0;
        for (let z = 1; z < wz - 1; z++) for (let y = 1; y < wy - 1; y++) for (let x = 1; x < wx - 1; x++) {
          const i = x + wx * (y + wy * z); if (!liquid(i)) continue;
          let ap = 0, diagonal = 0;
          for (const [axis, stride] of [1, wx, wx * wy].entries()) for (const sign of [-1, 1]) {
            const j = i + sign * stride, a = c[4 * (sign > 0 ? i : j) + axis]!;
            diagonal += a; ap += a * (p[i]! - (liquid(j) ? p[j]! : 0));
          }
          const r = rhs[i]! - ap, gap = Math.max(0, p[i]! - minimum[i]!);
          const projected = Math.max(r < 0 && -r >= gap * diagonal ? gap * diagonal : Math.abs(r), Math.max(0, minimum[i]! - p[i]!) * diagonal);
          const residual = projected * sampleStep / scene.fluid.density_kg_m3;
          if (worst.length < 20 || residual > worst.at(-1)!.residual) {
            worst.push({ cell: [x - 1, y - 1, z - 1], residual, open: open[x - 1 + nx * (y - 1 + ny * (z - 1))]!, pressure: p[i]!, rhs: rhs[i]!, diagonal });
            worst.sort((a, b) => b.residual - a.residual); worst.length = Math.min(20, worst.length);
          }
        }
        pressureDiagnostic = { hierarchy, worst, cpuResidualNote: "Double precision reconstruction from GPU fields; compare with GPU f32 residual, not bitwise.", gpuResidual: stats.uniformCM11aFineResidualInfinity };
        assert.ok(worst.length > 0, "capture populated pressure rows before scratch reuse");
        assert.ok(Math.abs(worst[0]!.residual - stats.uniformCM11aFineResidualInfinity!) < 1e-4, "CPU reconstruction agrees with the GPU residual");
        console.log(JSON.stringify({ arm, pressureDiagnostic }));
      }
      if (frame > 3 && frame > toggleFrames && frame % 10 && frame !== frames && !(switchFrame && (frame === switchFrame || frame === switchFrame + 1)) && !tileDumps.includes(frame) && ![arg("cell-dump", ""), arg("velocity-dump", ""), arg("root-rows", ""), arg("band-tiles", ""), arg("band-column", "")].some(dump => dump.split(":")[1]?.split(",").map(Number).includes(frame))) continue;
      // Coarse owners carry averages; their fine backing texels and hanging
      // vertices are not independent state. Measure the represented fields.
      const fields = await readUniformFields(device, solver), volume = fields.density;
      tiles = fields.tiles;
      const velocity = await read(solver.velocityTexture);
      // An owner's + faces: its own texel at h, its anchors o+3e_a at 4h.
      const velocityAt = (i: number) => {
        const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nx / ny), w = fields.widthAt(x, y, z), o = [x - x % w, y - y % w, z - z % w];
        return [0, 1, 2].map(a => velocity[4 * (o[0]! + (a === 0 ? w - 1 : 0) + nx * (o[1]! + (a === 1 ? w - 1 : 0) + ny * (o[2]! + (a === 2 ? w - 1 : 0)))) + a]!);
      };
      const regions = { all: [] as number[], interior: [] as number[] };
      let missing = 0;
      const surfaceWindow = Number(arg("surface-window", "4"));
      for (const c of columns) {
        let height: number | undefined;
        for (let y = Math.max(0, surfaceY - surfaceWindow + 1); y < Math.min(ny, surfaceY + surfaceWindow); y++) {
          const a = fields.vertex(c.x, y, c.z), b = fields.vertex(c.x, y + 1, c.z);
          if (a <= 0 && b > 0) height = y - a / (b - a);
        }
        if (height === undefined) { missing++; continue; }
        regions.all.push((height - level) * h * 1000);
        if (c.interior) regions.interior.push((height - level) * h * 1000);
      }
      const surface = Object.fromEntries(Object.entries(regions).map(([key, a]) => [key, { count: a.length, mean_mm: a.reduce((s, v) => s + v, 0) / a.length, rms_mm: Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length), max_mm: Math.max(...a.map(Math.abs)), range_mm: Math.max(...a) - Math.min(...a) }])) as PondRestSample["surface"];
      let maxSpeed = 0, coarseSpeed = 0, excess = 0, sum = 0, maxCell = 0;
      const chamberSums = [0, 0];
      // --excess-rows=on: where the inaccessible mass sits, per h row (stderr).
      const excessRows = arg("excess-rows", "off") === "on" ? Array.from({ length: ny }, () => ({ cells: 0, excess: 0, max: 0, capacity: 0 })) : undefined;
      for (let i = 0; i < volume.length; i++) {
        const v = volume[i]!; assert.ok(Number.isFinite(v)); sum += v;
        const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nx / ny), w = fields.widthAt(x, y, z);
        if (vessel && (x < vessel.xw || x >= vessel.xw + 4)) chamberSums[x < vessel.xw ? 0 : 1]! += v;
        // Inaccessible mass at the owner's width: a 4h owner holds its tile's mean open fraction.
        const capacity = w === 1 ? open[i]! : tileOpen[(x >> 2) + tx * ((y >> 2) + ty * (z >> 2))]!;
        excess += Math.max(0, v - capacity);
        if (excessRows && v > capacity) { const row = excessRows[y]!; row.cells++; row.excess += v - capacity; if (v - capacity > row.max) { row.max = v - capacity; row.capacity = capacity; } }
        if (v > 1e-5) { const speed = Math.hypot(...velocityAt(i)); if (speed > maxSpeed) { maxSpeed = speed; maxCell = i; } if (w !== 1 && speed > coarseSpeed) coarseSpeed = speed; }
      }
      // The seam: faces on the Fine box's planes, the + face of the cell under each (a 4h owner's patch texel or an h cell).
      let seamSpeed: number | undefined;
      if (box) {
        seamSpeed = 0; const n = [nx, ny, nz];
        for (let a = 0; a < 3; a++) for (const plane of [box[a]!, box[a + 3]!]) {
          if (plane <= 0 || plane >= n[a]!) continue;
          const u = (a + 1) % 3, v = (a + 2) % 3;
          for (let j = Math.max(0, box[v]!); j < Math.min(n[v]!, box[v + 3]!); j++) for (let i = Math.max(0, box[u]!); i < Math.min(n[u]!, box[u + 3]!); i++) {
            const q = [0, 0, 0]; q[a] = plane - 1; q[u] = i; q[v] = j;
            const low = q[0]! + nx * (q[1]! + ny * q[2]!); q[a] = plane; const high = q[0]! + nx * (q[1]! + ny * q[2]!);
            if (volume[low]! > 1e-5 || volume[high]! > 1e-5) seamSpeed = Math.max(seamSpeed, Math.abs(velocity[4 * low + a]!));
          }
        }
      }
      if (excessRows) console.error(`${arm} frame ${frame} excess rows: ${JSON.stringify(excessRows.map((row, y) => ({ y, ...row })).filter(row => row.cells))}`);
      // --top=<n>: the n fastest wet owners (stderr): position, width, open fraction, V, + face velocity in mm/s.
      const top = Number(arg("top", "0"));
      if (top > 0 && frame) {
        const fast: { i: number; speed: number }[] = [];
        for (let i = 0; i < volume.length; i++) {
          const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nx / ny), w = fields.widthAt(x, y, z);
          if (volume[i]! <= 1e-5 || x % w || y % w || z % w) continue;
          fast.push({ i, speed: Math.hypot(...velocityAt(i)) });
        }
        fast.sort((a, b) => b.speed - a.speed);
        console.error(`${arm} frame ${frame} fastest: ${fast.slice(0, top).map(({ i }) => { const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nx / ny); return `[${x},${y},${z}] w${fields.widthAt(x, y, z)} open ${open[i]!.toFixed(3)} V ${volume[i]!.toFixed(4)} u ${velocityAt(i).map(v => (1000 * v).toFixed(3)).join(",")}`; }).join(" | ")}`);
      }
      // The split pressure authority leaves the correction and the pressure
      // phi on the all-4h pressure owners (tile origins, tile order).
      let correction = 0, airBelow = 0, liquidAbove = 0;
      if (frame) {
        const rate = await read(frameFields.fields.correction), phi = await readMixedBuffer(device, frameFields.presentation.phi.buffer as GPUBuffer);
        for (let t = 0; t < tx * ty * tz; t++) {
          const x = t % tx, y = Math.floor(t / tx) % ty, z = Math.floor(t / tx / ty);
          correction = Math.max(correction, Math.abs(rate[4 * x + nx * (4 * y + ny * 4 * z)]!));
          if (tileOpen[t]! <= 1e-5) continue;
          // umPressureLiquid: liquid only deeper than theta_min of the 4h spacing.
          // A centre within the 0.5 mm surface bound of that crossing may be either.
          const depth = level - (4 * y + 2), margin = 4 * UNIFORM_MIXED_THETA_MIN, slack = 0.5e-3 / h, liquid = phi[t]! < -margin * h;
          if (depth > margin + slack && !liquid) airBelow++;
          if (depth < margin - slack && liquid) liquidAbove++;
        }
      }
      // The mixed path reports its accepted residual (the fine one) and has no
      // recovery: a rejected solve is a fatal frame, which fails the run.
      // Frame 0 has solved nothing yet.
      // The common level: the total V over every open row of the block.
      if (vessel && frame === 0) vesselFinal = levelOf(rowsOpen(0, nx), sum);
      const sample: PondRestSample = { frame, chambers: vessel ? chamberSums.map((amount, k) => levelOf(chamberRows[k]!, amount)) : undefined, surface, missing, maxSpeed, coarseSpeed, fineTiles: tiles.reduce((n, word) => n + (word & 0x80000000 ? 1 : 0), 0), seamSpeed, maxCell: [maxCell % nx, Math.floor(maxCell / nx) % ny, Math.floor(maxCell / nx / ny)], sum, excess, correction, airBelow, liquidAbove,
        residual: stats.uniformCM11aFineResidualInfinity ?? stats.uniformPressureAcceptedResidual ?? (frame === 0 ? 0 : undefined), acceptedResidual: stats.uniformPressureAcceptedResidual, bandResidual: stats.uniformPressureBandResidual, bandTiles: stats.uniformPressureBandTiles, recovery: stats.uniformPressureRecoverySweeps ?? 0, full: stats.uniformCM11aFullCyclesExecuted, vcycles: stats.uniformCM11aVCyclesExecuted };
      // --dump=on: the fastest cell's neighbourhood at frame 1. --cell-dump=x,z,r:f,f:
      // --root-rows=X0,X1,Y0,Y1,Z:f,f: the all-4h root's rows on a tile window (stderr): pressure error against hydrostatic
      // (1e-6 cells of head, about the window's median), the rhs, the residual and phi.
      await dumpRootRows(frame);
      // --vertex-dump=x0,x1,y0,y1,z:f,f: vertex phi minus the authored plane on an x-y window (1e-3 cells; stderr).
      const vertexDump = arg("vertex-dump", "").split(":");
      if (vertexDump[0] && (vertexDump[1] ?? "").split(",").map(Number).includes(frame)) {
        const [x0, x1, y0, y1, z] = vertexDump[0].split(",").map(Number) as [number, number, number, number, number];
        const unit = Number(arg("vertex-unit", "1e-3"));
        console.error(`${arm} frame ${frame} vertex phi - plane (${unit} cells), x ${x0}..${x1}, z ${z}:\n` + Array.from({ length: y1 - y0 + 1 }, (_, r) => `${String(y1 - r).padStart(3)}` + Array.from({ length: x1 - x0 + 1 }, (_, c) => ((fields.vertex(x0 + c, y1 - r, z) / h - (y1 - r - level)) / unit).toFixed(0).padStart(7)).join("")).join("\n"));
      }
      // --volume-dump=x0,x1,y0,y1,z:f,f: V minus the authored plane's fill of the owner on an x-y window (1e-6 of the owner; a 4h owner at its origin texel; stderr).
      const volumeDump = arg("volume-dump", "").split(":");
      if (volumeDump[0] && (volumeDump[1] ?? "").split(",").map(Number).includes(frame)) {
        const [x0, x1, y0, y1, z] = volumeDump[0].split(",").map(Number) as [number, number, number, number, number];
        console.error(`${arm} frame ${frame} V - authored fill (1e-6), x ${x0}..${x1}, z ${z}:\n` + Array.from({ length: y1 - y0 + 1 }, (_, r) => { const y = y1 - r; return `${String(y).padStart(3)}${fields.widthAt(x0, y, z) === 1 ? "h" : " "}` + Array.from({ length: x1 - x0 + 1 }, (_, c) => {
          const x = x0 + c, w = fields.widthAt(x, y, z); if (x % w || y % w || z % w) return "      .";
          return (1e6 * (volume[x + nx * (y + ny * z)]! - Math.min(1, Math.max(0, (level - y) / w)))).toFixed(0).padStart(7);
        }).join(""); }).join("\n"));
      }
      // --velocity-dump=x0,x1,y0,y1,z:f,f: the raw + face texels on an x-y window (um/s; a 4h owner's non-anchor texels are not state).
      const velocityDump = arg("velocity-dump", "").split(":");
      if (velocityDump[0] && (velocityDump[1] ?? "").split(",").map(Number).includes(frame)) {
        const [x0, x1, y0, y1, z] = velocityDump[0].split(",").map(Number) as [number, number, number, number, number];
        for (const a of [0, 1, 2]) console.error(`${arm} frame ${frame} u${"xyz"[a]} (um/s), x ${x0}..${x1}, z ${z}:\n` + Array.from({ length: y1 - y0 + 1 }, (_, r) => `${String(y1 - r).padStart(3)}${fields.widthAt(x0, y1 - r, z) === 1 ? "h" : " "}` + Array.from({ length: x1 - x0 + 1 }, (_, c) => (1e6 * velocity[4 * (x0 + c + nx * (y1 - r + ny * z)) + a]!).toFixed(0).padStart(6)).join("")).join("\n"));
      }
      // --band-tiles=X0,X1,Y0,Y1,Z:f,f: per h tile of a tile window (stderr), in the root's rhs units (-rho div / dt):
      // the mean over the tile's 64 cells of the band's liquid row residuals after the solve (double, from the f32
      // iterate), the same in units of diagonal * ulp(p) per liquid row, and the tile's mean h divergence of the
      // published velocity over all cells and over the cells under the authored plane.
      const bandTiles = arg("band-tiles", "").split(":");
      if (bandTiles[0] && (bandTiles[1] ?? "").split(",").map(Number).includes(frame)) {
        const [X0, X1, Y0, Y1, Z] = bandTiles[0].split(",").map(Number) as [number, number, number, number, number];
        const band = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer; rows: GPUBuffer; iterate: GPUBuffer; slots: number; slotMapOffset: number } } }).mixedFrame.band;
        const index = new Uint32Array((await readMixedBuffer(device, band.index)).buffer), rows = await readMixedBuffer(device, band.rows), solve = await readMixedBuffer(device, band.iterate);
        const kindsWords = new Uint32Array(rows.buffer), N = 64 * band.slots, slotBase = band.slotMapOffset / 4;
        const rowOf = (x: number, y: number, zz: number) => {
          if (x < 0 || y < 0 || zz < 0 || x >= nx || y >= ny || zz >= nz) return -1;
          const slot = index[slotBase + (x >> 2) + tx * ((y >> 2) + ty * (zz >> 2))]!; if (!slot) return -1;
          const l = [x & 3, y & 3, zz & 3];
          return (slot - 1) * 64 + ((l[0]! + l[1]! + l[2]!) & 1) * 32 + (l[0]! >> 1) + 2 * l[1]! + 8 * l[2]!;
        };
        const halo = (q: number[], axis: number, side: number) => N + (axis === 0 ? side * ny * nz + q[1]! + ny * q[2]! : axis === 1 ? 2 * ny * nz + side * nx * nz + q[0]! + nx * q[2]! : 2 * (ny * nz + nx * nz) + side * nx * ny + q[0]! + nx * q[1]!);
        const ulp = (v: number) => Math.pow(2, Math.floor(Math.log2(Math.max(Math.abs(v), 1e-30))) - 23), scale = -scene.fluid.density_kg_m3 / sampleStep;
        // With the correction form the rows' last field is the start pressure and the iterate its correction.
        const fieldCount = Math.round(rows.length / N), baseOf = (c: number) => fieldCount === 16 || fieldCount === 22 ? rows[(fieldCount - 1) * N + c]! : 0;
        const face = (x: number, y: number, zz: number, a: number) => x < 0 || y < 0 || zz < 0 ? 0 : velocity[4 * (x + nx * (y + ny * zz)) + a]!;
        const tiles: { residual?: number; ulps?: number; liquid?: number; all: number; wet: number; pressure?: number }[] = [];
        for (let Y = Y1; Y >= Y0; Y--) for (let X = X0; X <= X1; X++) {
          let residual = 0, ulps = 0, liquid = 0, all = 0, wet = 0, pressure = 0, banded = false;
          for (let l = 0; l < 64; l++) {
            const x = 4 * X + (l & 3), y = 4 * Y + ((l >> 2) & 3), zz = 4 * Z + (l >> 4);
            const d = (face(x, y, zz, 0) - face(x - 1, y, zz, 0) + face(x, y, zz, 1) - face(x, y - 1, zz, 1) + face(x, y, zz, 2) - face(x, y, zz - 1, 2)) / h;
            all += scale * d / 64; if (y + 0.5 < level) wet += scale * d / 64;
            const c = rowOf(x, y, zz); if (c < 0) continue; banded = true;
            const kinds = kindsWords[2 * N + c]!; if (!(kinds & 0x80000000) || !(rows[N + c]! > 0)) continue;
            let off = 0;
            for (let f = 0; f < 6; f++) {
              const axis = f >> 1, sign = f & 1 ? 1 : -1, kind = (kinds >>> (3 * f)) & 7, q = [x, y, zz];
              if (kind === 1) { q[axis]! += sign; off += rows[(3 + f) * N + c]! * solve[rowOf(q[0]!, q[1]!, q[2]!)]!; }
              else if (kind === 2) off += rows[(3 + f) * N + c]! * solve[halo(q, axis, sign > 0 ? 1 : 0)]!;
            }
            const r = rows[c]! + off - rows[N + c]! * solve[c]!;
            residual += r / 64; ulps += r / (rows[N + c]! * ulp(solve[c]! + baseOf(c))); liquid++; pressure += solve[c]! + baseOf(c);
          }
          tiles.push(banded ? { residual, ulps: liquid ? ulps / liquid : 0, liquid, all, wet, pressure: liquid ? pressure / liquid : 0 } : { all, wet });
        }
        const table = (name: string, value: (t: typeof tiles[number]) => string) => console.error(`${arm} frame ${frame} band tiles ${name}, tiles x ${X0}..${X1}, z ${Z}:\n` + Array.from({ length: Y1 - Y0 + 1 }, (_, r) => `${String(Y1 - r).padStart(3)} ` + tiles.slice(r * (X1 - X0 + 1), (r + 1) * (X1 - X0 + 1)).map(t => value(t).padStart(10)).join("")).join("\n"));
        table("liquid rows", t => t.liquid === undefined ? "." : String(t.liquid));
        table("mean pressure (Pa)", t => t.pressure === undefined ? "." : t.pressure.toFixed(1));
        table("row residual, tile mean (rhs units)", t => t.residual === undefined ? "." : t.residual.toPrecision(4));
        table("row residual / (diagonal ulp(p)), liquid mean", t => t.ulps === undefined ? "." : t.ulps.toFixed(3));
        table("published -rho div/dt, tile mean (rhs units)", t => t.all.toPrecision(4));
        table("published -rho div/dt, cells under the plane (rhs units)", t => t.wet.toPrecision(4));
      }
      // --band-column=x,z,y0,y1:f,f: one column of band rows (stderr): total pressure, the step to the row above in
      // ulp(p), the row's + y forced face and published face (um/s), the face the row's own data gives in double,
      // the cell's published -rho div/dt and its row residual (rhs units).
      const bandColumn = arg("band-column", "").split(":");
      if (bandColumn[0] && (bandColumn[1] ?? "").split(",").map(Number).includes(frame)) {
        const [x, z, y0, y1] = bandColumn[0].split(",").map(Number) as [number, number, number, number];
        const band = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer; rows: GPUBuffer; iterate: GPUBuffer; slots: number; slotMapOffset: number } } }).mixedFrame.band;
        const index = new Uint32Array((await readMixedBuffer(device, band.index)).buffer), rows = await readMixedBuffer(device, band.rows), solve = await readMixedBuffer(device, band.iterate);
        const kindsWords = new Uint32Array(rows.buffer), N = 64 * band.slots, slotBase = band.slotMapOffset / 4;
        const rowOf = (xx: number, y: number, zz: number) => {
          if (xx < 0 || y < 0 || zz < 0 || xx >= nx || y >= ny || zz >= nz) return -1;
          const slot = index[slotBase + (xx >> 2) + tx * ((y >> 2) + ty * (zz >> 2))]!; if (!slot) return -1;
          const l = [xx & 3, y & 3, zz & 3];
          return (slot - 1) * 64 + ((l[0]! + l[1]! + l[2]!) & 1) * 32 + (l[0]! >> 1) + 2 * l[1]! + 8 * l[2]!;
        };
        const halo = (q: number[], axis: number, side: number) => N + (axis === 0 ? side * ny * nz + q[1]! + ny * q[2]! : axis === 1 ? 2 * ny * nz + side * nx * nz + q[0]! + nx * q[2]! : 2 * (ny * nz + nx * nz) + side * nx * ny + q[0]! + nx * q[1]!);
        const ulp = (v: number) => Math.pow(2, Math.floor(Math.log2(Math.max(Math.abs(v), 1e-30))) - 23), scale = -scene.fluid.density_kg_m3 / sampleStep;
        const fieldCount = Math.round(rows.length / N), baseOf = (c: number) => fieldCount === 16 || fieldCount === 22 ? rows[(fieldCount - 1) * N + c]! : 0;
        const face = (xx: number, y: number, zz: number, a: number) => velocity[4 * (xx + nx * (y + ny * zz)) + a]!;
        const lines = [`${arm} frame ${frame} band column x ${x}, z ${z} (row fields ${fieldCount}):\n  y kinds   pressure Pa   step ulps  correction   rhs  forced+y  published+y  from row    div(pub)   residual   div x,y,z parts`];
        for (let y = y1; y >= y0; y--) {
          const c = rowOf(x, y, z); if (c < 0) { lines.push(`${String(y).padStart(3)} not a band row`); continue; }
          const kinds = kindsWords[2 * N + c]!, total = solve[c]! + baseOf(c), above = rowOf(x, y + 1, z);
          let off = 0; const names: string[] = [];
          for (let f = 0; f < 6; f++) {
            const axis = f >> 1, sign = f & 1 ? 1 : -1, kind = (kinds >>> (3 * f)) & 7, q = [x, y, z]; names.push("GBWONC"[kind] ?? "?");
            if (kind === 1) { q[axis]! += sign; off += rows[(3 + f) * N + c]! * solve[rowOf(q[0]!, q[1]!, q[2]!)]!; }
            else if (kind === 2) off += rows[(3 + f) * N + c]! * solve[halo(q, axis, sign > 0 ? 1 : 0)]!;
          }
          const r = rows[c]! + off - rows[N + c]! * solve[c]!;
          const step = above >= 0 ? (solve[above]! + baseOf(above)) - total : NaN;
          const fromRow = above >= 0 ? rows[(9 + 3) * N + c]! - (sampleStep / scene.fluid.density_kg_m3) * step * rows[(3 + 3) * N + c]! * h : NaN;
          const parts = [0, 1, 2].map(a => { const q = [x, y, z]; q[a]! -= 1; return scale * (face(x, y, z, a) - face(q[0]!, q[1]!, q[2]!, a)) / h; });
          lines.push(`${String(y).padStart(3)} ${names.join("")}${kinds & 0x80000000 ? "L" : "a"} ${total.toFixed(5).padStart(12)} ${(step / ulp(total)).toFixed(2).padStart(11)} ${solve[c]!.toPrecision(5).padStart(11)} ${rows[c]!.toPrecision(4).padStart(9)} ${(1e6 * (rows[(9 + 3) * N + c]! + -scene.fluid.gravity_m_s2.y * sampleStep)).toFixed(3).padStart(9)} ${(1e6 * face(x, y, z, 1)).toFixed(4).padStart(11)} ${(1e6 * fromRow).toFixed(4).padStart(10)} ${(parts[0]! + parts[1]! + parts[2]!).toPrecision(4).padStart(10)} ${r.toPrecision(4).padStart(10)}   ${parts.map(v => v.toPrecision(3)).join(" ")}`);
        }
        console.error(lines.join("\n"));
      }
      // the (2r+1)^2 columns about (x,z) at those frames, five rows about the waterline.
      const cellDump = arg("cell-dump", "").split(":"), dumpAt = cellDump[0] ? cellDump[0].split(",").map(Number) : undefined;
      const dumpFrames = (cellDump[1] ?? "").split(",").filter(Boolean).map(Number);
      if ((arg("dump", "off") === "on" && frame === 1) || (dumpAt && dumpFrames.includes(frame))) {
        const cells = [], cx = dumpAt?.[0] ?? sample.maxCell[0]!, cz = dumpAt?.[1] ?? sample.maxCell[2]!, r = dumpAt?.[2] ?? 1;
        for (let z = Math.max(0, cz - r); z <= Math.min(nz - 1, cz + r); z++) for (let y = surfaceY - 2; y <= surfaceY + 2; y++) for (let x = Math.max(0, cx - r); x <= Math.min(nx - 1, cx + r); x++) {
          const i = x + nx * (y + ny * z);
          cells.push({ p: [x, y, z], width: fields.widthAt(x, y, z), open: open[i], volume: volume[i], velocity: [...velocityAt(i)], phi: Array.from({ length: 8 }, (_, k) => fields.vertex(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1))) });
        }
        mkdirSync(resolve("artifacts/pond-rest"), { recursive: true });
        writeFileSync(`artifacts/pond-rest/${arm}-cells${dumpAt ? `-${frame}` : ""}.json`, JSON.stringify(cells, null, 2));
      }
      // --band-dump=x0,x1,y0,y1,z:f,f: the h band's solve at those frames on an
      // x-y window: per cell the pressure over its hydrostatic value (p/depth over
      // the window's median, minus one, in 1e-4) and the row residual over
      // diagonal*p (1e-4); '.' is not a liquid band row.
      const bandDump = arg("band-dump", "").split(":");
      if (bandDump[0] && (bandDump[1] ?? "").split(",").map(Number).includes(frame)) {
        const [x0, x1, y0, y1, z] = bandDump[0].split(",").map(Number) as [number, number, number, number, number];
        const band = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer; rows: GPUBuffer; iterate: GPUBuffer; slots: number; slotMapOffset: number } } }).mixedFrame.band;
        const index = new Uint32Array((await readMixedBuffer(device, band.index)).buffer), rows = await readMixedBuffer(device, band.rows), solve = await readMixedBuffer(device, band.iterate);
        const kindsWords = new Uint32Array(rows.buffer), N = 64 * band.slots, slotBase = band.slotMapOffset / 4;
        const rowOf = (x: number, y: number, zz: number) => {
          if (x < 0 || y < 0 || zz < 0 || x >= nx || y >= ny || zz >= nz) return -1;
          const slot = index[slotBase + (x >> 2) + tx * ((y >> 2) + ty * (zz >> 2))]!; if (!slot) return -1;
          const l = [x & 3, y & 3, zz & 3];
          return (slot - 1) * 64 + ((l[0]! + l[1]! + l[2]!) & 1) * 32 + (l[0]! >> 1) + 2 * l[1]! + 8 * l[2]!;
        };
        const halo = (q: number[], axis: number, side: number) => N + (axis === 0 ? side * ny * nz + q[1]! + ny * q[2]! : axis === 1 ? 2 * ny * nz + side * nx * nz + q[0]! + nx * q[2]! : 2 * (ny * nz + nx * nz) + side * nx * ny + q[0]! + nx * q[1]!);
        const cells: { x: number; y: number; ratio?: number; residual?: number; kinds?: string }[] = [];
        for (let y = y1; y >= y0; y--) for (let x = x0; x <= x1; x++) {
          const c = rowOf(x, y, z); if (c < 0) { cells.push({ x, y }); continue; }
          const kinds = kindsWords[2 * N + c]!; if (!(kinds & 0x80000000)) { cells.push({ x, y, kinds: "air" }); continue; }
          let off = 0; const names: string[] = [];
          for (let f = 0; f < 6; f++) {
            const axis = f >> 1, sign = f & 1 ? 1 : -1, kind = (kinds >>> (3 * f)) & 7, q = [x, y, z]; names.push("GBWONC"[kind] ?? "?");
            if (kind === 1) { q[axis]! += sign; off += rows[(3 + f) * N + c]! * solve[rowOf(q[0]!, q[1]!, q[2]!)]!; }
            else if (kind === 2) off += rows[(3 + f) * N + c]! * solve[halo(q, axis, sign > 0 ? 1 : 0)]!;
          }
          const depth = level - (y + 0.5);
          cells.push({ x, y, ratio: solve[c]! / depth, residual: (rows[c]! + off - rows[N + c]! * solve[c]!) / (rows[N + c]! * solve[c]!), kinds: names.join("") });
        }
        const ratios = cells.filter(c => c.ratio !== undefined && Number.isFinite(c.ratio)).map(c => c.ratio!).sort((a, b) => a - b), median = ratios[ratios.length >> 1] ?? 1;
        const table = (value: (c: typeof cells[number]) => string) => Array.from({ length: y1 - y0 + 1 }, (_, r) => `${String(y1 - r).padStart(3)} ` + cells.slice(r * (x1 - x0 + 1), (r + 1) * (x1 - x0 + 1)).map(value).join(" ")).join("\n");
        console.error(`${arm} frame ${frame} band p/hydrostatic - 1 (1e-4), x ${x0}..${x1}, z ${z}:\n${table(c => c.ratio === undefined ? "     ." : (1e4 * (c.ratio / median - 1)).toFixed(1).padStart(6))}`);
        console.error(`${arm} frame ${frame} band row residual / (diagonal p) (1e-4):\n${table(c => c.residual === undefined ? "     ." : (1e4 * c.residual).toFixed(1).padStart(6))}`);
        {
          // The all-4h root the band starts from, on the window's tiles: pressure over hydrostatic at the tile centre, same median.
          const root = await readMixedBuffer(device, (frameFields as unknown as { presentation: { pressure: GPUBufferBinding } }).presentation.pressure.buffer as GPUBuffer);
          const lines: string[] = [];
          for (let Y = y1 >> 2; Y >= y0 >> 2; Y--) lines.push(`${String(Y).padStart(3)} ` + Array.from({ length: (x1 >> 2) - (x0 >> 2) + 1 }, (_, i) => {
            const X = (x0 >> 2) + i, depth = level - (4 * Y + 2), value = root[X + tx * (Y + ty * (z >> 2))]!;
            return value === 0 ? "      ." : (1e4 * (value / depth / median - 1)).toFixed(1).padStart(7);
          }).join(" "));
          console.error(`${arm} frame ${frame} root p/hydrostatic - 1 (1e-4), tiles x ${x0 >> 2}..${x1 >> 2}:\n${lines.join("\n")}`);
        }
        {
          // The rows against the exact hydrostatic field (the root's scale): rhs - A p_h over the diagonal, in 1e-6 cells of head.
          const root = await readMixedBuffer(device, (frameFields as unknown as { presentation: { pressure: GPUBufferBinding } }).presentation.pressure.buffer as GPUBuffer);
          const exact: number[] = [];
          for (let Y = y0 >> 2; Y <= y1 >> 2; Y++) for (let X = x0 >> 2; X <= x1 >> 2; X++) { const r = root[X + tx * (Y + ty * (z >> 2))]! / (level - (4 * Y + 2)); if (Math.abs(r / median - 1) < 0.01) exact.push(r); }
          exact.sort((a, b) => a - b); const scale = exact[exact.length >> 1] ?? median;
          const hydro = (x: number, y: number, zz: number) => { const c = rowOf(x, y, zz); return c >= 0 && kindsWords[2 * N + c]! & 0x80000000 ? scale * (level - (y + 0.5)) : 0; };
          const lines: string[] = [];
          for (let y = y1; y >= y0; y--) lines.push(`${String(y).padStart(3)} ` + Array.from({ length: x1 - x0 + 1 }, (_, i) => {
            const x = x0 + i, c = rowOf(x, y, z); if (c < 0 || !(kindsWords[2 * N + c]! & 0x80000000)) return "     .";
            let off = 0; const kinds = kindsWords[2 * N + c]!;
            for (let f = 0; f < 6; f++) {
              const axis = f >> 1, sign = f & 1 ? 1 : -1, kind = (kinds >>> (3 * f)) & 7, q = [x, y, z];
              if (kind === 1) { q[axis]! += sign; off += rows[(3 + f) * N + c]! * hydro(q[0]!, q[1]!, q[2]!); }
              else if (kind === 2) off += rows[(3 + f) * N + c]! * hydro(x, y, z);
            }
            return (1e6 * (rows[c]! + off - rows[N + c]! * hydro(x, y, z)) / (rows[N + c]! * scale)).toFixed(1).padStart(6);
          }).join(" "));
          console.error(`${arm} frame ${frame} hydrostatic row defect / diagonal (1e-6 cells of head):\n${lines.join("\n")}`);
          // The band's largest row residuals after the solve, anywhere: residual over diagonal and the pressure error, in 1e-6 cells of head.
          const worst: { at: number[]; r: number; e: number; kinds: string }[] = [], count = Math.min(index[0]!, band.slots);
          let net = 0, gross = 0, dirichlet = 0, residualSum = 0;
          for (let slot = 0; slot < count; slot++) {
            const t = index[26 + slot]!, o = [4 * (t % tx), 4 * (Math.floor(t / tx) % ty), 4 * Math.floor(t / (tx * ty))];
            for (let l = 0; l < 64; l++) {
              const x = o[0]! + (l & 3), y = o[1]! + ((l >> 2) & 3), zz = o[2]! + (l >> 4), c = rowOf(x, y, zz), kinds = kindsWords[2 * N + c]!;
              if (!(kinds & 0x80000000) || !(rows[N + c]! > 0)) continue;
              let off = 0; const names: string[] = [];
              for (let f = 0; f < 6; f++) {
                const axis = f >> 1, sign = f & 1 ? 1 : -1, kind = (kinds >>> (3 * f)) & 7, q = [x, y, zz]; names.push("GBWONC"[kind] ?? "?");
                if (kind === 1) { q[axis]! += sign; off += rows[(3 + f) * N + c]! * solve[rowOf(q[0]!, q[1]!, q[2]!)]!; }
                else if (kind === 2) off += rows[(3 + f) * N + c]! * solve[halo(q, axis, sign > 0 ? 1 : 0)]!;
              }
              net += rows[c]!; gross += Math.abs(rows[c]!); dirichlet += names.filter(k => k === "G" || k === "O").length; residualSum += rows[c]! + off - rows[N + c]! * solve[c]!;
              worst.push({ at: [x, y, zz], r: (rows[c]! + off - rows[N + c]! * solve[c]!) / (rows[N + c]! * scale), e: solve[c]! / scale - (level - (y + 0.5)), kinds: names.join("") + (kinds & 0x40000000 ? " inside" : "") + (kinds & 0x20000000 ? " shut" : "") });
            }
          }
          const rank = (key: (w: typeof worst[number]) => number) => [...worst].sort((a, b) => key(b) - key(a)).slice(0, 10).map(w => `  ${w.at.join(",")} r ${(1e6 * w.r).toFixed(1)} e ${(1e6 * w.e).toFixed(1)} ${w.kinds}`).join("\n");
          // A band with no Dirichlet face is singular: its rows are solvable only when they sum to zero.
          console.error(`${arm} frame ${frame} band rhs sum ${net.toExponential(3)} of |rhs| ${gross.toExponential(3)}, residual sum ${residualSum.toExponential(3)}, Dirichlet faces ${dirichlet}, unit ${(6 * scale / (0.0125 * 0.0125)).toExponential(3)}`);
          console.error(`${arm} frame ${frame} band rows ${worst.length}; largest residual/diagonal (1e-6 cells):\n${rank(w => Math.abs(w.r))}\nlargest pressure error:\n${rank(w => Math.abs(w.e))}`);
        }
        console.error(`${arm} frame ${frame} band face kinds (-x+x-y+y-z+z; G ghost B band W wall O open N neumann C closed):\n${table(c => (c.kinds ?? ".").padStart(6))}`);
      }
      if (tileDumps.includes(frame)) {
        // The represented state of the three tile rows around the waterline:
        // 4-aligned vertex phi, owner V (h cells averaged over the tile), the
        // tile's + face anchors, pressure phi and correction.
        const phi = frame ? await readMixedBuffer(device, frameFields.presentation.phi.buffer as GPUBuffer) : undefined, rateField = frame ? await read(frameFields.fields.correction) : undefined;
        // The extension of this state: the field the next advance's surface trace samples.
        let extended: Float32Array | undefined;
        if (frame) {
          const mixed = frameFields as unknown as { encodeExtension(e: GPUCommandEncoder): void; fields: { velocityScratch: GPUTexture } };
          const encoder = device.createCommandEncoder(); mixed.encodeExtension(encoder); device.queue.submit([encoder.finish()]);
          extended = await read(mixed.fields.velocityScratch);
        }
        const extendedAt = (i: number) => {
          const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / nx / ny), w = fields.widthAt(x, y, z), o = [x - x % w, y - y % w, z - z % w];
          return [0, 1, 2].map(a => extended ? extended[4 * (o[0]! + (a === 0 ? w - 1 : 0) + nx * (o[1]! + (a === 1 ? w - 1 : 0) + ny * (o[2]! + (a === 2 ? w - 1 : 0)))) + a]! : 0);
        };
        const rows = [];
        // --tile-rows=lo,hi: those tile rows instead (a sloped floor's wet rows).
        const tileRows = arg("tile-rows", `${surfaceTile - 2},${surfaceTile + 1}`).split(",").map(Number) as [number, number];
        for (let ty0 = Math.max(0, tileRows[0]); ty0 <= Math.min(ty - 1, tileRows[1]); ty0++) {
          const row = { tileY: ty0, vertex: [] as number[], vertexTop: [] as number[], volume: [] as number[], open: [] as number[], velocity: [] as number[][], extended: [] as number[][], pressurePhi: [] as number[], correction: [] as number[], width: [] as number[] };
          for (let z = 0; z <= tz; z++) for (let x = 0; x <= tx; x++) { row.vertex.push(fields.vertex(4 * x, 4 * ty0, 4 * z)); row.vertexTop.push(fields.vertex(4 * x, 4 * ty0 + 4, 4 * z)); }
          for (let z = 0; z < tz; z++) for (let x = 0; x < tx; x++) {
            const i = 4 * x + nx * (4 * ty0 + ny * 4 * z), t = x + tx * (ty0 + ty * z);
            let v = 0; for (let k = 0; k < 64; k++) v += volume[i + (k & 3) + nx * (((k >> 2) & 3) + ny * (k >> 4))]! / 64;
            row.volume.push(v); row.velocity.push(velocityAt(i)); row.extended.push(extendedAt(i)); row.width.push(fields.widthAt(4 * x, 4 * ty0, 4 * z));
            row.pressurePhi.push(phi ? phi[t]! : 0); row.correction.push(rateField ? rateField[i]! : 0); row.open.push(tileOpen[t]!);
          }
          rows.push(row);
        }
        mkdirSync(resolve("artifacts/pond-rest"), { recursive: true });
        writeFileSync(`artifacts/pond-rest/${arm}-tiles-${frame}.json`, JSON.stringify({ arm, frame, tx, tz, level, h, rows }));
      }
      samples.push(sample); console.log(JSON.stringify({ arm, ...sample }));
    }
    const fineTiles = tiles.reduce((n, word) => n + (word & 0x80000000 ? 1 : 0), 0);
    const result: PondRestResult = { arm, method: method.id, dimensions: [nx, ny, nz], level, h, depth_m: depthCells * h, time_s: solver.info.simulatedTime_s ?? frames * sampleStep, profile, detail, basin, fineTiles, tiles: tiles.length, values, hose: scene.fluid.inflow.enabled, jacobiWeight: weight, dampingReplacements, noRecovery, sigma: scene.fluid.surfaceTension_N_m, gravity: scene.fluid.gravity_m_s2, vesselFinal, samples, pressureDiagnostic, stepTimes_ms };
    mkdirSync(resolve(out, ".."), { recursive: true });
    writeFileSync(out, JSON.stringify(result, null, 2));
    return result;
  } finally { compiler.createShaderModule = createModule; solver?.destroy(); }
}
