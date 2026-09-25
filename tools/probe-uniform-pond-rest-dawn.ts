import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { WebGPUUniformPressureMultigrid } from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import type { UniformTexturePages } from "../lib/methods/uniform/uniform-texture-pages";

const arg = (key: string, fallback: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const arm = arg("arm", "baseline");
const frames = Number(arg("frames", "30"));
const overrides = JSON.parse(arg("values", "{}"));
const out = arg("out", `artifacts/pond-rest/${arm}.json`);
async function read(device: GPUDevice, texture: GPUTexture) {
  const c = texture.format === "rgba32float" ? 4 : 1;
  const width = texture.width * c, row = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: row * texture.height * texture.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const e = device.createCommandEncoder();
    e.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: texture.height }, [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([e.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(buffer.getMappedRange()), result = new Float32Array(width * texture.height * texture.depthOrArrayLayers);
    for (let z = 0; z < texture.depthOrArrayLayers; z++) for (let y = 0; y < texture.height; y++) result.set(src.subarray((z * texture.height + y) * row / 4, (z * texture.height + y) * row / 4 + width), (z * texture.height + y) * width);
    return result;
  } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
}

await acquireWebGPUExclusiveLock("dawn-probe", `uniform pond rest ${arm}`);
let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
  const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const definition = getSceneDefinition("hero-garden-hose-x10");
  const scene = sceneDocument(definition);
  scene.fluid.inflow = { ...scene.fluid.inflow!, enabled: arg("hose", "off")==="on" };
  const cell = scene.voxelDomain.finestCellSize_m;
  scene.container.fillFraction += Number(arg("waterline-shift-cells", "0")) * cell / scene.container.height_m;
  if (arg("basin", "pond") === "flat") {
    // Same domain, resolution and waterline; floor chosen for the pond's
    // median 5.6-cell depth. Footprint differs, so this isolates a group of
    // geometric effects, not curvature alone.
    scene.terrain = { baseHeight_m: 15 * cell, features: [] };
    scene.solidVoxels = []; scene.rigidBodies = []; scene.scenery = undefined;
  }
  if (arg("sigma", "scene") !== "scene") scene.fluid.surfaceTension_N_m = Number(arg("sigma", "0"));
  if (arg("gravity", "scene") !== "scene") scene.fluid.gravity_m_s2 = { x: 0, y: Number(arg("gravity", "0")), z: 0 };
  // Keep the historical full-solve control, but also exercise the authored UI
  // profile so a passing reference solve cannot hide an overly loose default.
  const pressureMode = arg("pressure", "full");
  assert.ok(pressureMode === "full" || pressureMode === "scene");
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", {
    ...definition.methodProfile?.overrides,
    ...(pressureMode === "full" ? { pressureResidualTolerance: 0 } : {}), ...overrides,
  });
  const weightOverride=arg("jacobi-weight", "production");
  const weight = weightOverride==="production" ? 0.6666667 : Number(weightOverride);
  let dampingReplacements=0;
  if (weightOverride !== "production") {
    assert.ok(weight > 0 && weight <= 1);
    const compiler=gpuCompilationManagerFor(device), create=compiler.createShaderModule.bind(compiler);
    compiler.createShaderModule = descriptor => create({...descriptor,code:descriptor.code
      .replace(/const MG_JACOBI_WEIGHT:f32=[0-9.]+;/g,() => {dampingReplacements++;return `const MG_JACOBI_WEIGHT:f32=${weight.toFixed(8)};`;})});
  }
  solver = await uniformVolumeMethod.createSolverAsync!(device, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  if(weightOverride!=="production")assert.ok(dampingReplacements>0,"experimental damping must patch the compiled shader");
  const { nx, ny, nz } = solver.info, h = scene.container.height_m / ny;
  const open = await read(device, solver.denseLevelSetVolumeSource!.openFraction);
  const level = scene.container.height_m * scene.container.fillFraction / h;
  const surfaceY = Math.floor(level);
  const columns: { x: number; z: number; interior: boolean }[] = [];
  for (let z = 3; z < nz - 3; z++) for (let x = 3; x < nx - 3; x++) {
    if (open[x + nx * (surfaceY - 1 + ny * z)]! < 0.999) continue;
    let interior = true;
    for (let dz = -2; dz <= 2; dz++) for (let dx = -2; dx <= 2; dx++) if (open[x + dx + nx * (surfaceY - 2 + ny * (z + dz))]! < 0.999) interior = false;
    columns.push({ x, z, interior });
  }
  assert.ok(columns.some(c => c.interior));
  const mg = (solver as unknown as { pressureMultigrid: WebGPUUniformPressureMultigrid }).pressureMultigrid;
  const snapshots = new Map<GPUTexture,GPUTexture>();
  const noRecovery=arg("recovery", "on")==="off";
  if (arg("pressure-dump", "off") === "on" || noRecovery) {
    const fields = (mg as unknown as {scratchFields: UniformTexturePages}).scratchFields;
    if(arg("pressure-dump", "off") === "on")for (const l of mg.levels) for (const t of [l.volume[0],l.coefficients,l.pressure[0],l.rhs[0],l.minimum[0]]) snapshots.set(t,fields.snapshotTexture(t));
    const encode = mg.encode.bind(mg);
    mg.encode = (...args: Parameters<typeof mg.encode>) => {
      if(noRecovery){
        assert.equal(values.pressureResidualTolerance,0,"no-recovery control requires the fixed schedule");
        const plan=mg as unknown as {finishStart:number;finalStart:number;plan:unknown[]};
        encode(args[0],args[1],args[2],args[3],args[4],{start:0,end:plan.finishStart,initialize:true,publish:false});
        encode(args[0],args[1],args[2],args[3],args[4],{start:plan.finalStart,end:plan.plan.length,initialize:false,publish:true});
      }else encode(...args);
      // Scratch is reused by later fluid stages. Capture while pressure still
      // owns it, rather than reading the placeholder texture after the frame.
      for (const t of snapshots.keys()) fields.encodeSnapshot(args[0],t);
    };
  }
  const readPressure = (t:GPUTexture) => read(device!, snapshots.get(t) ?? t);
  const samples: unknown[] = [];
  const stepTimes_ms:number[]=[];
  let pressureDiagnostic: unknown;
  for (let frame = 0; frame <= frames; frame++) {
    if (frame) { const started=performance.now();assert.ok(solver.advanceTo(frame / 30, [])); await solver.awaitFrameCompletion();await device.queue.onSubmittedWorkDone();if(frame>5)stepTimes_ms.push(performance.now()-started); }
    const stats = await solver.readStats();
    if (frame === 1 && arg("pressure-dump", "off") === "on") {
      // Diagnostic-only access: inspect the actual solver hierarchy, without
      // adding a production readback or changing any pressure dispatch.
      const hierarchy = [];
      for (const l of mg.levels) {
        const topology = await readPressure(l.volume[0]), coefficients = await readPressure(l.coefficients);
        let liquid = 0, openLiquid = 0, mixedLiquid = 0, solidLiquid = 0;
        for (let i = 0; i < coefficients.length / 4; i++) if ((coefficients[4*i+3]! & 1) !== 0) {
          liquid++;
          const [wx,wy] = l.dimensions, x=i%wx-1, y=Math.floor(i/wx)%wy-1, z=Math.floor(i/wx/wy)-1;
          // Finest topology aliases cycle scratch after coefficient baking.
          const capacity = l===mg.levels[0] ? (x>=0&&x<nx&&y>=0&&y<ny&&z>=0&&z<nz ? open[x+nx*(y+ny*z)]! : 0) : topology[4*i]!;
          if (capacity <= 1e-5) solidLiquid++; else { openLiquid++; if (capacity < 0.99999) mixedLiquid++; }
        }
        hierarchy.push({ dimensions: l.dimensions.map(v => v-2), liquid, openLiquid, mixedLiquid, solidLiquid });
      }
      const l = mg.levels[0]!, [wx,wy,wz] = l.dimensions;
      const p = await readPressure(l.pressure[0]), rhs = await readPressure(l.rhs[0]), minimum = await readPressure(l.minimum[0]);
      const c = await readPressure(l.coefficients);
      const worst: { cell: number[]; residual: number; open: number; pressure: number; rhs: number; diagonal: number }[] = [];
      const liquid = (i: number) => i >= 0 && i < p.length && (c[4*i+3]! & 1) !== 0;
      for (let z=1; z<wz-1; z++) for (let y=1; y<wy-1; y++) for (let x=1; x<wx-1; x++) {
        const i=x+wx*(y+wy*z); if (!liquid(i)) continue;
        let ap=0, diagonal=0;
        for (const [axis,stride] of [1,wx,wx*wy].entries()) for (const sign of [-1,1]) {
          const j=i+sign*stride, a=c[4*(sign>0?i:j)+axis]!;
          diagonal+=a; ap+=a*(p[i]!-(liquid(j)?p[j]!:0));
        }
        const r=rhs[i]!-ap, gap=Math.max(0,p[i]!-minimum[i]!);
        const projected=Math.max(r<0 && -r>=gap*diagonal ? gap*diagonal : Math.abs(r), Math.max(0,minimum[i]!-p[i]!)*diagonal);
        const residual=projected/30/scene.fluid.density_kg_m3;
        if (worst.length<20 || residual>worst.at(-1)!.residual) {
          worst.push({cell:[x-1,y-1,z-1],residual,open:open[x-1+nx*(y-1+ny*(z-1))]!,pressure:p[i]!,rhs:rhs[i]!,diagonal});
          worst.sort((a,b)=>b.residual-a.residual); worst.length=Math.min(20,worst.length);
        }
      }
      pressureDiagnostic={hierarchy,worst,cpuResidualNote:"Double precision reconstruction from GPU fields; compare with GPU f32 residual, not bitwise.",gpuResidual:stats.uniformCM11aFineResidualInfinity};
      assert.ok(worst.length>0,"capture populated pressure rows before scratch reuse");
      assert.ok(Math.abs(worst[0]!.residual-stats.uniformCM11aFineResidualInfinity!)<1e-4,"CPU reconstruction agrees with the GPU residual");
      console.log(JSON.stringify({arm,pressureDiagnostic}));
    }
    if (frame > 3 && frame % 10 && frame !== frames) continue;
    const phi = await read(device, solver.vertexPhiTexture!), velocity = await read(device, solver.velocityTexture), volume = await read(device, solver.volumeTexture);
    const regions = { all: [] as number[], interior: [] as number[] };
    let missing = 0;
    for (const c of columns) {
      let height: number | undefined;
      for (let y = Math.max(0, surfaceY - 3); y < Math.min(ny, surfaceY + 4); y++) {
        const a = phi[c.x + (nx + 1) * (y + (ny + 1) * c.z)]!, b = phi[c.x + (nx + 1) * (y + 1 + (ny + 1) * c.z)]!;
        if (a <= 0 && b > 0) height = y - a / (b - a);
      }
      if (height === undefined) { missing++; continue; }
      regions.all.push((height - level) * h * 1000);
      if (c.interior) regions.interior.push((height - level) * h * 1000);
    }
    const surface = Object.fromEntries(Object.entries(regions).map(([key, a]) => [key, { count: a.length, mean_mm: a.reduce((s, v) => s + v, 0) / a.length, rms_mm: Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length), max_mm: Math.max(...a.map(Math.abs)), range_mm: Math.max(...a) - Math.min(...a) }]));
    let maxSpeed = 0, excess = 0, sum = 0, maxCell = 0;
    for (let i = 0; i < volume.length; i++) { const v = volume[i]!; assert.ok(Number.isFinite(v)); sum += v; excess += Math.max(0, v - open[i]!); if (v > 1e-5) { const speed = Math.hypot(velocity[i * 4]!, velocity[i * 4 + 1]!, velocity[i * 4 + 2]!); if (speed > maxSpeed) { maxSpeed = speed; maxCell = i; } } }
    const sample = { frame, surface, missing, maxSpeed, maxCell: [maxCell % nx, Math.floor(maxCell / nx) % ny, Math.floor(maxCell / nx / ny)], sum, excess, residual: stats.uniformCM11aFineResidualInfinity, acceptedResidual: stats.uniformPressureAcceptedResidual, recovery: stats.uniformPressureRecoverySweeps, full: stats.uniformCM11aFullCyclesExecuted, vcycles: stats.uniformCM11aVCyclesExecuted };
    if (arg("dump", "off") === "on" && frame === 1) {
      const cells = [];
      for (let z = sample.maxCell[2]! - 1; z <= sample.maxCell[2]! + 1; z++) for (let y = surfaceY - 2; y <= surfaceY + 2; y++) for (let x = sample.maxCell[0]! - 1; x <= sample.maxCell[0]! + 1; x++) {
        const i = x + nx * (y + ny * z);
        cells.push({ p: [x,y,z], open: open[i], volume: volume[i], velocity: [...velocity.subarray(4*i,4*i+4)], phi: Array.from({length:8}, (_, k) => phi[x+(k&1)+(nx+1)*(y+((k>>1)&1)+(ny+1)*(z+((k>>2)&1)))]) });
      }
      writeFileSync(`artifacts/pond-rest/${arm}-cells.json`, JSON.stringify(cells,null,2));
    }
    samples.push(sample); console.log(JSON.stringify({ arm, ...sample }));
  }
  assert.deepEqual(errors, []);
  mkdirSync(resolve(out, ".."), { recursive: true });
  writeFileSync(out, JSON.stringify({ arm, dimensions: [nx, ny, nz], level, h, values, hose:scene.fluid.inflow.enabled, jacobiWeight: weight, dampingReplacements, noRecovery, sigma: scene.fluid.surfaceTension_N_m, gravity: scene.fluid.gravity_m_s2, samples, pressureDiagnostic, stepTimes_ms }, null, 2));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
