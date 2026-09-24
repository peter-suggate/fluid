/** Matched Uniform Geometric symmetric-expansion A/B, with stage and D4 diagnostics. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT = "1";
const frames = Number(process.env.FLUID_AIRBORNE_AB_FRAMES ?? 45);
const modes = process.env.FLUID_AIRBORNE_AB_MODE === "off" ? ["off"] as const
  : process.env.FLUID_AIRBORNE_AB_MODE === "on" ? ["on"] as const : ["off", "on"] as const;
const densePressureAudit = process.env.FLUID_AIRBORNE_SYM_DENSE_PRESSURE === "1";
const valueOverrides = JSON.parse(process.env.FLUID_AIRBORNE_SYM_VALUES ?? "{}") as Record<string, string | number>;
if (densePressureAudit) process.env.FLUID_UNIFORM_PRESSURE_SETUP_AUDIT = "1";
const checkpoints = new Set(Array.from({ length: frames + 1 }, (_, frame) => frame));
const dt = 1 / 30;
type Snapshot = Record<string, Float32Array>;
async function read(device: GPUDevice, texture: GPUTexture, components: number,
  dims: readonly [number, number, number]): Promise<Float32Array> {
  const [nx, ny, nz] = dims;
  const row = Math.ceil(nx * components * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: row * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: ny },
      { width: nx, height: ny, depthOrArrayLayers: nz });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(buffer.getMappedRange());
    const values = new Float32Array(nx * ny * nz * components);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) {
      const source = (z * ny + y) * row / 4;
      values.set(raw.subarray(source, source + nx * components),
        (z * ny + y) * nx * components);
    }
    return values;
  } finally { buffer.unmap(); buffer.destroy(); }
}
async function readBuffer(device: GPUDevice, source: GPUBuffer): Promise<Float32Array> {
  const staging=device.createBuffer({size:source.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,0,staging,0,source.size);
    device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
    return new Float32Array(staging.getMappedRange()).slice();
  } finally {staging.unmap();staging.destroy();}
}
function negativeBoundaryKinetic(volume: Float32Array, boundary: Float32Array | undefined,
  dims: readonly [number,number,number]) {
  if(!boundary)return 0;
  const [nx,ny,nz]=dims;let energy=0;
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)energy+=.25*volume[nx*(y+ny*z)]!*boundary[y+ny*z]!**2;
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)energy+=.25*volume[x+nx*ny*z]!*boundary[ny*nz+x+nx*z]!**2;
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)energy+=.25*volume[x+nx*y]!*boundary[ny*nz+nx*nz+x+nx*y]!**2;
  return energy;
}
function difference(a: Float32Array, b: Float32Array, components: number) {
  let sum = 0, max = 0, rms = 0, count = 0;
  for (let i = 0; i < a.length; i++) {
    if (components === 4 && i % 4 === 3) continue;
    const d = Math.abs(a[i]! - b[i]!);
    sum += d; rms += d * d; max = Math.max(max, d); count++;
  }
  return { mean: sum / count, rms: Math.sqrt(rms / count), max };
}
function range(field: Float32Array) {
  let min = Infinity, max = -Infinity, nonzero = 0;
  for (const value of field) { min = Math.min(min, value); max = Math.max(max, value); if (value !== 0) nonzero++; }
  return { min, max, nonzero };
}
function largestDifferences(a: Float32Array, b: Float32Array,
  dims: readonly [number, number, number], components: number) {
  const [nx, ny] = dims;
  const ranked: { delta: number; on: number; off: number; xyz: number[]; component: number }[] = [];
  for (let i = 0; i < a.length; i++) {
    if (components === 4 && i % 4 === 3) continue;
    const delta = Math.abs(a[i]! - b[i]!);
    if (delta === 0 || (ranked.length === 5 && delta <= ranked[4]!.delta)) continue;
    const cell = Math.floor(i / components);
    ranked.push({ delta, on: a[i]!, off: b[i]!,
      xyz: [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))],
      component: i % components });
    ranked.sort((left, right) => right.delta - left.delta);
    if (ranked.length > 5) ranked.pop();
  }
  return ranked;
}
function metrics(snapshot: Snapshot, dims: readonly [number, number, number], cellHeight: number, gravity: number) {
  const [nx, ny, nz] = dims;
  const volume = snapshot.volume!, velocity = snapshot.velocity!;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  let mass = 0, wet = 0, kinetic = 0, velocityHighFrequency = 0;
  let faceKinetic = 0, potential = 0, peakCellVolume = 0, excessVolume = 0;
  let d4Mean = 0, d4Max = 0, d4Count = 0;
  const height = new Int16Array(nx * nz).fill(-1);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = index(x, y, z), v = volume[i]!;
    mass += v; peakCellVolume = Math.max(peakCellVolume, v); excessVolume += Math.max(0,v-1);
    if (v >= 0.5) { wet++; height[x + nx * z] = y; }
    const vx = velocity[4 * i]!, vy = velocity[4 * i + 1]!, vz = velocity[4 * i + 2]!;
    kinetic += v * (vx * vx + vy * vy + vz * vz) / 2;
    // MAC velocities live between cells: use both incident volumes for a
    // reflection-invariant energy proxy. Keep the old proxy for comparisons.
    faceKinetic += ((v + (x+1<nx ? volume[i+1]! : 0))*vx*vx
      + (v + (y+1<ny ? volume[i+nx]! : 0))*vy*vy
      + (v + (z+1<nz ? volume[i+nx*ny]! : 0))*vz*vz) / 4;
    potential += v * gravity * cellHeight * (y + 0.5);
    if (x > 0 && x + 1 < nx && z > 0 && z + 1 < nz && v > 0.1) {
      const lapX = vx - (velocity[4 * index(x - 1, y, z)]! + velocity[4 * index(x + 1, y, z)]!) / 2;
      const lapZ = vz - (velocity[4 * index(x, y, z - 1) + 2]! + velocity[4 * index(x, y, z + 1) + 2]!) / 2;
      velocityHighFrequency += v * (lapX * lapX + lapZ * lapZ);
    }
    for (const j of [index(nx - 1 - x, y, z), index(x, y, nz - 1 - z), ...(nx===nz?[index(z, y, x)]:[])]) {
      const d = Math.abs(v - volume[j]!);
      d4Mean += d; d4Max = Math.max(d4Max, d); d4Count++;
    }
  }
  let heightD4 = 0, heightRoughness = 0, heightCount = 0;
  for (let z = 1; z + 1 < nz; z++) for (let x = 1; x + 1 < nx; x++) {
    const h = height[x + nx * z]!;
    heightD4 += Math.abs(h - height[(nx - 1 - x) + nx * z]!);
    heightD4 += Math.abs(h - height[x + nx * (nz - 1 - z)]!);
    if(nx===nz)heightD4 += Math.abs(h - height[z + nx * x]!);
    const neighbors = [height[x - 1 + nx * z]!, height[x + 1 + nx * z]!,
      height[x + nx * (z - 1)]!, height[x + nx * (z + 1)]!];
    heightRoughness += Math.abs(h - neighbors.reduce((a, b) => a + b, 0) / 4);
    heightCount++;
  }
  faceKinetic+=negativeBoundaryKinetic(volume,snapshot.negativeVelocity,dims);
  return { mass, wet, kinetic, faceKinetic, potential, mechanical: faceKinetic + potential, peakCellVolume, excessVolume, velocityHighFrequency, d4Mean: d4Mean / d4Count,
    d4Max, heightD4: heightD4 / ((nx===nz?3:2) * heightCount), heightRoughness: heightRoughness / heightCount };
}
function energyStages(previous: Snapshot, current: Snapshot, dims: readonly [number,number,number]) {
  const [nx,ny,nz]=dims;
  const stages={stored:0,extended:0,transported:0,forced:0,projected:0};
  let projectionPositiveWork=0,projectionNegativeWork=0;
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const i=x+nx*(y+ny*z);
    for(const [axis,q,valid] of [[0,i+1,x+1<nx],[1,i+nx,y+1<ny],[2,i+nx*ny,z+1<nz]] as const){
      const oldMass=.5*(previous.volume![i]!+(valid?previous.volume![q]!:0));
      const newMass=.5*(current.volume![i]!+(valid?current.volume![q]!:0));
      const k=4*i+axis;
      stages.stored+=.5*oldMass*current.preVelocity![k]!**2;
      stages.extended+=.5*oldMass*current.extrapolated![k]!**2;
      stages.transported+=.5*newMass*current.extrapolated![k]!**2;
      const forced=.5*newMass*current.advection![k]!**2;
      const projected=.5*newMass*current.projection![k]!**2;
      stages.forced+=forced;stages.projected+=projected;
      projectionPositiveWork+=Math.max(0,projected-forced);
      projectionNegativeWork+=Math.min(0,projected-forced);
    }
  }
  const boundaryWork=(i:number,j:number)=>{
    const work=.25*current.volume![i]!*((current.negativeVelocity?.[j]??0)**2-(current.negativeAdvection?.[j]??0)**2);
    projectionPositiveWork+=Math.max(0,work);projectionNegativeWork+=Math.min(0,work);
  };
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)boundaryWork(nx*(y+ny*z),y+ny*z);
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++)boundaryWork(x+nx*ny*z,ny*nz+x+nx*z);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)boundaryWork(x+nx*y,ny*nz+nx*nz+x+nx*y);
  stages.stored+=negativeBoundaryKinetic(previous.volume!,current.negativePreVelocity,dims);
  stages.extended+=negativeBoundaryKinetic(previous.volume!,current.negativeExtrapolated,dims);
  stages.transported+=negativeBoundaryKinetic(current.volume!,current.negativeExtrapolated,dims);
  stages.forced+=negativeBoundaryKinetic(current.volume!,current.negativeAdvection,dims);
  stages.projected+=negativeBoundaryKinetic(current.volume!,current.negativeVelocity,dims);
  return {...stages,projectionPositiveWork,projectionNegativeWork,
    extensionChange:stages.extended-stages.stored,
    volumeChange:stages.transported-stages.extended,
    advectionAndForcesChange:stages.forced-stages.transported,
    projectionChange:stages.projected-stages.forced};
}
/** This probe's box has no embedded solids; the two-cell wall clearance is exact. */
function airborneConnectivity(snapshot: Snapshot, dims: readonly [number, number, number], h: number) {
  const [nx, ny, nz] = dims, volume = snapshot.volume!, phi = snapshot.phi!;
  const centre = new Float64Array(volume.length), eligible = new Uint8Array(volume.length);
  let candidates = 0, candidateMass = 0, connectedToPressure = 0;
  for (let z=0;z<nz;z++) for (let y=0;y<ny;y++) for (let x=0;x<nx;x++) {
    const i=x+nx*(y+ny*z);
    for (let k=0;k<8;k++) centre[i]! += phi[x+(k&1)+(nx+1)*(y+((k>>1)&1)+(ny+1)*(z+(k>>2)))]!/8;
    if (volume[i]! > .05 && centre[i]! > 1.5*h && x>=2 && x<nx-2 && y>=2 && y<ny-2 && z>=2 && z<nz-2) {
      eligible[i]=1; candidates++; candidateMass+=volume[i]!;
    }
  }
  const seen = new Uint8Array(volume.length);
  for (let seed=0;seed<volume.length;seed++) {
    if (seen[seed] || volume[seed]! <= .05) continue;
    const pending=[seed]; seen[seed]=1; let pressure=false, airborne=0;
    while (pending.length) {
      const i=pending.pop()!, x=i%nx, y=Math.floor(i/nx)%ny, z=Math.floor(i/(nx*ny));
      pressure ||= centre[i]! < 0; airborne += eligible[i]!;
      for (const [j, valid] of [[i-1,x>0],[i+1,x+1<nx],[i-nx,y>0],[i+nx,y+1<ny],
        [i-nx*ny,z>0],[i+nx*ny,z+1<nz]] as const) {
        if (valid && !seen[j] && volume[j]! > .05) { seen[j]=1; pending.push(j); }
      }
    }
    if (pressure) connectedToPressure+=airborne;
  }
  return { candidates, candidateMass, connectedToPressure };
}
/** Connected negative-phi cell centres approximate the published surface components. */
function surfaceComponents(snapshot: Snapshot, dims: readonly [number,number,number], field: string) {
  const [nx,ny,nz]=dims,phi=snapshot[field]!,volume=snapshot.volume!;
  const wet=new Uint8Array(volume.length),seen=new Uint8Array(volume.length);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    let sum=0;for(let k=0;k<8;k++)sum+=phi[x+(k&1)+(nx+1)*(y+((k>>1)&1)+(ny+1)*(z+(k>>2)))]!;
    wet[x+nx*(y+ny*z)]=sum<0?1:0;
  }
  const components:{cells:number;mass:number;unsupported:number;centre:number[]}[]=[];
  for(let seed=0;seed<wet.length;seed++){
    if(!wet[seed]||seen[seed])continue;
    const pending=[seed];seen[seed]=1;let cells=0,mass=0,unsupported=0;const centre=[0,0,0];
    while(pending.length){const i=pending.pop()!,x=i%nx,y=Math.floor(i/nx)%ny,z=Math.floor(i/(nx*ny));
      cells++;mass+=volume[i]!;unsupported+=volume[i]!<.05?1:0;centre[0]!+=x;centre[1]!+=y;centre[2]!+=z;
      for(const [j,valid] of [[i-1,x>0],[i+1,x+1<nx],[i-nx,y>0],[i+nx,y+1<ny],[i-nx*ny,z>0],[i+nx*ny,z+1<nz]] as const){
        if(valid&&wet[j]&&!seen[j]){seen[j]=1;pending.push(j);}}
    }
    components.push({cells,mass,unsupported,centre:centre.map(v=>v/cells)});
  }
  components.sort((a,b)=>b.cells-a.cells);
  return {count:components.length,detached:components.slice(1),main:components[0]};
}
function largestSymmetryErrors(volume: Float32Array, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number) => volume[x + nx * (y + ny * z)]!;
  const ranked: { delta: number; xyz: number[]; mirror: number[]; value: number; mirrored: number }[] = [];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    for (const mirror of [[nx - 1 - x, y, z], [x, y, nz - 1 - z], ...(nx===nz?[[z, y, x]]:[])]) {
      const value = at(x, y, z), mirrored = at(mirror[0]!, mirror[1]!, mirror[2]!);
      const delta = Math.abs(value - mirrored);
      if (delta === 0 || (ranked.length === 5 && delta <= ranked[4]!.delta)) continue;
      ranked.push({ delta, xyz: [x, y, z], mirror, value, mirrored });
      ranked.sort((left, right) => right.delta - left.delta);
      if (ranked.length > 5) ranked.pop();
    }
  }
  return ranked;
}
function scalarD4(field: Float32Array, dims: readonly [number, number, number], interiorOnly = false) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number) => field[x + nx * (y + ny * z)]!;
  let maximum = 0, sum = 0, count = 0, signMismatch = 0;
  let worst: { xyz: number[]; mirror: number[]; value: number; mirrored: number; delta: number } | undefined;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (interiorOnly && (x === 0 || y === 0 || z === 0 || x + 1 === nx || y + 1 === ny || z + 1 === nz)) continue;
    for (const mirror of [[nx - 1 - x, y, z], [x, y, nz - 1 - z], ...(nx===nz?[[z, y, x]]:[])]) {
      const value = at(x, y, z), mirrored = at(mirror[0]!, mirror[1]!, mirror[2]!);
      const d = Math.abs(value - mirrored);
      if ((value < 0) !== (mirrored < 0)) signMismatch++;
      if (d > maximum) worst = { xyz: [x, y, z], mirror, value, mirrored, delta: d };
      maximum = Math.max(maximum, d); sum += d; count++;
    }
  }
  return { max: maximum, mean: sum / count, signMismatch, worst };
}
function physicalFaceD4(field: Float32Array, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number, axis: number) =>
    field[4 * (x + nx * (y + ny * z)) + axis]!;
  let maximum = 0, sum = 0, count = 0;
  const worst: { delta: number; face: number[]; reflected: number[]; value: number; mirrorValue: number }[] = [];
  const add = (x: number, y: number, z: number, axis: number,
    rx: number, ry: number, rz: number, raxis: number, sign: number) => {
    const value = at(x, y, z, axis), mirrorValue = sign * at(rx, ry, rz, raxis);
    const delta = Math.abs(value - mirrorValue);
    maximum = Math.max(maximum, delta); sum += delta; count++;
    if (delta > 0 && (worst.length < 3 || delta > worst[2]!.delta)) {
      worst.push({ delta, face: [x, y, z, axis], reflected: [rx, ry, rz, raxis], value, mirrorValue });
      worst.sort((a, b) => b.delta - a.delta);
      if (worst.length > 3) worst.pop();
    }
  };
  // Compare stored interior positive faces only. Negative domain faces are in a separate buffer.
  for (let z = 1; z + 1 < nz; z++) for (let y = 1; y + 1 < ny; y++)
    for (let x = 1; x + 1 < nx; x++) {
      add(x, y, z, 0, nx - 2 - x, y, z, 0, -1);
      add(x, y, z, 2, nx - 1 - x, y, z, 2, 1);
      add(x, y, z, 1, nx - 1 - x, y, z, 1, 1);
      add(x, y, z, 0, x, y, nz - 1 - z, 0, 1);
      add(x, y, z, 2, x, y, nz - 2 - z, 2, -1);
      add(x, y, z, 1, x, y, nz - 1 - z, 1, 1);
      if(nx===nz){add(x, y, z, 0, z, y, x, 2, 1);
      add(x, y, z, 2, z, y, x, 0, 1);
      add(x, y, z, 1, z, y, x, 1, 1);}
    }
  return { max: maximum, mean: sum / count, worst };
}

await acquireWebGPUExclusiveLock("dawn-probe", "Uniform Geometric airborne symmetric expansion A/B");
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  // Probe-only interventions isolate the three responsibilities of airborne
  // momentum without adding experimental switches to simulation code.
  const ablation = process.env.FLUID_AIRBORNE_ABLATION;
  let ablationEdits = 0;
  if (ablation) {
    assert.ok(["authority", "gravity", "projection", "pressure-support", "residual-cutoff", "unclamped-preserve"].includes(ablation));
    const createShaderModule = device.createShaderModule.bind(device);
    device.createShaderModule = descriptor => {
      let code = descriptor.code;
      if(ablation === "unclamped-preserve")code=code.replace("value=clamp(initial,-cell,cell);", "value=initial;");
      if(ablation === "residual-cutoff")code=code.replace(
        /if\(params\.splash\.x>0\.5&&lastNorm>1e-16\)\{let root=q-uvPhi\(q\)\*lastGradient\/\(h\*h\*lastNorm\);\s*value=sign\(initial\)\*min\(band,length\(\(p-root\)\*h\)\);\}\s*else /, "");
      if (ablation === "authority") code = code.replace(
        "return select(rho,max(rho,CM12_LIQUID_ISOVALUE+1e-3),uvAirborneCell(id));", "return rho;");
      if (ablation === "gravity") code = code.replace("||uvAirborneCell(id)||uvAirborneCell(qy)", "");
      if (ablation === "projection") code = code.replace(
        "return select(0.0,predicted,uvAirborneCell(id)||uvAirborneCell(q));", "return 0.0;");
      if (ablation === "pressure-support") code = code.replace(
        /if\(params\.splashB\.w>0\.5\)\{let v=volume\(q\);\s*let h=min\(params\.cellGravity\.x,min\(params\.cellGravity\.y,params\.cellGravity\.z\)\);\s*phi=min\(phi,max\(h\*\(1\.0-v\),-0\.5\*h\)\);\}/, "");
      if (ablation === "gravity" || ablation === "pressure-support") code = code.replace(
        "||(params.splashB.w>0.5&&(volume(id)>1.0||volume(qy)>1.0))", "");
      if (code !== descriptor.code) ablationEdits++;
      return createShaderModule({ ...descriptor, code });
    };
  }
  device = managedGPUDevice(device, { requireWorkerRealm: false });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  const arms = new Map<string, Map<number, Snapshot>>();
  let dims: [number, number, number] = [0, 0, 0];
  for (const mode of modes) {
    const scene = sceneDocument(getSceneDefinition(process.env.FLUID_AIRBORNE_SCENE ?? "symmetric-expansion"));
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
    const gravityStopFrame = Number(process.env.FLUID_AIRBORNE_GRAVITY_STOP_FRAME ?? Infinity);
    const solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined,
      { ...uniformGeometricSolverOptions({ ...valueOverrides, airborneMomentum: mode }, scene),
        ...(process.env.FLUID_AIRBORNE_FIXED_PRESSURE === "1" ? { adaptivePressure:false } : {}),
        ...(process.env.FLUID_AIRBORNE_SYM_LINEAR_EXTENSION === "1" ? { sourceAwareExtension: false } : {}),
        ...(densePressureAudit ? { scratchStorageForQA: "separate" as const } : {}) }, () => {});
    if (ablation) assert.ok(ablationEdits > 0, `no shader matched airborne ${ablation} intervention`);
    const samples = new Map<number, Snapshot>(); arms.set(mode, samples);
    let redistanceCapture:GPUTexture|undefined;
    const surfaceCorrection=(solver as unknown as {surfaceVolumeCorrection?:{encode(encoder:GPUCommandEncoder):void;diagnostics?:GPUBuffer}}).surfaceVolumeCorrection;
    if(process.env.FLUID_AIRBORNE_DROPLETS === "1"&&surfaceCorrection){
      const source=solver.vertexPhiTexture!;
      redistanceCapture=device.createTexture({label:"Probe phi before volume correction",dimension:"3d",format:"r32float",
        size:[source.width,source.height,source.depthOrArrayLayers],usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
      const encode=surfaceCorrection.encode.bind(surfaceCorrection);
      surfaceCorrection.encode=encoder=>{encoder.copyTextureToTexture({texture:source},{texture:redistanceCapture!},
        [source.width,source.height,source.depthOrArrayLayers]);encode(encoder);};
    }

    try {
      dims = [solver.info.nx, solver.info.ny, solver.info.nz];
      if(!process.env.FLUID_AIRBORNE_SCENE)assert.equal(dims[0], dims[2]);
      const capture = async (frame: number) => {
        await device!.queue.onSubmittedWorkDone();
        const audit = solver.symmetryStageAuditTextures!;
        const entries = [
          ["volume", solver.volumeTexture, 1], ["velocity", solver.velocityTexture, 4],

          ["authority", audit.extrapolationDensityAuthority, 1],
          ["preVelocity", audit.preExtrapolationVelocity, 4],
          ["prediction", audit.velocityPrediction, 4],
          ["advection", audit.velocityAdvection, 4],
          ["projection", audit.pressureProjection, 4],
        ] as const;
        const sample: Snapshot = {};
        for (const [name, texture, components] of entries) {
          sample[name] = await read(device!, texture, components, dims);
        }
        const boundaries=solver as unknown as {symmetryStageAuditNegativeBoundaryVelocity:GPUBuffer;boundaryVelocityB:GPUBuffer};
        sample.negativeVelocity=await readBuffer(device!,solver.negativeBoundaryVelocityBuffer);
        sample.negativePreVelocity=await readBuffer(device!,boundaries.symmetryStageAuditNegativeBoundaryVelocity);
        sample.negativeAdvection=await readBuffer(device!,boundaries.boundaryVelocityB);
        const hd: [number, number, number] = [dims[0]+2,dims[1]+2,dims[2]+2];
        const halo = await read(device!, solver.extrapolatedVelocityTexture, 4, hd);
        sample.negativeExtrapolated=new Float32Array(dims[1]*dims[2]+dims[0]*dims[2]+dims[0]*dims[1]);
        for(let z=0;z<dims[2];z++)for(let y=0;y<dims[1];y++)sample.negativeExtrapolated[y+dims[1]*z]=halo[4*hd[0]*(y+1+hd[1]*(z+1))]!;
        for(let z=0;z<dims[2];z++)for(let x=0;x<dims[0];x++)sample.negativeExtrapolated[dims[1]*dims[2]+x+dims[0]*z]=halo[4*(x+1+hd[0]*hd[1]*(z+1))+1]!;
        for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++)sample.negativeExtrapolated[dims[1]*dims[2]+dims[0]*dims[2]+x+dims[0]*y]=halo[4*(x+1+hd[0]*(y+1))+2]!;
        sample.extrapolated = new Float32Array(dims[0]*dims[1]*dims[2]*4);
        for (let z=0;z<dims[2];z++) for(let y=0;y<dims[1];y++) for(let x=0;x<dims[0];x++) {
          const src = 4*((x+1)+hd[0]*((y+1)+hd[1]*(z+1)));
          sample.extrapolated.set(halo.subarray(src,src+4),4*(x+dims[0]*(y+dims[1]*z)));
        }
        sample.phi = await read(device!, solver.vertexPhiTexture!, 1,
          [dims[0] + 1, dims[1] + 1, dims[2] + 1]);
        sample.phiAdvected = await read(device!, solver.advectedVertexPhiTexture!, 1,
          [dims[0] + 1, dims[1] + 1, dims[2] + 1]);
        if(redistanceCapture&&frame>0)sample.phiRedistanced=await read(device!,redistanceCapture,1,
          [dims[0]+1,dims[1]+1,dims[2]+1]);
        if(surfaceCorrection?.diagnostics&&process.env.FLUID_AIRBORNE_DROPLETS === "1")
          sample.surfaceCorrection=await readBuffer(device!,surfaceCorrection.diagnostics);
        const pressureFields = solver.physicsFieldsForQA;
        sample.pressure = await read(device!, pressureFields.pressure, 1,
          pressureFields.latticeDimensions);
        if (densePressureAudit) {
          const multigrid = (solver as unknown as { pressureMultigrid: {
            levels: readonly { rhs: readonly [GPUTexture, GPUTexture];
              phi: readonly [GPUTexture, GPUTexture]; volume: readonly [GPUTexture, GPUTexture];
              coefficients: GPUTexture }[];
            setupRhsSnapshot?: readonly [GPUTexture, GPUTexture] } }).pressureMultigrid;
          const finest = multigrid.levels[0]!;
          for (const [name, texture, components] of [
            ["rhsA", finest.rhs[0], 1], ["rhsB", finest.rhs[1], 1],
            ["pressurePhiA", finest.phi[0], 1], ["pressurePhiB", finest.phi[1], 1],
            ["pressureVolume", finest.volume[0], 4],
            ["pressureCoefficients", finest.coefficients, 4],
            ...(multigrid.setupRhsSnapshot ? [
              ["setupRhsA", multigrid.setupRhsSnapshot[0], 1] as const,
              ["setupRhsB", multigrid.setupRhsSnapshot[1], 1] as const,
            ] : []),
          ] as const) sample[name] = await read(device!, texture, components,
            pressureFields.latticeDimensions);
        }
        if (process.env.FLUID_AIRBORNE_SYM_DUMP === "1" && [2,8,9,10,24,30].includes(frame)) {
          const extension = (solver as unknown as {velocityExtrapolator: {
            resolvedValues: GPUTexture; hierarchyLevels: {dims: [number,number,number]; down:GPUTexture; up:GPUTexture}[]
          }}).velocityExtrapolator;
          const resolved = await read(device!, extension.resolvedValues, 4, dims);
          await writeFile(`/tmp/uniform-extension-resolved-${mode}-${frame}.json`, JSON.stringify({dims, values:Array.from(resolved), authority:Array.from(sample.authority!), preVelocity:Array.from(sample.preVelocity!), phi:Array.from(sample.phi!), phiAdvected:Array.from(sample.phiAdvected!)}));
          const hierarchy = [];
          for (const level of extension.hierarchyLevels) {
            const down = await read(device!,level.down,4,level.dims), up = await read(device!,level.up,4,level.dims);
            hierarchy.push({dims:level.dims, down:physicalFaceD4(down,level.dims),up:physicalFaceD4(up,level.dims)});
          }
          console.error(JSON.stringify({frame, resolved:physicalFaceD4(resolved,dims), hierarchy}));
        }
        if (process.env.FLUID_AIRBORNE_STATE_DUMP === "1" && frame >= Number(process.env.FLUID_AIRBORNE_DUMP_START ?? 22)) {
          await writeFile(`${process.env.FLUID_AIRBORNE_DUMP_PREFIX ?? "/tmp/uniform-airborne"}-state-${mode}-${frame}.json`, JSON.stringify({ dims,
            fields: Object.fromEntries(Object.entries(sample).map(([name, data]) => [name, Array.from(data)])) }));
        }
        samples.set(frame, sample);
        const measured=metrics(sample,dims,scene.container.height_m/dims[1],-scene.fluid.gravity_m_s2.y);
        if(process.env.FLUID_AIRBORNE_ASSERT_SYMMETRY === "1" && mode === "off" && frame<=30){
          assert.ok(measured.d4Mean<1e-4,`frame ${frame}: mean volume symmetry error ${measured.d4Mean}`);
          assert.equal(measured.heightD4,0,`frame ${frame}: liquid column-height symmetry`);
        }
        console.log(JSON.stringify({ mode, frame, time: frame * dt, ...measured,
          ...(frame > 0 ? {energyStages:energyStages(samples.get(frame-1)!,sample,dims)} : {}),
          ...(process.env.FLUID_AIRBORNE_CONNECTIVITY === "1" ? {
            airborneConnectivity: airborneConnectivity(sample,dims,scene.container.height_m/dims[1]) } : {}),
          ...(process.env.FLUID_AIRBORNE_DROPLETS === "1" ? {surfaces:{
            ...(sample.surfaceCorrection?{correction:Array.from(sample.surfaceCorrection)}:{}),
            ...(sample.phiRedistanced?{redistanced:surfaceComponents(sample,dims,"phiRedistanced")}:{}),
            current:surfaceComponents(sample,dims,"phi"),advected:surfaceComponents(sample,dims,"phiAdvected")}} : {}),
          symmetry: { authority: scalarD4(sample.authority!,dims), phi: scalarD4(sample.phi!, [dims[0] + 1, dims[1] + 1, dims[2] + 1]),
            phiAdvected: scalarD4(sample.phiAdvected!, [dims[0] + 1, dims[1] + 1, dims[2] + 1]),
            pressure: scalarD4(sample.pressure!, solver.physicsFieldsForQA.latticeDimensions, true),
            ...(densePressureAudit ? Object.fromEntries(["rhsA", "rhsB", "setupRhsA", "setupRhsB", "pressurePhiA", "pressurePhiB"]
              .map(name => [name, scalarD4(sample[name]!, solver.physicsFieldsForQA.latticeDimensions)])) : {}),
            preVelocity: physicalFaceD4(sample.preVelocity!, dims),
            extrapolated: physicalFaceD4(sample.extrapolated!, dims),
            prediction: physicalFaceD4(sample.prediction!, dims),
            advection: physicalFaceD4(sample.advection!, dims),
            projection: physicalFaceD4(sample.projection!, dims) },
          ...(densePressureAudit ? { pressureRanges: Object.fromEntries(
            ["pressure", "setupRhsA", "setupRhsB", "rhsA", "rhsB"]
              .map(name => [name, range(sample[name]!)])) } : {}),
          ...([2, 5, 12, 20, 24].includes(frame) ? { largestSymmetryErrors: largestSymmetryErrors(sample.volume!, dims) } : {}) }));
      };
      if (checkpoints.has(0)) await capture(0);
      for (let frame = 1; frame <= frames; frame++) {
        if (frame === gravityStopFrame) scene.fluid.gravity_m_s2.y = 0;
        assert.ok(solver.advanceTo(frame * dt, []));
        await solver.awaitFrameCompletion();
        if (checkpoints.has(frame)) await capture(frame);
      }
    } finally { redistanceCapture?.destroy();solver.destroy(); }
  }
  for (const frame of modes.length === 2 ? checkpoints : []) {
    const off = arms.get("off")!.get(frame), on = arms.get("on")!.get(frame);
    if (!off || !on) continue;
    const vector = (name: string) => ["velocity", "extrapolated", "preVelocity", "prediction", "advection", "projection"].includes(name);
    const stages = Object.fromEntries(Object.keys(off).map(name =>
      [name, difference(on[name]!, off[name]!, vector(name) ? 4 : 1)]));
    const firstChanged = Object.keys(stages).filter(name => stages[name]!.max > 0);
    const context = frame >= 23 && frame <= 30 ? (() => {
      const [nx, ny] = dims;
      const cells = [[2, 6, 2], [2, 7, 2], [2, 8, 2], [nx-3, 7, nx-3],
        [1, 8, 1], [nx-2, 8, 1], [nx-2, 8, nx-2]];
      const phiCenter = (sample: Snapshot, x: number, y: number, z: number) => {
        let sum = 0;
        for (let dz = 0; dz <= 1; dz++) for (let dy = 0; dy <= 1; dy++)
          for (let dx = 0; dx <= 1; dx++) sum += sample.phi![x + dx + (nx + 1) * (y + dy + (ny + 1) * (z + dz))]!;
        return sum / 8;
      };
      return cells.map(([x, y, z]) => ({ xyz: [x, y, z],
        ...Object.fromEntries([["off", off], ["on", on]].map(([mode, field]) => {
          const sample = field as Snapshot, i = x! + nx * (y! + ny * z!);
          return [mode, { volume: sample.volume![i], phi: phiCenter(sample, x!, y!, z!),
            authority: sample.authority![i],
            ...Object.fromEntries(["preVelocity", "extrapolated", "prediction", "projection"].map(
              name => [name, Array.from(sample[name]!.subarray(4*i,4*i+3))])) }];
        })) }));
    })() : undefined;
    console.log(JSON.stringify({ comparison: true, frame, time: frame * dt, stages, context,
      ...(firstChanged.length > 0 && frame <= 30 ? {
        largest: Object.fromEntries(firstChanged.filter(name=>!name.startsWith("negative")).map(name => [name,
          largestDifferences(on[name]!, off[name]!,
            name === "phi" || name === "phiAdvected" ? [dims[0]+1,dims[1]+1,dims[2]+1]
              : name === "pressure" ? [dims[0]+2,dims[1]+2,dims[2]+2] : dims,
            vector(name) ? 4 : 1)])),
      } : {}) }));
  }
  assert.deepEqual(errors, []);
} finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
