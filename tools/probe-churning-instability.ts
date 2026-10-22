/** Controlled churning-scene ablations and full-session volume measurements.
 * Default: the authored motor session. "blade" holds the blade still;
 * "blade-disconnected" restores the old frozen pressure-link defect;
 * "blade-no-freeze" disables freezing; dt30/dt60/dt120 select the clock.
 * To replay identical failing pressure input against the fixed operator, add
 * --legacy-inactive-links --stages --pressure-trace --matrix-only
 * --snapshot-pressure --compare-fixed.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-churning-instability.ts
 * Add --receipts to record pressure residuals and freeze transitions,
 * --trace-failure to capture a failed trajectory's two sample positions.
 */
import { installChurningStageTrace, dumpChurningPressure, installPressureTrace } from "./churning-stage-trace";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import "../lib/methods";
import { getScenePreset } from "../lib/core/scenes";
import { initializeRigidBodies, quaternionInverseRotate } from "../lib/core/rigid-body";
import { samplePrescribedRigidBody } from "../lib/core/prescribed-rigid-motion";
import { sceneShape } from "../lib/core/scene-shape";
import { sceneAtFinestCellSize } from "../lib/core/scene-scale";
import { createUniformSolver, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { mixedPhysical, readMixedTexture, readMixedTileWords, readMixedBuffer } from "../tests/helpers/uniform-mixed-native-fields";
import { uniformTileKey } from "../lib/methods/uniform/uniform-lattice-address";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const cases = (process.argv.find(arg => arg.startsWith("--cases="))?.slice(8)
  ?? "baseline").split(",");
const rows: unknown[] = [];
const seconds = Number(process.argv.find(arg => arg.startsWith("--seconds="))?.slice(10) ?? 2);
const receipts = process.argv.includes("--receipts");
const fence=process.argv.find(arg=>arg.startsWith("--fence="))?.slice(8);
const hz=Number(process.argv.find(arg=>arg.startsWith("--hz="))?.slice(5)??0);
const fill=Number(process.argv.find(arg=>arg.startsWith("--fill="))?.slice(7)??0);
const stopStep=Number(process.argv.find(arg=>arg.startsWith("--stop-step="))?.slice(12)??0);
const seamHead = process.argv.find(arg=>arg.startsWith("--seam-head="))?.slice(12);
const stageTrace = process.argv.includes("--stages");
const legacyLinksRequested=process.argv.includes("--legacy-inactive-links");let legacyLinks=legacyLinksRequested;
const traceFailure = process.argv.includes("--trace-failure");
const interval = Number(process.argv.find(arg => arg.startsWith("--interval="))?.slice(11) ?? 15);
if (!Number.isFinite(seconds) || seconds <= 0) throw new Error("--seconds must be positive and finite");
if (!Number.isInteger(interval) || interval <= 0) throw new Error("--interval must be a positive integer");
const output = process.argv.find(arg => arg.startsWith("--out="))?.slice(6) ?? "artifacts/churning-instability/probe.json";
await mkdir(dirname(output), { recursive: true });
paceDawnEventPolling();
await withUniformDevice("churning instability", async device => {
  for (const name of cases) {
    legacyLinks=legacyLinksRequested;
    let scene = getScenePreset("churning-tank").create();
    if (name.endsWith("-roomy")) scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: 66 };
    if(hz) scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/hz;
    if(fill) scene.container.fillFraction=fill/scene.container.height_m;
    if (name === "empty") scene.rigidBodies = [];
    if (name.startsWith("blade")) scene.rigidBodies = scene.rigidBodies.filter(body => body.shape === "box");
    if (name === "handle") scene.rigidBodies = scene.rigidBodies.filter(body => body.shape === "cylinder");
    if (["stationary", "handle"].includes(name) || name.startsWith("blade")) {
      for (const body of scene.rigidBodies) body.prescribedMotion!.startTime_s = 100;
    }
    if (name === "no-capillarity") scene.fluid.surfaceTension_N_m = 0;
    if (name === "translation") for (const body of scene.rigidBodies) body.prescribedMotion!.rotateWithOrbit = false;
    if (name === "welded") for (const body of scene.rigidBodies) body.prescribedMotion!.rotateWithOrbit = true;
    if (name === "coarse") scene = sceneAtFinestCellSize(scene, 0.05);
    if (name === "dt30") scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
    if (name === "dt60") scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
    if (name === "dt120") scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
    if (name === "dt240") scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 240;
    if (name === "blade-no-gravity") scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    if (name === "blade-centered") {
      scene.rigidBodies[0]!.position_m.x = 0;
      delete scene.rigidBodies[0]!.prescribedMotion;
    }
    if (name.startsWith("blade-fine")) scene.fluid.refinementRegions = [{ id: "all-fine", rule: "minimum-cell-size",
      minimumCellSize_cells: 1, maximumCellSize_cells: 1,
      min_m: { x: -1.2, y: 0, z: -1.2 }, max_m: { x: 1.2, y: 1.6, z: 1.2 } }];
    const bodies = initializeRigidBodies(scene.rigidBodies), dt = scene.numerics.fixedDt_s;
    console.log(JSON.stringify({ case: name, phase: "compile", dt, h: scene.voxelDomain.finestCellSize_m }));
    const progress = setInterval(() => {
      const status = gpuCompilationManagerFor(device).snapshot();
      console.log(JSON.stringify({ case: name, phase: "compile", active: status.active, queued: status.queued, label: status.progress?.label }));
    }, 30000);
    let rewrites = 0;
    const traceWord = 24; // Eight unused words in the particle receipt header.
    const probeDevice = new Proxy(device, { get(target, key) {
      if (key === "createBuffer" && (receipts || traceFailure)) return (descriptor: GPUBufferDescriptor) => {
        return target.createBuffer({ ...descriptor,
          usage: descriptor.usage | (descriptor.label === "Uniform cold freeze" ? GPUBufferUsage.COPY_SRC : 0) });
      };
      if (key === "createShaderModule" && (name === "blade-disconnected" || traceFailure || seamHead || legacyLinks)) return (descriptor: GPUShaderModuleDescriptor) => {
        // Reproduce the original frozen/live pressure disconnection.
        const before = "// Frozen liquid still has native pressure rows.";
        let code = name === "blade-disconnected" ? descriptor.code.replace(before, "else if(l[axis]==select(0u,3u,sign>0)&&umFrozenCell(q)){kind=K_NEUMANN;}\n" + before) : descriptor.code;
        if(legacyLinks){code=code.replace(/for\(var axis=0u;axis<3u;axis\+\+\)\{\s*let bit=l\[axis\]&1u;let inner=[\s\S]*?if\(cDiag\(cAddr\(qo\)\)>0\.0\)\{cross\[axis\]=pF\(K\+outer\*PL,at\);\}\s*\}/,
          'for(var axis=0u;axis<3u;axis++){let bit=l[axis]&1u;part-=pF(K+(2u*axis+1u-bit)*PL,at);cross[axis]=pF(K+(2u*axis+bit)*PL,at);}');}
        if(seamHead==="all")code=code.replace("if(!bLiquid(pool[P_KINDS+at])){return 0.0;}", "");
        if(seamHead&&seamHead!=="all") code=code.replace("let y=f32(((at%UBA_STRIDE)/8u)&3u)-1.5;",(seamHead==="wet"?"if(!bLiquid(pool[P_KINDS+at])){return 0.0;}\n":seamHead==="off"?"return 0.0;\n":"")+"let y=f32(((at%UBA_STRIDE)/8u)&3u)-1.5;");
        if (traceFailure) code = code.replaceAll("atomicStore(&state[3],1u);return NbMoved(p,1u);", `
          if(atomicExchange(&state[${traceWord}u],1u)==0u){
            atomicStore(&state[${traceWord+1}u],stage);
            for(var a=0u;a<3u;a++){
              atomicStore(&state[${traceWord+2}u+a],bitcast<u32>(umSolidTraceWorld(q)[a]));
              atomicStore(&state[${traceWord+5}u+a],bitcast<u32>(umSolidTraceWorld(x)[a]));
            }
          }
          atomicStore(&state[3],1u);return NbMoved(p,1u);`);
        if(traceFailure) code=code.replace("let k=nbMoveSample(x);", `let k=nbMoveSample(x);
          if(any(abs(k)>vec3f(100.0))&&atomicExchange(&state[24u],1u)==0u){
            atomicStore(&state[25u],stage+10u);
            for(var a=0u;a<3u;a++){atomicStore(&state[26u+a],bitcast<u32>(umSolidTraceWorld(x)[a]));atomicStore(&state[29u+a],bitcast<u32>(k[a]));}
          }`);
        if (code !== descriptor.code) rewrites++;
        return target.createShaderModule({ ...descriptor, code });
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    const solver = await createUniformSolver(probeDevice, scene, { rigidCoupling: "on",
      ...(name.includes("no-freeze") ? { coldRegions: "fine" } : {}),
      ...(name.includes("no-skip") ? { coldFrameSkip: "off" } : {}),
    }).finally(() => clearInterval(progress));
    if (name === "blade-full-return") {
      (solver as unknown as { mixedFrame: { split: { transfer: { skipProjectedBandReturns: boolean } } } }).mixedFrame.split.transfer.skipProjectedBandReturns = false;
    }
    if (name === "blade-disconnected" && rewrites === 0) throw new Error("Frozen pressure-link experiment did not match its shader");
    console.log(JSON.stringify({ case: name, phase: "ready", rewrites }));
    if(fence){
      const frame=(solver as any).mixedFrame;
      if(fence==="all")frame.surfaceWorkControl.framePlan.coalescePasses=false;
      else{
        const target:Record<string,[any,string]>={move:[frame.narrowBandFlip,"move"],momentum:[frame,"encodeParticleMomentum"],forces:[frame.forces,"encode"],pressure:[frame.band,"encodeProject"],extension:[frame.extension,"finishPair"]};
        const [object,method]=target[fence]!;const original=object[method].bind(object);
        object[method]=(encoder:GPUCommandEncoder,...args:any[])=>{const result=original(encoder,...args);frame.blit.finishStage(encoder);return result;};
      }
    }
    if(process.argv.includes('--skip-native-projection')){const band=(solver as any).mixedFrame.band,dispatch=band.dispatch.bind(band);band.dispatch=(pass:any,key:string,launch:any)=>{if(key!=='pProjectNative'&&key!=='pCopyNative')dispatch(pass,key,launch);};}
    const stages = stageTrace ? await installChurningStageTrace(device, solver) : undefined;
    const mg=process.argv.includes("--pressure-trace")?await installPressureTrace(device,solver):undefined;
    try {
      for (let step = 1; step <= Math.round(seconds / dt); step++) {
        const time = step * dt;
        const sample = [1, 2, 4, 8].includes(step) || step % interval === 0 || (step >= 58 && step <= 70);
        if (sample) (solver as unknown as { mixedFrame: { requestNarrowBandVolumeProbe(): void } }).mixedFrame.requestNarrowBandVolumeProbe();
        stages?.reset();mg?.reset();
        if (!solver.advanceTo(time, bodies)) throw new Error(`advance refused at ${step}`);
        await solver.awaitFrameCompletion();
        if (stages) {
          const maxima = await stages.read();
          if (sample || Object.values(maxima).some(v => v! > 30)) {
            const trace = {case:name,step,time,maxima};console.log(JSON.stringify(trace));rows.push(trace);
          }
          if (stopStep ? step===stopStep : Object.values(maxima).some(v => v! > 30)) {
            if(mg){const pressureTrace=await mg.read(),coarse=await mg.dump();console.log(JSON.stringify({step,pressureTrace,coarse}));rows.push({step,pressureTrace,coarse});if(process.argv.includes("--snapshot-pressure")){await mg.replay();const replayTrace=await mg.read(),replayCoarse=await mg.dump();console.log(JSON.stringify({step,replayTrace,replayCoarse}));rows.push({step,replayTrace,replayCoarse});if(legacyLinks&&(process.argv.includes('--compare-fixed')||process.argv.includes('--compare-tail'))){
                legacyLinks=false;const band=(solver as any).mixedFrame.band,mirror=band.simulation.mirrorCurrent;band.simulation.mirrorCurrent=true;band.schedule.coarseScale+=Number.EPSILON;try{await band.initialize();}finally{band.simulation.mirrorCurrent=mirror;}await mg.replay();const fixedTrace=await mg.read(),fixedCoarse=await mg.dump();console.log(JSON.stringify({step,fixedTrace,fixedCoarse}));rows.push({step,fixedTrace,fixedCoarse});
              }}}
            const detail=await dumpChurningPressure(device,solver);console.log(JSON.stringify({step,detail}));rows.push({step,detail});break;
          }
        }
        if (!sample) continue;
        const stats = await solver.readStats();
        const address = await readUniformLiveAddress(device, solver);
        const [velocity, volume, tiles] = await Promise.all([
          readMixedTexture(device, solver.velocityTexture, address),
          readMixedTexture(device, solver.volumeTexture, address),
          readMixedTileWords(device, solver),
        ]);
        const { nx, ny } = solver.info, h = scene.voxelDomain.finestCellSize_m;
        const origin = (solver as unknown as { frame: { origin_m: number[] } }).frame.origin_m;
        const poses = bodies.map(body => samplePrescribedRigidBody(body, time, dt));
        let peak = 0, rms = 0, wet = 0, maximumY = 0, at: number[] = [], peakVector: number[] = [];
        for (let i = 0; i < volume.length; i++) {
          const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
          const tile = tiles[uniformTileKey(address, x >> 2, y >> 2, z >> 2)]!;
          const width = (tile & 0x80000000) ? 1 : 4;
          if (x % width || y % width || z % width) continue;
          if (!(volume[i]! > .1)) continue;
          const p = { x: origin[0]! + (x + width / 2) * h,
            y: origin[1]! + (y + width / 2) * h, z: origin[2]! + (z + width / 2) * h };
          if (poses.some(body => sceneShape(body.description.shape).distance_m(body.description.dimensions_m,
            quaternionInverseRotate(body.orientation, { x: p.x - body.position_m.x, y: p.y - body.position_m.y, z: p.z - body.position_m.z })) < h)) continue;
          const value = [1, nx, nx * ny].map((stride, axis) => {
            const high = velocity[4 * (i + (width - 1) * stride) + axis]!;
            const low = [x, y, z][axis]! > 0 ? velocity[4 * (i - stride) + axis]! : 0;
            return (high + low) / 2;
          });
          const speed = Math.hypot(...value);
          if (speed > peak) { peak = speed; at = [p.x, p.y, p.z]; peakVector = value; }
          rms += speed * speed * width ** 3; wet += width ** 3; maximumY = Math.max(maximumY, p.y);
        }
        let pressure: unknown;
        if (receipts) {
          const frame = (solver as unknown as { mixedFrame: { band: { index: GPUBuffer }; freeze?: { buffer: GPUBuffer; shape: { regions: number } } } }).mixedFrame;
          const arena = await readMixedBuffer(device, frame.band.index), bits = new Uint32Array(arena.buffer);
          const freeze = frame.freeze ? await readMixedBuffer(device, frame.freeze.buffer) : undefined;
          const states = [0, 0, 0, 0]; let maximumTerm = 0;
          if (freeze) for (let r = 0; r < frame.freeze!.shape.regions; r++) {
            states[freeze[4*r+3]!]!++;
            maximumTerm = Math.max(maximumTerm, Math.hypot(freeze[4*r]!, freeze[4*r+1]!, freeze[4*r+2]!));
          }
          pressure = { bandTiles: bits[0], residual: arena[2], cycles: bits[4], freezeStates: states, maximumTerm };
        }
        const row = { case: name, step, time, peak, rms: Math.sqrt(rms / Math.max(1, wet)), at, peakVector, maximumY, pressure,
          pages: { ...solver.pagePoolInfo },
          volume_m3: stats.volumeCellSum! * h ** 3, volumeDrift: stats.narrowBandVolumeBudgetDrift,
          outflow_m3: (stats.narrowBandOutflowVolume_cells ?? 0) * h ** 3,
          motorSpeed: Math.max(0, ...poses.map(body => Math.hypot(...Object.values(body.linearVelocity_m_s)))),
          completedTime: solver.info.completedTime_s };
        console.log(JSON.stringify(row)); rows.push(row);
        if (peak > 30 || !Number.isFinite(rms)) break;
      }
    } catch (error) {
      if(stages){const maxima=await stages.read(); console.log(JSON.stringify({case:name,failedMaxima:maxima}));rows.push({case:name,failedMaxima:maxima});}
      const row = { case: name, error: String(error) }; console.log(JSON.stringify(row)); rows.push(row);
      if (traceFailure) {
        const state = (solver as unknown as { mixedFrame: { narrowBandFlip: { state: GPUBuffer } } }).mixedFrame.narrowBandFlip.state;
        const values = await readMixedBuffer(device, state), bits = new Uint32Array(values.buffer);
        const trace = { case: name, captured: bits[traceWord], stage: bits[traceWord+1],
          receipt: Array.from(bits.slice(0, 24)),
          worldStart: Array.from(values.slice(traceWord+2, traceWord+5)), worldSample: Array.from(values.slice(traceWord+5, traceWord+8)) };
        console.log(JSON.stringify(trace)); rows.push(trace);
        const raw = await readMixedTexture(device, mixedPhysical(solver.velocityTexture));
        let max = 0, at = 0;
        for (let i = 0; i < raw.length; i++) if (i % 4 !== 3 && Math.abs(raw[i]!) > max) { max = Math.abs(raw[i]!); at = i; }
        const grid = { case: name, maximumRawGridVelocity: max, rawIndex: at };
        console.log(JSON.stringify(grid)); rows.push(grid);
      }
    } finally { stages?.destroy();mg?.destroy(); solver.destroy(); }
    await writeFile(output, JSON.stringify(rows, null, 2) + "\n");
  }
}, ["subgroups"]);
