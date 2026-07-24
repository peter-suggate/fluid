import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { OCTREE_POWER_COARSE_LEVELSET_VALID } from "../lib/webgpu-octree-power-coarse-levelset";
import { WebGPUFineLevelSetVolumeCorrection,
  fineLevelSetVolumeCorrectionWGSL,
  unpackFineLevelSetGPUVolumeControl } from "../lib/webgpu-octree-fine-levelset-volume";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("fine volume classifies compact-air overlap only through the authoritative coarse directory", () => {
  const shader = fineLevelSetVolumeCorrectionWGSL.replace(/\s+/g, "");
  const encode = WebGPUFineLevelSetVolumeCorrection.prototype.encode.toString().replace(/\s+/g, "");
  const measure = WebGPUFineLevelSetVolumeCorrection.prototype.encodeMeasurement.toString().replace(/\s+/g, "");
  assert.match(encode, /^encode\(broker\)/);
  assert.doesNotMatch(encode, /newPassBroker|broker\.fence/,
    "volume correction must remain inside the caller-owned publication pass");
  assert.match(measure, /^encodeMeasurement\(broker\)\{this\.encodePasses\(broker,false\);?\}$/,
    "the paper path must measure volume without applying a global phi offset");
  assert.match(shader,
    /fnvalidDirectory\(\)->bool\{if\(arrayLength\(&coarsePublication\)<12u[\s\S]*coarseDirectory\.state==PUBLISHED[\s\S]*coarsePublication\[2\]==coarseDirectory\.rowCount[\s\S]*coarseDirectory\.rowCount<=c\.rowCapacity[\s\S]*all\(coarseDirectory\.dimensions==c\.dimensions\)/,
    "a directory miss is meaningful only after the compact sorted publication header validates");
  assert.match(shader,
    /coarsePublication\[0\]==0u&&coarsePublication\[2\]>0u[\s\S]*coarsePublication\[10\]==coarseDirectory\.generation&&coarsePublication\[11\]==PUBLISHED/,
    "directory authority must agree with the paired compact-coarse publication control");
  assert.match(shader,
    /generation==fineGeneration\|\|generation==priorFineGeneration/,
    "cold topology may share the coarse generation while recurring topology consumes exactly its predecessor");
  assert.match(shader,
    /entry\.row>=coarsePublication\[2\][\s\S]*entry\.physicalVolume<=0\.0/,
    "a malformed entry in the accepted coarse snapshot remains publication-fatal");
  assert.match(shader, /returnvec2u\(low,OWNER_FOUND\)/,
    "fine ownership must retain the sorted directory slot, not reinterpret the pressure row as a slot");
  assert.match(shader, /flat<coarseDirectory\.rowCount&&flat<arrayLength\(&coarseDirectory\.entries\)/,
    "coarse integration must scan the accepted directory rather than next-topology pressure rows");
  assert.doesNotMatch(shader, /hash|probe|maximumHashProbes/i);
  assert.match(shader, /result\.rows!=coarsePublication\[2\]/,
    "the accepted directory must contain exactly every published compact row");
  assert.doesNotMatch(shader, /requested=select\(0u,rowCountSource|headers\[found\.x\]/,
    "target N+1 volume must not validate coarse N through rebuilt N+1 row buffers");
  assert.match(shader,
    /if\(ownership\.y==OWNER_ABSENT\)\{expectedAir=select\(0u,1u,value>=0\.0\);fineVolume=occupancy\(value,h\)\*cellVolume;area=select\(0\.,h\*h,abs\(value\)<=\.5\*h\);samples=1u;\}/,
    "fine liquid inside the proven coarse-air complement must replace zero occupancy, not require a pressure row");
  assert.match(shader,
    /elseif\(ownership\.y!=OWNER_FOUND\)\{lookupFailure=1u;errors\|=ERROR_OWNER;\}/,
    "malformed and probe-exhausted directory queries remain publication-fatal");
  assert.doesNotMatch(shader, /OWNER_ABSENT\)\{if\(value>=0\.0\)/,
    "fine-only liquid must not be confused with an uncertain owner lookup");
  assert.match(shader,
    /letflat=fineLinearWorkgroup\(w,n\)\*64u\+lid;if\(flat==0u\)\{control\.corrected=1u;\}leta=activeSample\(flat\)/,
    "the completed correction pass must publish independently of whether sparse sample zero is valid");
  assert.match(shader, /fnoccupancy\(value:f32,width:f32\)->f32\{returnclamp\(\.5-value\/width,0\.,1\.\);\}/,
    "the conservative controller must use the same compact-field Heaviside width as the published-field QA");
  assert.match(shader,
    /@compute@workgroup_size\(256\)fnfinalizeMeasuredFineVolume\(@builtin\(local_invocation_index\)lid:u32\)\{finalizeCorrectedMeasurement\(false,lid\);\}/,
    "publication telemetry must be remeasured after both bounded correction passes");
  assert.match(shader,
    /fnbalancedReductionRange\(count:u32,lid:u32\)->vec2u\{letwidth=count\/256u;letremainder=count%256u;letbegin=lid\*width\+min\(lid,remainder\)/,
    "final reductions must partition the partial arena into deterministic contiguous lane ranges");
  for (const entryPoint of ["finalizeCoarseVolume", "finalizeFineVolume",
    "finalizeCorrectedFineVolume", "finalizeMeasuredFineVolume"]) {
    assert.match(shader, new RegExp(`@compute@workgroup_size\\(256\\)fn${entryPoint}\\(`),
      `${entryPoint} must use the cooperative hierarchy`);
    assert.doesNotMatch(shader, new RegExp(`@compute@workgroup_size\\(1\\)fn${entryPoint}\\(`),
      `${entryPoint} must not retain a singleton capacity scan`);
  }
  assert.match(shader,
    /fnreduceCoarsePartialHierarchy[\s\S]*for\(vargroup=range\.x;group<range\.y;group\+=1u\)[\s\S]*for\(varstride=128u;stride>0u;stride\/=2u\)/,
    "coarse partials must be reduced in bounded lane chunks and a fixed tree");
  assert.match(shader,
    /fnreduceFinePartialHierarchy[\s\S]*for\(vargroup=range\.x;group<range\.y;group\+=1u\)[\s\S]*for\(varstride=128u;stride>0u;stride\/=2u\)/,
    "all fine measurements must share the deterministic cooperative reduction");
});

test("Dawn total volume is invariant to translating factor-4/factor-8 interfaces and changing band coverage", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const headers = device.createBuffer({ size: 4 * 48, usage: storage });
  const headerWords = new Uint32Array(4 * 12); for (let row = 0; row < 4; row += 1) {
    headerWords[row * 12] = row; headerWords[row * 12 + 3] = 1;
  } device.queue.writeBuffer(headers, 0, headerWords);
  const records = device.createBuffer({ size: 4 * 16, usage: storage });
  const physicalVolumes = device.createBuffer({ size: 16, usage: storage });
  device.queue.writeBuffer(physicalVolumes, 0, new Float32Array([1, 1, 1, 1]));
  const rowCount = device.createBuffer({ size: 4, usage: storage }); device.queue.writeBuffer(rowCount, 0, new Uint32Array([4]));
  const publicationControl = device.createBuffer({ size: 64, usage: storage });
  const writePublication = (buffer: GPUBuffer, rows: number, generation: number) => {
    const words = new Uint32Array(16); words[2] = rows; words[10] = generation;
    words[11] = OCTREE_POWER_COARSE_LEVELSET_VALID; device.queue.writeBuffer(buffer, 0, words);
  };
  writePublication(publicationControl, 4, 1);
  const sampleDirectory = device.createBuffer({ size: 32 + 8 * 32, usage: storage });
  const directory = (dimensions: readonly [number, number, number], rows: readonly number[],
    phiForRow: (row: number) => number = () => 0) => {
    const words = new Uint32Array(8 + 8 * 8), floats = new Float32Array(words.buffer);
    words.set([OCTREE_POWER_COARSE_LEVELSET_VALID, 1, rows.length, 1, ...dimensions], 0); floats[7] = 1;
    for (const [slot, row] of rows.entries()) {
      const base = 8 + slot * 8, value = phiForRow(row);
      words.set([row + 1, 1], base); floats[base + 2] = value; floats[base + 3] = value;
      floats[base + 4] = value; words[base + 5] = 9; words[base + 6] = row;
      floats[base + 7] = 1;
    }
    return words;
  };
  device.queue.writeBuffer(sampleDirectory, 0, directory([4, 1, 1], [0, 1, 2, 3]));
  for (const factor of [4, 8] as const) {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 1, 1],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4,
      maximumResidentBricks: 4 * (factor / 4) ** 3 });
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const sourceA = owner.prepareGPUGeneration(1); const sourceB = owner.prepareGPUGeneration(2);
    const coarse = { headers, records, physicalVolumes, sampleDirectory, publicationControl, rowCount,
      dimensions: [4, 1, 1] as const,
      physicalCellSize: 1, maximumLeafSize: 1, sampleRowCapacity: 8 };
    const volumeA = new WebGPUFineLevelSetVolumeCorrection(device, sourceA, coarse);
    const volumeB = new WebGPUFineLevelSetVolumeCorrection(device, sourceB, coarse, volumeA.control);
    const phases = [[1, 2, false], [2, 3, true], [1, 2, true]] as const;
    const oracle = new FineLevelSetBrickOracle(plan);
    let reference = 0;
    for (let phase = 0; phase < phases.length; phase += 1) {
      const [left, right, grow] = phases[phase];
      const brickWidth = 4 / factor, bx = plan.brickDimensions[0]; const by = plan.brickDimensions[1];
      const keys: number[] = [];
      for (let z = 0; z < plan.brickDimensions[2]; z += 1) for (let y = 0; y < by; y += 1) {
        for (const boundary of [left, right]) {
          const x = Math.min(bx - 1, Math.max(0, Math.floor(boundary / brickWidth)));
          keys.push(packFineLevelSetBrickKey(plan, [x, y, z]));
        }
        if (grow) keys.push(packFineLevelSetBrickKey(plan, [0, y, z]));
      }
      oracle.publishInterfaceAndRing([...new Set(keys)], ([x]) => Math.max(left - x, x - right));
      const source = owner.uploadGeneration(oracle.exportGPUGeneration());
      const currentDirectory = directory([4, 1, 1], [0, 1, 2, 3],
        (row) => Math.max(left - (row + 0.5), row + 0.5 - right));
      currentDirectory[1] = source.generation;
      device.queue.writeBuffer(sampleDirectory, 0, currentDirectory);
      writePublication(publicationControl, 4, source.generation);
      const coarseWords = new ArrayBuffer(64), coarseF = new Float32Array(coarseWords), coarseU = new Uint32Array(coarseWords);
      for (let row = 0; row < 4; row += 1) {
        const value = Math.max(left - (row + 0.5), row + 0.5 - right);
        coarseF[row * 4] = value; coarseF[row * 4 + 1] = value; coarseF[row * 4 + 2] = value;
        coarseU[row * 4 + 3] = 3;
      } device.queue.writeBuffer(records, 0, coarseWords);
      const correction = source.generationSlot === sourceA.generationSlot ? volumeA : volumeB;
      const readback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder(); correction.encode(new PassBroker(encoder));
      encoder.copyBufferToBuffer(correction.control, 0, readback, 0, 64); device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
      let control = unpackFineLevelSetGPUVolumeControl(readback.getMappedRange().slice(0)); readback.unmap(); readback.destroy();
      if (phase > 0) {
        const verify = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const verifyEncoder = device.createCommandEncoder(); correction.encode(new PassBroker(verifyEncoder));
        verifyEncoder.copyBufferToBuffer(correction.control, 0, verify, 0, 64); device.queue.submit([verifyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone(); await verify.mapAsync(GPUMapMode.READ);
        control = unpackFineLevelSetGPUVolumeControl(verify.getMappedRange().slice(0)); verify.unmap(); verify.destroy();
      }
      assert.equal(control.flags, 0x8000_0000, JSON.stringify(control));
      assert.ok(control.samples > 0 && control.coarseRows === 4);
      assert.ok(control.coarseVolume > 0,
        "nonzero coarse partials must survive finalization into the shared control");
      assert.ok(Number.isFinite(control.currentVolume) && Number.isFinite(control.correction));
      assert.ok(Math.abs(control.correction) <= 0.5 * plan.fineCellWidth + 1e-7);
      if (phase === 0) reference = control.referenceVolume;
      assert.ok(Math.abs(control.currentVolume - reference) <= 2e-5,
        `factor ${factor} phase ${phase} total ${control.currentVolume} drifted from ${reference}`);
    }
    volumeB.destroy(); volumeA.destroy(); owner.destroy();
  }
  // Project-specific overlap convention: the compact coarse site set contains
  // liquid rows only. An empty-slot proof may therefore classify a positive
  // fine sample as outside air; lookup uncertainty is never equivalent to air.
  const airPlan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0],
    finestCellDimensions: [2, 1, 1], finestCellWidth: 1,
    fineFactor: 4, brickResolution: 4, maximumResidentBricks: 2 });
  const airOracle = new FineLevelSetBrickOracle(airPlan);
  airOracle.publishInterfaceAndRing([
    packFineLevelSetBrickKey(airPlan, [0, 0, 0]),
    packFineLevelSetBrickKey(airPlan, [1, 0, 0]),
  ], ([x]) => x - 1);
  const airOwner = new WebGPUFineLevelSetBricks(device, airPlan);
  const airSource = airOwner.uploadGeneration(airOracle.exportGPUGeneration());
  const airSampleDirectory = device.createBuffer({ size: 32 + 8 * 32, usage: storage });
  device.queue.writeBuffer(airSampleDirectory, 0, directory([2, 1, 1], [0], () => -0.5));
  const airRowCount = device.createBuffer({ size: 4, usage: storage });
  device.queue.writeBuffer(airRowCount, 0, new Uint32Array([1]));
  const airPublicationControl = device.createBuffer({ size: 64, usage: storage });
  writePublication(airPublicationControl, 1, airSource.generation);
  const oneRecord = new ArrayBuffer(16); const oneRecordF = new Float32Array(oneRecord);
  const oneRecordU = new Uint32Array(oneRecord); oneRecordF.set([-0.5, -0.5, -0.5]); oneRecordU[3] = 3;
  device.queue.writeBuffer(records, 0, oneRecord);
  const airVolume = new WebGPUFineLevelSetVolumeCorrection(device, airSource, {
    headers, records, physicalVolumes, sampleDirectory: airSampleDirectory,
    publicationControl: airPublicationControl, rowCount: airRowCount,
    dimensions: [2, 1, 1], physicalCellSize: 1, maximumLeafSize: 1, sampleRowCapacity: 8,
  });
  const readAirControl = async () => {
    const readback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); airVolume.encode(new PassBroker(encoder));
    encoder.copyBufferToBuffer(airVolume.control, 0, readback, 0, 64); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
    const control = unpackFineLevelSetGPUVolumeControl(readback.getMappedRange().slice(0));
    readback.unmap(); readback.destroy(); return control;
  };
  const positiveAir = await readAirControl();
  assert.equal(positiveAir.flags, 0x8000_0000);
  assert.ok(positiveAir.samples > 0 && positiveAir.fineVolume > 0);
  assert.ok(positiveAir.expectedAirSamples > 0);
  assert.equal(positiveAir.lookupFailureSamples, 0);
  assert.equal(positiveAir.staleOwnerSamples, 0);

  const exhaustedDirectory = directory([2, 1, 1], []);
  for (let slot = 0; slot < 8; slot += 1) exhaustedDirectory.set([100 + slot, 1], 8 + slot * 8);
  device.queue.writeBuffer(airSampleDirectory, 0, exhaustedDirectory);
  const exhausted = await readAirControl();
  assert.equal(exhausted.flags & 4, 4,
    "a malformed live-prefix header must fail closed even for a positive fine sample");
  assert.ok(exhausted.lookupFailureSamples > 0);

  const staleDirectory = directory([2, 1, 1], [0], () => -0.5);
  staleDirectory[8 + 6] = 100;
  device.queue.writeBuffer(airSampleDirectory, 0, staleDirectory);
  const stale = await readAirControl();
  assert.equal(stale.flags & 4, 4, "a found key outside the owner buffers is stale, not air");
  assert.ok(stale.lookupFailureSamples > 0,
    "owner validation must classify a stale directory row as a malformed lookup before sampling it");

  // Retain one published liquid-set row so the directory/control pair stays
  // authoritative while cell zero exercises its proven empty-slot complement.
  const liquidComplement = directory([2, 1, 1], [1], () => 0.5);
  liquidComplement[8 + 6] = 0;
  device.queue.writeBuffer(airSampleDirectory, 0, liquidComplement);
  const missingLiquid = await readAirControl();
  assert.equal(missingLiquid.flags, 0x8000_0000,
    "fine liquid in the authoritative coarse-air complement contributes against zero coarse occupancy");
  assert.ok(missingLiquid.fineVolume > 0);
  airVolume.destroy(); airPublicationControl.destroy(); airRowCount.destroy(); airSampleDirectory.destroy(); airOwner.destroy();
  headers.destroy(); records.destroy(); physicalVolumes.destroy(); sampleDirectory.destroy(); publicationControl.destroy(); rowCount.destroy(); device.destroy();
});
