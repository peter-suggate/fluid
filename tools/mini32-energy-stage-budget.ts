import assert from "node:assert/strict";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Read-only stage copies on a fixed, authored topology; no tracer kernels. */
export async function createMini32EnergyBudget(device: GPUDevice, solver: WebGPUAdaptiveMassSolver,
  cellWidth_m: number, gammaDiffusionEnabled = true, gravity = 9.81, densityOverride?: {cell:number;value:number}) {
  const source = solver.fieldSnapshotSourceForQA;
  let overrideUpload: GPUBuffer | undefined, overrideDestination=0;
  if(densityOverride) {
    const parityCopy=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.topologyArena,4*(source.frameControlBaseWords+source.scalarParityWord),parityCopy,0,4);
    device.queue.submit([encoder.finish()]);await parityCopy.mapAsync(GPUMapMode.READ);
    const destinationParity=new Uint32Array(parityCopy.getMappedRange())[0]!^1;
    parityCopy.unmap();parityCopy.destroy();
    overrideDestination=4*((destinationParity?source.layout.densityB:source.layout.densityA)+densityOverride.cell);
    overrideUpload=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_SRC,mappedAtCreation:true});
    new Float32Array(overrideUpload.getMappedRange())[0]=densityOverride.value;overrideUpload.unmap();
  }
  const records = (await solver.readGPUActivityPolicy()).bricks;
  assert.ok(records.every(r => r.leafId < source.templateWords[13]!), "budget requires authored leaves");
  const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const nc = source.cellCapacity, nr = source.rowCapacity;
  const staticRows = words[3]!;
  const rowBase = words[7]!, termBase = words[8]!, incidenceBase = words[9]!, incidenceRecords = words[10]!;
  const rowEnabled = new Uint8Array(nr);
  for (let row = 0; row < staticRows; row++) {
    const requirements = words[rowBase + staticRows + row]! & 0x0fffffff;
    const owners = words[requirements]!;
    assert.ok(owners <= 32, `valid row ownership count ${owners} for ${row}`);
    if (!owners) continue;
    rowEnabled[row] = Number(Array.from({ length: owners }, (_, i) => {
      const meta = words[requirements + 1 + i]!;
      const record = records[meta >>> 5];
      return record?.active && record.acceptedResolution === (meta & 31);
    }).every(Boolean));
  }
  const cells: { id: number; x: number; z: number; y: number; volume: number; width: number;
    faces: { row: number; axis: number; weight: number }[] }[] = [];
  const levels = 4;
  for (const record of records) {
    if (!record.active) continue;
    const range = words[11]! + 2 * (levels * record.leafId + Math.log2(record.acceptedResolution));
    const first = words[range]!, count = words[range + 1]!;
    for (let id = first; id < first + count; id++) {
      const base = words[6]! + 8 * id;
      const faces = [];
      for (let at = words[incidenceBase + id]!; at < words[incidenceBase + id + 1]!; at++) {
        const row = words[incidenceRecords + 2 * at]!, term = words[incidenceRecords + 2 * at + 1]!;
        if (!rowEnabled[row]) continue;
        faces.push({ row, axis: words[rowBase + staticRows + row]! >>> 30,
          weight: Math.abs(floats[termBase + 2 * term + 1]!) * floats[rowBase + 2 * staticRows + row]! });
      }
      assert.ok(faces.length > 0, `nonempty incidence for cell ${id}`);
      cells.push({ id, x:floats[base]!, z:floats[base+2]!, y: floats[base + 1]!, volume: floats[base + 3]!, width: floats[base + 4]!, faces });
    }
  }
  const captures = new Map<string, GPUBuffer>();
  const stages = ["transport-velocity-extension", "face-preparation", "conservative-transport",
    "gamma-diffusion", "surface-sharpening", "symmetry-authority", "body-forces", "velocity-projection"];
  return {
    cells: cells.length,
    arm() {
      assert.equal(captures.size, 0);
      solver.setStageCaptureForQA((stage, encoder) => {
        if(stage === "symmetry-authority" && overrideUpload)
          encoder.copyBufferToBuffer(overrideUpload,0,source.state,overrideDestination,4);
        if (!stages.includes(stage)) return;
        const bytes = 4 * (10 * nc + 2 * nr + 2);
        const copy = device.createBuffer({ label: `mini32 energy ${stage}`, size: bytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        let offset = 0;
        for (const [base, count] of [[source.layout.densityA, nc], [source.layout.densityB, nc],
          [source.layout.pressure, nc], [source.layout.faceA, nr], [source.layout.faceB, nr], [source.layout.faceVelocitySupport, 4*nc], [source.layout.solidCellOpen,nc], [source.layout.solidVoxelCellOpen,nc], [source.layout.liquid,nc]]) {
          encoder.copyBufferToBuffer(source.state, 4 * base!, copy, offset, 4 * count!);
          offset += 4 * count!;
        }
        for (const word of [source.scalarParityWord, source.faceParityWord]) {
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + word), copy, offset, 4);
          offset += 4;
        }
        captures.set(stage, copy);
      });
    },
    async read() {
      solver.setStageCaptureForQA(undefined);
      const result = [];
      let beforeForces: Float32Array | undefined;
      for (const [stage, buffer] of captures) {
        try {
          await buffer.mapAsync(GPUMapMode.READ);
          const state = new Float32Array(buffer.getMappedRange());
          const parity = new Uint32Array(state.buffer, 4 * (10 * nc + 2 * nr), 2);
          const scalarParity = parity[0]! ^ Number(stages.indexOf(stage) >= 2);
          const faceParity = parity[1]! ^ Number(stages.indexOf(stage) >= 1);
          const densityOffset = stage === "gamma-diffusion" && gammaDiffusionEnabled ? 2 * nc : scalarParity * nc, faceOffset = 3 * nc + faceParity * nr;
          let mass = 0, potential = 0, kinetic = 0, collocatedKinetic = 0;
          let momentumY=0, transportMomentumY=0, transportKinetic=0, gravityLinear=0,gravityQuadratic=0;
          const cellSamples: unknown[] = [];
          const byPhase: Record<string, {mass:number;kinetic:number;potential:number}> = {};
          const byWidth: Record<string, { mass: number; kinetic: number; potential: number }> = {};
          for (const cell of cells) {
            const m = state[densityOffset + cell.id]! * cell.volume * cellWidth_m ** 3;
            const transportAt=3*nc+2*nr+4*cell.id;
            transportMomentumY+=m*state[transportAt+1]!*cellWidth_m;
            transportKinetic+=.5*m*cellWidth_m**2*[0,1,2].reduce((n,a)=>n+state[transportAt+a]!**2,0);
            const forceLinear=[0,0,0],forceQuadratic=[0,0,0];
            const linear = [0, 0, 0], square = [0, 0, 0], weight = [0, 0, 0];
            for (const face of cell.faces) {
              const v = state[faceOffset + face.row]! * cellWidth_m;
              linear[face.axis]! += face.weight * v;
              square[face.axis]! += face.weight * v * v;
              weight[face.axis]! += face.weight;
              if(stage === "body-forces" && beforeForces) {
                const old=beforeForces[faceOffset+face.row]!*cellWidth_m, dv=v-old;
                forceLinear[face.axis]!+=face.weight*old*dv;
                forceQuadratic[face.axis]!+=face.weight*.5*dv*dv;
              }
            }
            let v2 = 0, meanV2 = 0;
            for (let axis = 0; axis < 3; axis++) if (weight[axis]! > 0) {
              v2 += square[axis]! / weight[axis]!;
              gravityLinear+=m*forceLinear[axis]!/weight[axis]!;
              gravityQuadratic+=m*forceQuadratic[axis]!/weight[axis]!;
              meanV2 += (linear[axis]! / weight[axis]!) ** 2;
            }
            const ke = .5 * m * v2, pe = m * gravity * cell.y * cellWidth_m;
            if(process.env.ENERGY_CAPTURE_CELLS === "1") cellSamples.push({id:cell.id,x:cell.x,y:cell.y,z:cell.z,
              density:state[densityOffset+cell.id],open:state[7*nc+2*nr+cell.id]!*state[8*nc+2*nr+cell.id]!,pressureMember:state[9*nc+2*nr+cell.id],mass:m,ke,pe,velocity:linear.map((v,a)=>weight[a]!>0?v/weight[a]!:0)});
            momentumY += m * (weight[1]! > 0 ? linear[1]!/weight[1]! : 0);
            const phase=state[densityOffset+cell.id]!>=.5 ? "core" : "thin";
            const phaseBin=byPhase[phase] ??= {mass:0,kinetic:0,potential:0};
            phaseBin.mass+=m;phaseBin.kinetic+=ke;phaseBin.potential+=pe;
            mass += m; potential += pe; kinetic += ke; collocatedKinetic += .5 * m * meanV2;
            const bin = byWidth[cell.width] ??= { mass: 0, kinetic: 0, potential: 0 };
            bin.mass += m; bin.kinetic += ke; bin.potential += pe;
          }
          assert.ok([mass, potential, kinetic, collocatedKinetic].every(Number.isFinite));
          if(stage === "symmetry-authority") beforeForces=state.slice();
          result.push({ stage, mass, potential, kinetic, collocatedKinetic, total: potential + kinetic, momentumY, transportMomentumY, transportKinetic, gravityLinear,gravityQuadratic, byPhase, byWidth, cellSamples:cellSamples.length?cellSamples:undefined });
        } finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
      }
      captures.clear();
      overrideUpload?.destroy();
      return result;
    },
  };
}
