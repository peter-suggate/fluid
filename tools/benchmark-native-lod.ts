import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { resolveDisplayGrade, sceneLinearToDisplay } from "../lib/core/webgpu-lighting";
import { benchmarkSvoPrimaryPass } from "./benchmark-svo-primary-pass";
import { writeFramePng } from "./write-frame-png";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";

/** Benchmark-only selector variants. Geometry, materials and the full-resolution
 * raster pipeline stay identical; only the permitted projected LOD error varies. */
export function nativeLodStudy(device: GPUDevice, budgets: number[]) {
  assert.ok(budgets[0] === .25 && budgets.every(n => Number.isFinite(n) && n > 0));
  const modules = new WeakMap<GPUShaderModule, GPUShaderModule[]>();
  const pipelines = new WeakMap<GPUComputePipeline, GPUComputePipeline[]>();
  let args: GPUBuffer | undefined;
  let selected = 0;
  const wrapped = new Proxy(device, {
    get(target, property) {
      if (property === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
        const capture = descriptor.label === "Native asset draw-group arguments";
        const buffer = target.createBuffer(capture ? { ...descriptor, usage: descriptor.usage | GPUBufferUsage.COPY_SRC } : descriptor);
        if (capture) args = buffer;
        return buffer;
      };
      if (property === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        const module = target.createShaderModule(descriptor);
        const search = "bitcast<f32>(candidate.z)*pixelsPerMetre<=0.25";
        if (descriptor.code.includes(search)) modules.set(module, budgets.map(budget => target.createShaderModule({
          ...descriptor, label: `${descriptor.label} LOD ${budget}px`,
          code: descriptor.code.replace(search, `bitcast<f32>(candidate.z)*pixelsPerMetre<=${budget.toFixed(6)}`),
        })));
        return module;
      };
      if (property === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
        const pipeline = await target.createComputePipelineAsync(descriptor);
        if (descriptor.label === "Native asset screen-error selection") {
          const variants = modules.get(descriptor.compute.module);
          assert.ok(variants, "native LOD selector source must match");
          pipelines.set(pipeline, await Promise.all(variants.map(module => target.createComputePipelineAsync({
            ...descriptor, compute: { ...descriptor.compute, module },
          }))));
        }
        return pipeline;
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  function encode(encoder: GPUCommandEncoder): GPUCommandEncoder {
    return new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
          const pass = target.beginComputePass(descriptor);
          return new Proxy(pass, {
            get(target, property) {
              if (property === "setPipeline") return (pipeline: GPUComputePipeline) => target.setPipeline(pipelines.get(pipeline)?.[selected] ?? pipeline);
              const value = Reflect.get(target, property, target);
              return typeof value === "function" ? value.bind(target) : value;
            },
          });
        };
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  }
  async function census() {
    assert.ok(args, "native asset draws must exist");
    const read = device.createBuffer({ size: args.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder();encoder.copyBufferToBuffer(args,0,read,0,args.size);
      device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(read.getMappedRange());
      let triangles = 0, instances = 0, draws = 0;
      for (let i=0;i<words.length;i+=5) {
        triangles += words[i]! / 3 * words[i+1]!;instances += words[i+1]!;
        if (words[i+1]) draws++;
      }
      return { triangles, instances, draws };
    } finally { read.destroy(); }
  }
  return { device: wrapped, encode, census, budgets, get drawArgs(){return args;}, select(index: number) { selected = index; } };
}

function half(v: number): number {
  const e=(v>>10)&31,m=v&1023,sign=v&32768?-1:1;
  return sign*(e===0?m*2**-24:e===31?(m?NaN:Infinity):(1+m/1024)*2**(e-15));
}

export interface PrimaryStudy {
  budgets: readonly (number | string)[];
  select(index: number): void;
  encode(encoder: GPUCommandEncoder): GPUCommandEncoder;
  census(): Promise<{ triangles: number; instances: number; draws: number; voxelVertices?: number }>;
  skipReason?(index: number): string | undefined;
}

export async function benchmarkNativeLod(input: {
  device: GPUDevice; study: PrimaryStudy;
  encode(encoder: GPUCommandEncoder): void; capture(): Promise<Uint8Array>;
  views: { name: string; set(): void }[];
  width: number; height: number; cycles: number; outPath: string;
  fullFrame?: boolean;
  grade: ReturnType<typeof resolveDisplayGrade>;
}) {
  const { device, study } = input, results = [];
  mkdirSync(dirname(input.outPath), { recursive: true });
  async function frame(index: number) {
    study.select(index);
    const encoder=device.createCommandEncoder();input.encode(study.encode(encoder));
    device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
  }
  try {
    for (const view of input.views) {
      view.set();await frame(0);
      const referenceBytes=await input.capture(), a=new Uint16Array(referenceBytes.buffer,referenceBytes.byteOffset,referenceBytes.byteLength/2);
      const referenceCensus=await study.census();
      writeFramePng(`${input.outPath}.${view.name}.reference.png`, {width:input.width,height:input.height,packedRows:new Uint32Array(referenceBytes.buffer,referenceBytes.byteOffset,referenceBytes.byteLength/4),grade:input.grade});
      for (let candidate=1;candidate<study.budgets.length;candidate++) {
        const skipped=study.skipReason?.(candidate);
        if(skipped){results.push({view:view.name,budget:study.budgets[candidate],skipped});writeFileSync(input.outPath,JSON.stringify(results,null,2));continue;}
        let frameIndex=0;
        const timing=await benchmarkSvoPrimaryPass(device,encoder=>{
          const slot=frameIndex++%4;study.select(slot===0||slot===3?0:candidate);
          input.encode(study.encode(encoder));
        },8,input.cycles*4,"Voxel surface mesh rasterization");
        const median=(xs:number[])=>xs.sort((a,b)=>a-b)[Math.floor(xs.length/2)]!;
        const reference_ms=median(timing.samples_ms.filter((_,i)=>i%4===0||i%4===3));
        const candidate_ms=median(timing.samples_ms.filter((_,i)=>i%4===1||i%4===2));
        let fullFrame: {reference_ms:number;candidate_ms:number;samples_ms:number[]}|undefined;
        if(input.fullFrame){
          const samples:number[]=[];
          for(let i=-8;i<input.cycles*4;i++){
            const slot=(i+8)%4;study.select(slot===0||slot===3?0:candidate);
            let span:number|undefined;
            for(let attempt=0;attempt<3&&span===undefined;attempt++){
              const encoder=device.createCommandEncoder(),recorder=new GPUPassTimestampRecorder(device,512,"Primary study complete dry frame");
              input.encode(study.encode(recorder.instrument(encoder)));
              if(!recorder.resolve(encoder)){recorder.destroy();continue;}
              device.queue.submit([encoder.finish()]);span=(await recorder.read())?.span_ms;
            }
            assert.ok(span!==undefined,"Full-frame GPU timestamps must resolve");if(i>=0)samples.push(span);
          }
          fullFrame={reference_ms:median(samples.filter((_,i)=>i%4===0||i%4===3)),candidate_ms:median(samples.filter((_,i)=>i%4===1||i%4===2)),samples_ms:samples};
        }
        await frame(candidate);const bytes=await input.capture(),b=new Uint16Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/2);
        let squares=0,over8=0,maximum=0,depthChanges=0,coverageChanges=0,receivers=0;
        for(let p=0;p<a.length;p+=4) {
          if(a[p+3]!==b[p+3])depthChanges++;
          if((half(a[p+3]!)>0)!==(half(b[p+3]!)>0))coverageChanges++;
          if(half(a[p+3]!)<=0&&half(b[p+3]!)<=0)continue;
          const x=sceneLinearToDisplay([half(a[p]!),half(a[p+1]!),half(a[p+2]!)],input.grade);
          const y=sceneLinearToDisplay([half(b[p]!),half(b[p+1]!),half(b[p+2]!)],input.grade);
          let error=0;for(let c=0;c<3;c++){const d=Math.abs(x[c]!-y[c]!)*255;squares+=d*d;error=Math.max(error,d);}
          maximum=Math.max(maximum,error);if(error>8)over8++;receivers++;
        }
        const budget=study.budgets[candidate]!;
        const candidateCensus=await study.census();
        if(referenceCensus.voxelVertices!==undefined)assert.equal(candidateCensus.voxelVertices,referenceCensus.voxelVertices,"Shading variants must preserve the reference voxel vertex count");
        const result={view:view.name,budget,reference_ms,candidate_ms,speedup:reference_ms/candidate_ms,referenceCensus,candidateCensus,
          quality:{receivers,depthChanges,coverageChanges,displayRmse_255:Math.sqrt(squares/(receivers*3)),fractionOver8:over8/receivers,maximum},samples_ms:timing.samples_ms,...(fullFrame?{fullFrame}:{})};
        results.push(result);console.log(`Native LOD study: ${JSON.stringify(result)}`);
        writeFramePng(`${input.outPath}.${view.name}.${budget}${typeof budget === "number" ? "px" : ""}.png`, {width:input.width,height:input.height,packedRows:new Uint32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4),grade:input.grade});
        writeFileSync(input.outPath,JSON.stringify(results,null,2));
      }
    }
  } finally {study.select(0);input.views[0]!.set();await frame(0);}
  return results;
}
