import { macPressureTargetWGSL } from "./pressure-target";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

/** Aggregate Galerkin hierarchy for the MAC pressure operator. Restriction is
 * Pᵀ (sum), prolongation is piecewise constant P, and Ac=PᵀAfP. Thus solid
 * and ghost-fluid boundary coefficients survive coarsening without inventing
 * a second boundary discretization. A symmetric V-cycle preconditions CG;
 * fresh original-operator residuals control acceptance on the GPU. */
const SHADER = /* wgsl */ `
struct Level { dims:vec4u, child:vec4u }
struct CG { x:f32, r:f32, d:f32, q:f32 }
@group(0) @binding(0) var<uniform> level:Level;
@group(0) @binding(1) var<storage,read_write> matrix:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> rhs:array<f32>;
@group(0) @binding(3) var<storage,read_write> pressure:array<f32>;
@group(0) @binding(4) var<storage,read> childMatrix:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> childPressure:array<f32>;
@group(0) @binding(6) var<storage,read> childRhs:array<f32>;
@group(0) @binding(7) var<storage,read_write> cg:array<CG>;
@group(0) @binding(8) var<storage,read_write> partial:array<vec4f>;
@group(0) @binding(9) var<storage,read_write> scalars:array<f32>;
@group(0) @binding(10) var<uniform> params:array<vec4f,5>;
@group(0) @binding(11) var<storage,read> originalRhs:array<f32>;
@group(0) @binding(12) var<storage,read_write> solutionLow:array<f32>;
var<workgroup> localPressure:array<f32,64>;
var<workgroup> reductions:array<vec2f,64>;
var<workgroup> running:u32;
fn mgActive()->bool{return scalars[12]==0.0&&scalars[21]>0.0&&scalars[20]==0.0;}
fn uniformActive(lane:u32)->bool{if(lane==0u){running=select(0u,1u,mgActive());}workgroupBarrier();return workgroupUniformLoad(&running)!=0u;}
fn linear(id:vec3u)->u32{return id.x+id.y*65535u*64u;}
fn coord(i:u32,d:vec3u)->vec3i{return vec3i(vec3u(i%d.x,(i/d.x)%d.y,i/(d.x*d.y)));}
fn index(q:vec3i,d:vec3u)->u32{return u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));}
fn inside(q:vec3i,d:vec3u)->bool{return all(q>=vec3i(0))&&all(q<vec3i(d));}
fn axis(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn apply(q:vec3i)->f32{
 let i=index(q,level.dims.xyz);if(matrix[i].w==0.0){return 0.0;}var v=matrix[i].w*pressure[i];
 for(var a=0u;a<3u;a++){let e=axis(a);if(inside(q+e,level.dims.xyz)){v-=matrix[i][a]*pressure[index(q+e,level.dims.xyz)];}
 if(inside(q-e,level.dims.xyz)){let j=index(q-e,level.dims.xyz);v-=matrix[j][a]*pressure[j];}}return v;
}
fn childApply(q:vec3i)->f32{
 let i=index(q,level.child.xyz);if(childMatrix[i].w==0.0){return 0.0;}var v=childMatrix[i].w*childPressure[i];
 for(var a=0u;a<3u;a++){let e=axis(a);if(inside(q+e,level.child.xyz)){v-=childMatrix[i][a]*childPressure[index(q+e,level.child.xyz)];}
 if(inside(q-e,level.child.xyz)){let j=index(q-e,level.child.xyz);v-=childMatrix[j][a]*childPressure[j];}}return v;
}
@compute @workgroup_size(64) fn initialize(@builtin(global_invocation_id) id:vec3u){
 let i=linear(id);if(!mgActive()||i>=level.dims.w){return;}if(scalars[3]>0.0){cg[i].r=cg[i].q;}pressure[i]=0.0;rhs[i]=cg[i].r;
}
@compute @workgroup_size(64) fn coarsen(@builtin(global_invocation_id) id:vec3u){
 let i=linear(id);if(!mgActive()||i>=level.dims.w){return;}let q=2*coord(i,level.dims.xyz);var diagonal=0.0;var edges=vec3f(0);
 for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let r=q+o;
 if(!inside(r,level.child.xyz)){continue;}let m=childMatrix[index(r,level.child.xyz)];diagonal+=m.w;
 for(var a=0u;a<3u;a++){if(o[a]==0){diagonal-=2.0*m[a];}else{edges[a]+=m[a];}}}
 matrix[i]=vec4f(edges,max(diagonal,0.0));
}
fn smoothCell(id:vec3u,colour:i32){
 let i=linear(id);if(!mgActive()||i>=level.dims.w){return;}let q=coord(i,level.dims.xyz);
 if(((q.x+q.y+q.z)&1)!=colour||matrix[i].w<=0.0){return;}
 pressure[i]+=(rhs[i]-apply(q))/matrix[i].w;
}
@compute @workgroup_size(64) fn red(@builtin(global_invocation_id) id:vec3u){smoothCell(id,0);}
@compute @workgroup_size(64) fn black(@builtin(global_invocation_id) id:vec3u){smoothCell(id,1);}
@compute @workgroup_size(64) fn restrictResidual(@builtin(global_invocation_id) id:vec3u){
 let i=linear(id);if(!mgActive()||i>=level.dims.w){return;}let q=2*coord(i,level.dims.xyz);var total=0.0;
 for(var k=0u;k<8u;k++){let r=q+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(inside(r,level.child.xyz)){total+=childRhs[index(r,level.child.xyz)]-childApply(r);}}
 rhs[i]=total;pressure[i]=0.0;
}
@compute @workgroup_size(64) fn prolong(@builtin(global_invocation_id) id:vec3u){
 let i=linear(id);if(!mgActive()||i>=level.child.w){return;}if(childMatrix[i].w>0.0){childPressure[i]+=pressure[index(coord(i,level.child.xyz)/2,level.dims.xyz)];}
}
// The coarsest level has at most 4³ rows. Entire visits stay in one workgroup,
// eliminating hundreds of tiny global smoothing dispatches.
@compute @workgroup_size(64) fn coarseSolve(@builtin(local_invocation_index) lane:u32){
 if(!uniformActive(lane)){return;}let valid=lane<level.dims.w;let q=coord(lane,level.dims.xyz);
 localPressure[lane]=0.0;workgroupBarrier();
 for(var sweep=0u;sweep<48u;sweep++){for(var visit=0;visit<4;visit++){let colour=select(visit,3-visit,visit>=2);
 if(valid&&matrix[lane].w>0.0&&((q.x+q.y+q.z)&1)==colour){var value=rhs[lane];
 for(var a=0u;a<3u;a++){let e=axis(a);if(inside(q+e,level.dims.xyz)){value+=matrix[lane][a]*localPressure[index(q+e,level.dims.xyz)];}
 if(inside(q-e,level.dims.xyz)){let j=index(q-e,level.dims.xyz);value+=matrix[j][a]*localPressure[j];}}
 localPressure[lane]=value/matrix[lane].w;}workgroupBarrier();}}
 if(valid){pressure[lane]=localPressure[lane];}
}
fn applyCG(q:vec3i,direction:bool)->f32{
 let i=index(q,level.dims.xyz);if(matrix[i].w==0.0){return 0.0;}var value=matrix[i].w*select(cg[i].x,cg[i].d,direction);
 for(var a=0u;a<3u;a++){let e=axis(a);if(inside(q+e,level.dims.xyz)){let j=index(q+e,level.dims.xyz);value-=matrix[i][a]*select(cg[j].x,cg[j].d,direction);}
 if(inside(q-e,level.dims.xyz)){let j=index(q-e,level.dims.xyz);value-=matrix[j][a]*select(cg[j].x,cg[j].d,direction);}}return value;
}
// Impact RHS values can be 10,000 while the absolute residual bound is 0.001.
// Split each input into 12-bit significands, giving four exact f32 products.
// Accumulate base-2^16 integer limbs and 2^-25 fractional units so Metal
// fast-math cannot reassociate compensation away. Short remainder timesteps
// can produce large warm-start terms even when their residual is small.
// At most 57 terms, each <1e12, keep the high limb below 870 million, the
// low limb below 3.8 million and fractional sum below 2^31. All fit i32.
fn addWideTerm(sum:vec4i,value:f32)->vec4i{
 if(!(abs(value)<1e12)){return vec4i(sum.xyz,1);}
 let whole=trunc(value);let high=trunc(whole/65536.0);let low=whole-high*65536.0;
 return sum+vec4i(i32(high),i32(low),i32((value-whole)*33554432.0),0);
}
fn addWideProduct(sum:vec4i,a:f32,b:f32)->vec4i{
 let ah=bitcast<f32>(bitcast<u32>(a)&0xfffff000u);let al=a-ah;
 let bh=bitcast<f32>(bitcast<u32>(b)&0xfffff000u);let bl=b-bh;
 return addWideTerm(addWideTerm(addWideTerm(addWideTerm(sum,ah*bh),ah*bl),al*bh),al*bl);
}
fn wideResidual(q:vec3i)->f32{
 let i=index(q,level.dims.xyz);if(matrix[i].w==0.0){return originalRhs[i];}
 var sum=addWideProduct(addWideTerm(vec4i(0),originalRhs[i]),-matrix[i].w,cg[i].x);
 sum=addWideProduct(sum,-matrix[i].w,solutionLow[i]);
 for(var a=0u;a<3u;a++){let e=axis(a);
 if(inside(q+e,level.dims.xyz)){let j=index(q+e,level.dims.xyz);sum=addWideProduct(addWideProduct(sum,matrix[i][a],cg[j].x),matrix[i][a],solutionLow[j]);}
 if(inside(q-e,level.dims.xyz)){let j=index(q-e,level.dims.xyz);sum=addWideProduct(addWideProduct(sum,matrix[j][a],cg[j].x),matrix[j][a],solutionLow[j]);}}
 if(sum.w!=0){return 1e30;}
 let low=sum.y+sum.z/33554432;let fraction=sum.z%33554432;
 let high=sum.x+low/65536;let remainder=low%65536;
 if(abs(high)<16384){
  let whole=high*65536+remainder;
  if(abs(whole)<32){return f32(whole*33554432+fraction)/33554432.0;}
  return f32(whole)+f32(fraction)/33554432.0;
 }
 return f32(high)*65536.0+f32(remainder)+f32(fraction)/33554432.0;
}
// Most cells fit the original two-limb accumulator. Only exceptional warm
// starts pay for the wider path; both evaluate the same unmodified stencil.
fn addTerm(sum:vec3i,value:f32)->vec3i{
 if(!(abs(value)<1e7)){return vec3i(sum.xy,1);}
 let whole=trunc(value);return sum+vec3i(i32(whole),i32((value-whole)*33554432.0),0);
}
fn addProduct(sum:vec3i,a:f32,b:f32)->vec3i{
 let ah=bitcast<f32>(bitcast<u32>(a)&0xfffff000u);let al=a-ah;
 let bh=bitcast<f32>(bitcast<u32>(b)&0xfffff000u);let bl=b-bh;
 return addTerm(addTerm(addTerm(addTerm(sum,ah*bh),ah*bl),al*bh),al*bl);
}
fn freshResidual(q:vec3i)->f32{
 let i=index(q,level.dims.xyz);if(matrix[i].w==0.0){return originalRhs[i];}
 var sum=addProduct(addTerm(vec3i(0),originalRhs[i]),-matrix[i].w,cg[i].x);
 sum=addProduct(sum,-matrix[i].w,solutionLow[i]);
 for(var a=0u;a<3u;a++){let e=axis(a);
 if(inside(q+e,level.dims.xyz)){let j=index(q+e,level.dims.xyz);sum=addProduct(addProduct(sum,matrix[i][a],cg[j].x),matrix[i][a],solutionLow[j]);}
 if(inside(q-e,level.dims.xyz)){let j=index(q-e,level.dims.xyz);sum=addProduct(addProduct(sum,matrix[j][a],cg[j].x),matrix[j][a],solutionLow[j]);}}
 if(sum.z!=0){return wideResidual(q);}
 let whole=sum.x+sum.y/33554432;let fraction=sum.y%33554432;
 if(abs(whole)<32){return f32(whole*33554432+fraction)/33554432.0;}
 return f32(whole)+f32(fraction)/33554432.0;
}
fn sumReduce(lane:u32,value:f32)->f32{
 reductions[lane]=vec2f(value,0);workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){reductions[lane]+=reductions[lane+stride];}workgroupBarrier();}let result=reductions[0].x;workgroupBarrier();return result;
}
@compute @workgroup_size(64) fn dotPreconditioned(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(!uniformActive(lane)){return;}let i=linear(id);var value=0.0;if(i<level.dims.w){value=cg[i].r*pressure[i];}let total=sumReduce(lane,value);if(lane==0u){partial[group.x+group.y*65535u]=vec4f(total,0,0,0);}
}
fn globalSum(lane:u32)->f32{var value=0.0;for(var i=lane;i<(level.dims.w+63u)/64u;i+=64u){value+=partial[i].x;}return sumReduce(lane,value);}
@compute @workgroup_size(64) fn beta(@builtin(local_invocation_index) lane:u32){
 if(!uniformActive(lane)){return;}let rz=globalSum(lane);if(lane==0u){scalars[2]=select(rz/max(scalars[0],1e-30),0.0,scalars[10]==0.0||scalars[3]>0.0);scalars[0]=rz;}
}
@compute @workgroup_size(64) fn direction(@builtin(global_invocation_id) id:vec3u){let i=linear(id);if(mgActive()&&i<level.dims.w){cg[i].d=pressure[i]+scalars[2]*cg[i].d;}}
@compute @workgroup_size(64) fn multiply(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(!uniformActive(lane)){return;}let i=linear(id);var value=0.0;if(i<level.dims.w){let q=applyCG(coord(i,level.dims.xyz),true);cg[i].q=q;value=cg[i].d*q;}
 let total=sumReduce(lane,value);if(lane==0u){partial[group.x+group.y*65535u]=vec4f(total,0,0,0);}
}
@compute @workgroup_size(64) fn alpha(@builtin(local_invocation_index) lane:u32){
 if(!uniformActive(lane)){return;}let dAd=globalSum(lane);if(lane==0u){if(!(dAd>0.0&&scalars[0]>0.0)){scalars[12]=1.0;}else{scalars[1]=scalars[0]/dAd;}}
}
// Store pressure as a 12-bit high part plus an f32 low part. The separate low
// update dispatch prevents fast-math from collapsing the compensation. q is
// scratch after its contribution to the residual has been consumed.
@compute @workgroup_size(64) fn resetSolution(@builtin(global_invocation_id) id:vec3u){let i=linear(id);if(mgActive()&&i<level.dims.w){solutionLow[i]=0.0;}}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) id:vec3u){let i=linear(id);if(mgActive()&&i<level.dims.w){
 let next=bitcast<f32>(bitcast<u32>(cg[i].x+(solutionLow[i]+scalars[1]*cg[i].d))&0xfffff000u);
 cg[i].r-=scalars[1]*cg[i].q;cg[i].q=cg[i].x-next;cg[i].x=next;
}}
@compute @workgroup_size(64) fn updateLow(@builtin(global_invocation_id) id:vec3u){let i=linear(id);if(mgActive()&&i<level.dims.w){solutionLow[i]=cg[i].q+(solutionLow[i]+scalars[1]*cg[i].d);}}
// Projection consumes the pressure pair through the existing CG buffer. The
// search direction is no longer needed after the final convergence check.
@compute @workgroup_size(64) fn finishSolution(@builtin(global_invocation_id) id:vec3u){let i=linear(id);if(scalars[12]==0.0&&scalars[21]>0.0&&i<level.dims.w){cg[i].d=solutionLow[i];}}
fn measureResidual(id:vec3u,lane:u32,group:vec3u,initial:bool){
 if(!uniformActive(lane)){return;}let i=linear(id);var value=vec2f(0);var recursive=0.0;
 if(i<level.dims.w){let r=freshResidual(coord(i,level.dims.xyz));
 // Keep the recursive residual paired with the conjugate direction. Replacing
 // it every iteration without restarting breaks that recurrence in f32.
 // q is free after updateLow. Keep a fresh residual there for a reliable
 // restart when the recursive residual has drifted or every 16 iterations.
 if(initial){cg[i].r=r;}cg[i].q=r;recursive=abs(cg[i].r);
 value=vec2f(abs(r),abs(originalRhs[i]));if(!(abs(cg[i].x)<1e30&&abs(r)<1e30)){value.x=1e30;}}
 let recursiveSum=sumReduce(lane,recursive);
 reductions[lane]=value;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){reductions[lane]=max(reductions[lane],reductions[lane+stride]);}workgroupBarrier();}
 if(lane==0u){partial[group.x+group.y*65535u]=vec4f(recursiveSum,reductions[0].x,0,reductions[0].y);}
}
@compute @workgroup_size(64) fn measureInitial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){measureResidual(id,lane,group,true);}
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){measureResidual(id,lane,group,false);}
${macPressureTargetWGSL}
fn check(lane:u32,cycle:bool){
 if(!uniformActive(lane)){return;}let recursiveSum=globalSum(lane);var value=vec2f(0);for(var i=lane;i<(level.dims.w+63u)/64u;i+=64u){value=max(value,partial[i].yw);}
 reductions[lane]=value;workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){reductions[lane]=max(reductions[lane],reductions[lane+stride]);}workgroupBarrier();}
 if(lane==0u){scalars[4]=reductions[0].x;scalars[5]=reductions[0].y;if(cycle){scalars[10]+=1.0;}
 scalars[3]=select(0.0,1.0,cycle&&(u32(scalars[10])%16u==0u||recursiveSum<0.5*reductions[0].x));
 if(!cycle){scalars[32]=reductions[0].x;scalars[33]=pressureTarget(scalars[32],params[3].z,params[0].w);}
 if(reductions[0].x<=scalars[33]){scalars[20]=1.0;}}
}
@compute @workgroup_size(64) fn checkInitial(@builtin(local_invocation_index) lane:u32){check(lane,false);}
@compute @workgroup_size(64) fn checkCycle(@builtin(local_invocation_index) lane:u32){check(lane,true);}
`;

const BINDINGS = {
  initialize: [0, 2, 3, 7, 9], coarsen: [0, 1, 4, 9], red: [0, 1, 2, 3, 9], black: [0, 1, 2, 3, 9],
  restrictResidual: [0, 2, 3, 4, 5, 6, 9], prolong: [0, 3, 4, 5, 9], coarseSolve: [0, 1, 2, 3, 9],
  measureInitial: [0, 1, 7, 8, 9, 11, 12], measure: [0, 1, 7, 8, 9, 11, 12], checkInitial: [0, 8, 9, 10], checkCycle: [0, 8, 9, 10],
  dotPreconditioned: [0, 3, 7, 8, 9], beta: [0, 8, 9], direction: [0, 3, 7, 9], multiply: [0, 1, 7, 8, 9], alpha: [0, 8, 9], update: [0, 7, 9, 12],
  resetSolution: [0, 9, 12], updateLow: [0, 7, 9, 12], finishSolution: [0, 7, 9, 12],
} as const;
type Entry = keyof typeof BINDINGS;
type Level = { dims: [number, number, number]; cells: number; matrix: GPUBuffer; rhs: GPUBuffer; pressure: GPUBuffer; commands: Partial<Record<Entry, { pipeline: GPUComputePipeline; group: GPUBindGroup }>> };
export class MacMultigrid {
  readonly allocatedBytes: number;
  private constructor(private readonly owned: GPUBuffer[], private readonly levels: Level[]) {
    this.allocatedBytes = owned.reduce((sum, buffer) => sum + buffer.size, 0);
  }
  static async create(device: GPUDevice, dimensions: readonly [number, number, number], fine: { matrix: GPUBuffer; rhs: GPUBuffer; cg: GPUBuffer; partial: GPUBuffer; scalars: GPUBuffer; params: GPUBuffer }, signal?: AbortSignal): Promise<MacMultigrid> {
    const owned: GPUBuffer[] = [], levels: Level[] = [];
    const buffer = (label: string, size: number, uniform = false) => {
      const b = device.createBuffer({ label: `MAC multigrid ${label}`, size, usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); owned.push(b); return b;
    };
    try {
      let dims: [number, number, number] = [...dimensions];
      while (true) {
        const cells = dims[0] * dims[1] * dims[2], first = !levels.length;
        levels.push({ dims, cells, matrix: first ? fine.matrix : buffer("coefficients", cells * 16), rhs: buffer("residual RHS", cells * 4), pressure: buffer("pressure", cells * 4), commands: {} });
        if (Math.max(...dims) <= 4) break;
        dims = dims.map(n => Math.ceil(n / 2)) as typeof dims;
      }
      const solutionLow = buffer("pressure low part", levels[0].cells * 4);
      const bundle = await gpuCompilationManagerFor(device).acquire({ id: "mac-aggregate-multigrid", revision: 12, modules: { mg: { source: SHADER } },
        compute: Object.fromEntries(Object.keys(BINDINGS).map(entry => [entry, { layout: "auto" as const, compute: { module: "mg", entryPoint: entry } }])), render: {} }, { signal });
      for (let i = 0; i < levels.length; i++) {
        const level = levels[i], child = levels[Math.max(0, i - 1)], parameters = buffer("level parameters", 32, true);
        device.queue.writeBuffer(parameters, 0, new Uint32Array([...level.dims, level.cells, ...child.dims, child.cells]));
        const resources = [parameters, level.matrix, level.rhs, level.pressure, child.matrix, child.pressure, child.rhs, fine.cg, fine.partial, fine.scalars, fine.params, fine.rhs, solutionLow];
        for (const entry of Object.keys(BINDINGS) as Entry[]) {
          // Only create layouts used at this level; level zero aliases its child.
          if (i === 0 && ["coarsen", "restrictResidual", "prolong"].includes(entry)) continue;
          const pipeline = bundle.compute[entry];
          level.commands[entry] = { pipeline, group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: BINDINGS[entry].map(binding => ({ binding, resource: { buffer: resources[binding] } })) }) };
        }
      }
      return new MacMultigrid(owned, levels);
    } catch (error) { owned.forEach(b => b.destroy()); throw error; }
  }
  encode(encoder: GPUCommandEncoder, cycleLimit: number): void {
    const pass = encoder.beginComputePass({ label: "MAC Galerkin multigrid pressure" });
    const run = (i: number, entry: Entry) => {
      const level = this.levels[i], command = level.commands[entry]!;
      const cells = entry === "prolong" ? this.levels[i - 1].cells : level.cells;
      const groups = ["coarseSolve", "checkInitial", "checkCycle", "beta", "alpha"].includes(entry) ? 1 : Math.ceil(cells / 64);
      pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.group); pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    };
    run(0, "resetSolution"); run(0, "measureInitial"); run(0, "checkInitial");
    for (let i = 1; i < this.levels.length; i++) run(i, "coarsen");
    const cycle = (i: number) => {
      if (i === this.levels.length - 1) { run(i, "coarseSolve"); return; }
      for (let sweep = 0; sweep < 2; sweep++) { run(i, "red"); run(i, "black"); }
      run(i + 1, "restrictResidual"); cycle(i + 1); run(i + 1, "prolong");
      for (let sweep = 0; sweep < 2; sweep++) { run(i, "black"); run(i, "red"); }
    };
    for (let i = 0; i < cycleLimit; i++) {
      run(0, "initialize"); cycle(0); run(0, "dotPreconditioned"); run(0, "beta"); run(0, "direction");
      run(0, "multiply"); run(0, "alpha"); run(0, "update"); run(0, "updateLow"); run(0, "measure"); run(0, "checkCycle");
    }
    run(0, "finishSolution");
    pass.end();
  }
  destroy(): void { this.owned.forEach(buffer => buffer.destroy()); }
}
