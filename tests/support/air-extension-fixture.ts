import { MOMENTUM_SNAPSHOT_ENTRY_POINTS, momentumSnapshotLayout } from "../../lib/methods/adaptive-volume/sparse-cm12-momentum-snapshot";
import { createMomentumSnapshotWGSL } from "../../lib/methods/adaptive-volume/sparse-cm12-momentum-snapshot.wgsl";
import { AIR_EXTENSION_ENTRY_POINTS, AIR_EXTENSION_ITERATIONS, AIR_EXTENSION_SWEEPS, airExtensionLayout,
  decodeAirExtensionReceipt } from "../../lib/methods/adaptive-volume/sparse-cm12-air-extension";
import { createAirExtensionWGSL } from "../../lib/methods/adaptive-volume/sparse-cm12-air-extension.wgsl";
import { createSparseAdaptiveMassAtlas, sparseBrickKey } from "../../lib/methods/adaptive-volume/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../../lib/methods/adaptive-volume/sparse-atlas-composite-projection";

export interface AirCell { center: number[]; widths: number[]; phi: number; volume: number; velocity: number[] }
export interface AirFace { center: number[]; axis: number; open: number; solid: number; weight: number;
  velocity: number; terms: { cell: number; coefficient: number }[]; prescribed?: boolean }
export interface AirFixture { cells: AirCell[]; faces: AirFace[]; dimensions: number[]; origin?: number[] }

export function mixedAirFixture(reflected = false): AirFixture {
  const atlas=createSparseAdaptiveMassAtlas([8,4,4],[0,1].map(x=>{
    const resolution=((x===0)!==reflected?2:4) as 2|4;
    return {key:sparseBrickKey([x,0,0],[2,1,1]),coordinate:[x,0,0] as const,resolution,
      density:new Float64Array(resolution**3),gamma:new Float64Array(resolution**3).fill(1)};
  }),2,4);
  const grid=buildSparseAtlasCompositeGrid(atlas);
  return {dimensions:[8,4,4],cells:grid.cells.map(c=>({center:[...c.centerFine],widths:[...c.widthsFine],
    phi:c.centerFine[1]-.9,volume:c.volume,velocity:[Math.sin(c.centerFine[0]),.2*Math.cos(c.centerFine[1]),.3*Math.sin(c.centerFine[2])]})),
    faces:grid.gradientRows.map(r=>({center:[...r.centerFine],axis:r.axis,open:r.terms.length===1?0:1,
      solid:0,weight:r.dualWeight,velocity:r.terms.length===1?0:[Math.sin(r.centerFine[0]),.2*Math.cos(r.centerFine[1]),.3*Math.sin(r.centerFine[2])][r.axis]!,terms:r.terms.map(t=>({cell:t.cellId,coefficient:t.coefficient}))}))};
}

/** Cartesian and rectilinear manufactured graphs; no pressure solver is involved. */
export function airFixture(widths = [1, 1, 1, 1], enclosed = false): AirFixture {
  const dimensions = [widths.reduce((a,b)=>a+b,0), 4, 4];
  const cells: AirCell[] = [], faces: AirFace[] = [];
  const nx=widths.length, ny=4, nz=4;
  const id=(x:number,y:number,z:number)=>x+nx*(y+ny*z);
  const edges=[0];for(const width of widths)edges.push(edges.at(-1)!+width);
  for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const center=[edges[x]!+widths[x]!/2,y+.5,z+.5];
    cells.push({center,widths:[widths[x]!,1,1],phi:enclosed ? .5 : center[1]!-.9,
      volume:widths[x]!,velocity:[Math.sin(center[0]!),.2*Math.cos(center[1]!),.3*Math.sin(center[2]!)]});
  }
  for(let axis=0;axis<3;axis++)for(let z=0;z<nz+(axis===2?1:0);z++)
    for(let y=0;y<ny+(axis===1?1:0);y++)for(let x=0;x<nx+(axis===0?1:0);x++){
      const q=[x,y,z], size=[nx,ny,nz];const terms:AirFace["terms"]=[];
      for(const side of [-1,0]){const c=[...q];c[axis]!+=side;
        if(c[axis]!>=0&&c[axis]!<size[axis]!)terms.push({cell:id(c[0]!,c[1]!,c[2]!),coefficient:side===-1?-1:1});}
      const cell=cells[terms[0]!.cell]!;
      const distance=terms.reduce((s,t)=>s+cells[t.cell]!.widths[axis]!/2,0);
      const area=cell.volume/cell.widths[axis]!;
      terms.forEach(t=>t.coefficient/=distance);
      const center=[...cell.center];center[axis]=axis===0?edges[q[axis]!]!:q[axis]!;
      const boundary=terms.length===1;
      faces.push({center,axis,open:boundary?0:1,solid:boundary&&enclosed&&q[axis]===0?.1:0,weight:area*distance,
        // Fixed wall flux deliberately makes enclosed compatibility nonzero.
        velocity:boundary?(enclosed && q[axis]===0 ? .1 : 0):[Math.sin(center[0]!),.2*Math.cos(center[1]!),.3*Math.sin(center[2]!)][axis]!,terms});
    }
  return {cells,faces,dimensions};
}

export async function runAirFixture(device: GPUDevice, fixture: AirFixture,
  options: { iterations?: number; samples?: number[][]; project?: boolean; snapshot?: boolean } = {}) {
  const {cells,faces,dimensions}=fixture;const origin=fixture.origin??[0,0,0];const upper=dimensions.map((n,a)=>n+origin[a]!);const n=cells.length,r=faces.length;
  const layout=airExtensionLayout(n,r);
  const samples=options.samples??[];
  const terms=faces.flatMap(f=>f.terms);const snapshotLayout=momentumSnapshotLayout(n,r,terms.length);let offset=0;
  const ranges=faces.map(f=>{const start=offset;offset+=f.terms.length;return [start,offset];});
  const incidence=cells.map((_,c)=>terms.flatMap((t,i)=>t.cell===c?[i]:[]));
  const incidenceRows=terms.map((_,t)=>ranges.findIndex(([a,b])=>t>=a!&&t<b!));
  const incidenceTerms=incidence.flat();offset=0;
  const cellRanges=incidence.map(list=>{const start=offset;offset+=list.length;return [start,offset];});
  const array=(name:string,type:string,values:string[])=>`const ${name}=array<${type},${Math.max(1,values.length)}>(${values.length?values.join(","):type+"(0)"});`;
  const number=(n:number)=>Number.isInteger(n)?`${n}.0`:String(n);
  const vector=(v:number[])=>`vec3f(${v.map(number).join(",")})`;
  const code=`
@group(0)@binding(0)var<storage,read_write>partials:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
@group(0)@binding(2)var<storage,read_write>state:array<f32>;
@group(0)@binding(3)var<storage,read_write>scalars:array<f32>;
const INVALID=0xffffffffu;
struct Params { frame:vec4f, dimensions:vec4u }
const p=Params(vec4f(0.0333333,1.0,1.0,1.0),vec4u(${dimensions.map(n=>n+"u").join(",")},0u));
${array("centers","vec3f",cells.map(c=>vector(c.center)))}
${array("widths","vec3f",cells.map(c=>vector(c.widths)))}
${array("phis","f32",cells.map(c=>number(c.phi)))}
${array("volumes","f32",cells.map(c=>number(c.volume)))}
${array("rowCenters","vec3f",faces.map(f=>vector(f.center)))}
${array("axes","u32",faces.map(f=>`${f.axis}u`))}
${array("apertures","f32",faces.map(f=>number(f.open)))}
${array("solids","f32",faces.map(f=>number(f.solid)))}
${array("weights","f32",faces.map(f=>number(f.weight)))}
${array("prescribed","f32",faces.map(f=>number(f.prescribed?1:0)))}
${array("rowRanges","vec2u",ranges.map(v=>`vec2u(${v.map(n=>n+"u").join(",")})`))}
${array("cellRanges","vec2u",cellRanges.map(v=>`vec2u(${v.map(n=>n+"u").join(",")})`))}
${array("termCells","u32",terms.map(t=>`${t.cell}u`))}
${array("coefficients","f32",terms.map(t=>number(t.coefficient)))}
${array("incidenceTerms","u32",incidenceTerms.map(i=>`${i}u`))}
${array("incidenceRows","u32",incidenceTerms.map(i=>`${incidenceRows[i]}u`))}
fn isFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn cnxAcceptedCellCount()->u32{return ${n}u;}
fn cnxAcceptedRowCount()->u32{return ${r}u;}
fn cnxAcceptedCellInvocation(i:u32)->u32{return select(INVALID,i,i<${n}u);}
fn cnxAcceptedRowInvocation(i:u32)->u32{return select(INVALID,i,i<${r}u);}
fn cnxStableRowUnchecked(i:u32)->u32{return i;}
fn cnxRowTermRangeByOrdinalUnchecked(i:u32)->vec2u{return rowRanges[i];}
fn cnxRowTermCellUnchecked(t:u32)->u32{return termCells[t];}
fn cnxRowTermCoefficientUnchecked(t:u32)->f32{return coefficients[t];}
fn cnxCellIncidenceRangeUnchecked(c:u32)->vec2u{if(state[${r+3*samples.length}u]>0.0){return vec2u(0u);}return cellRanges[c];}
fn cnxIncidenceRowOrdinalUnchecked(t:u32)->u32{return incidenceRows[t];}
fn cnxIncidenceOwnCoefficientUnchecked(t:u32)->f32{return coefficients[incidenceTerms[t]];}
fn cnxRowPackedMetadataByOrdinal(r:u32)->u32{return axes[r];}
fn cellWidths(c:u32)->vec3f{return select(widths[c],vec3f(99.0),state[${r+3*samples.length}u]>0.0);}
fn cellCenter(c:u32)->vec3f{return centers[c]+vec3f(100.0*state[${r+3*samples.length}u]);}
fn cm12WorldFineLower()->vec3f{return ${vector(origin)};}
fn cm12WorldFineUpper()->vec3f{return ${vector(upper)};}
fn cellOpenVolume(c:u32)->f32{return volumes[c];}
fn cm12ExtendedCellSelected(c:u32)->bool{_=c;return true;}
struct Sample {phi:f32,valid:bool}
fn lsvCellSample(c:u32)->Sample{return Sample(phis[c],true);}
fn rowKind(r:u32)->u32{return select(0u,3u,rowRanges[r].y-rowRanges[r].x==1u);}
fn rowOpenFraction(r:u32)->f32{return apertures[r];}
fn rowSolidVelocity(r:u32)->f32{return solids[r];}
fn rowStaticDualWeight(r:u32)->f32{return weights[r];}
fn rowCenter(row:u32)->vec3f{return rowCenters[row]+vec3f(100.0*state[${r+3*samples.length}u]);}
fn rowSeparatingFromClosedWorld(r:u32)->bool{_=r;return false;}
fn sparseCM12InflowFaceCoverage(r:u32)->f32{return prescribed[r];}
fn destinationFaceVelocity()->u32{return 0u;}
fn cm12EffectiveTransportVelocity(c:u32)->vec4f{return partials[c];}
fn cm12ClampToResidentWorld(q:vec3f,margin:vec3f)->vec3f{return clamp(q,${vector(origin)}+margin,${vector(upper)}-margin);}
struct Owner {cell:u32}
fn cm12TransportOwnerAtFine(q:vec3i,direct:bool)->Owner{_=direct;
  for(var c=0u;c<${n}u;c+=1u){if(all(vec3f(q)>=centers[c]-.5*widths[c])&&all(vec3f(q)<centers[c]+.5*widths[c])){return Owner(c);}}
  return Owner(INVALID);}
${createAirExtensionWGSL(n,r)}
${createMomentumSnapshotWGSL(n,r,terms.length)}
${array("samplePoints","vec3f",samples.map(vector))}
@compute @workgroup_size(1)fn retireSnapshotSource(){state[${r+3*samples.length}u]=1.0;}
@compute @workgroup_size(64)fn sampleVelocity(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=${samples.length}u){return;}let owner=cm12TransportOwnerAtFine(vec3i(floor(cm12ClampToResidentWorld(samplePoints[gid.x],vec3f(1e-4)))),true).cell;
  let v=${options.snapshot ? "momentumSnapshotSample(samplePoints[gid.x])" : "airSampleVelocity(samplePoints[gid.x],cellWidths(owner),true)"};
  for(var a=0u;a<3u;a+=1u){state[${r}u+3u*gid.x+a]=v[a];}}
`;
  const module=device.createShaderModule({code});const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(messages.length)throw new Error(messages.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const bindings=device.createBindGroupLayout({entries:[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[bindings]});
  const pipelines=new Map<string,GPUComputePipeline>();
  for(const entryPoint of [...AIR_EXTENSION_ENTRY_POINTS,...MOMENTUM_SNAPSHOT_ENTRY_POINTS,"retireSnapshotSource","sampleVelocity"]){pipelines.set(entryPoint,
    await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}}));}
  const buffers=[layout.byteLength,4*Math.max(4*n+1,snapshotLayout.hashCapacity+1),4*Math.max(1,r+3*samples.length+1),snapshotLayout.byteLength].map(size=>device.createBuffer({size,
    usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}));
  const initial=new Float32Array(layout.byteLength/4);cells.forEach((c,i)=>initial.set([...c.velocity,1],4*i));
  device.queue.writeBuffer(buffers[0]!,0,initial);device.queue.writeBuffer(buffers[2]!,0,new Float32Array(faces.map(f=>f.velocity)));
  const group=device.createBindGroup({layout:bindings,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  const readback=device.createBuffer({size:layout.byteLength+buffers[2]!.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{
    const encoder=device.createCommandEncoder();let pass=encoder.beginComputePass();pass.setBindGroup(0,group);
    let dispatches=0;
    const dispatch=(name:string,count=1)=>{pass.setPipeline(pipelines.get(name)!);pass.dispatchWorkgroups(count);dispatches++;};
    const nc=Math.ceil(n/64),nr=Math.ceil(r/64);
    dispatch("airBegin");dispatch("airClassifyCells",nc);dispatch("airSeedFaces",nr);
    for(let depth=0;depth<AIR_EXTENSION_SWEEPS;depth++)dispatch(depth%2===0?"airExtendFacesA":"airExtendFacesB",nr);
    dispatch("airPrepareRows",nr);
    dispatch("airConnect",nr);dispatch("airAssemble",nc);dispatch("airInitialize",nc);dispatch("airReduceInitial");
    for(let i=0;i<(options.iterations??AIR_EXTENSION_ITERATIONS);i++){
      dispatch("airApply",nc);dispatch("airReduceAlpha");dispatch("airUpdate",nc);dispatch("airReduceBeta");dispatch("airDirection",nc);}
    dispatch("airMeasure",nc);dispatch("airReduceFinal");if(options.project!==false)dispatch("airCorrect",nr);
    if(options.snapshot){
      pass.end();encoder.clearBuffer(buffers[1]!,0,4*(snapshotLayout.hashCapacity+1));
      pass=encoder.beginComputePass();pass.setBindGroup(0,group);
      dispatch("momentumSnapshotBegin");dispatch("momentumSnapshotCells",nc);dispatch("momentumSnapshotFaces",nr);dispatch("momentumSnapshotSeal");
      pass.end();encoder.copyBufferToBuffer(buffers[1]!,0,buffers[3]!,4*snapshotLayout.hashBase,4*snapshotLayout.hashCapacity);
      // Retire the source face bank: momentum must be entirely self-contained.
      encoder.clearBuffer(buffers[0]!);encoder.clearBuffer(buffers[2]!);
      pass=encoder.beginComputePass();pass.setBindGroup(0,group);dispatch("retireSnapshotSource");
    }
    if(samples.length)dispatch("sampleVelocity",Math.ceil(samples.length/64));
    pass.end();encoder.copyBufferToBuffer(buffers[0]!,0,readback,0,layout.byteLength);
    encoder.copyBufferToBuffer(buffers[2]!,0,readback,layout.byteLength,buffers[2]!.size);
    const start=performance.now();device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const elapsedMs=performance.now()-start;const output=new Float32Array(readback.getMappedRange()).slice();
    return {receipt:decodeAirExtensionReceipt(output.subarray(4*layout.header,4*layout.header+16)),
      corrected:faces.map((_,r)=>output[4*(layout.rowBase+r)]!),
      samples:Array.from(output.subarray(layout.byteLength/4+r,layout.byteLength/4+r+3*samples.length)),elapsedMs,dispatches,
      bytes:buffers.reduce((s,b)=>s+b.size,0),output,layout};
  } finally {if(readback.mapState==="mapped")readback.unmap();readback.destroy();buffers.forEach(b=>b.destroy());}
}
