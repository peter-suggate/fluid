/** Shared f32 scratch backing. Extension, conservative transport and pressure
 * execute in that order; none of their temporary fields survives its stage.
 * Persistent/published textures remain ordinary textures. */
import { planUniformCM11aHierarchy } from "./pressure-plan";
import { rewritePressureTextureCalls } from "./uniform-pressure-pages";

export class UniformScratchArena {
  readonly buffer: GPUBuffer;
  readonly donorOffset: number;
  readonly edgeBytes: number;
  readonly donorBytes: number;
  readonly sharpenBaseWords: number;
  readonly conditioningBytes: number;
  private readonly offsets = new Map<string, number>();
  constructor(device: GPUDevice, readonly dims: readonly [number,number,number], edgeBytes: number, retainDiagnostics=false) {
    const tiles=dims.reduce((n,d)=>n*Math.ceil(d/4),1);
    this.sharpenBaseWords=Math.ceil((6*tiles+3)/4)*4;
    this.conditioningBytes=Math.ceil((this.sharpenBaseWords+8+tiles)/4)*16;
    const words = dims.reduce((n,d) => n*(d+2),1)*4;
    ["FIM values A","FIM values B","FIM distances A","FIM distances B"].forEach((name,i) =>
      this.offsets.set(`Uniform Sec. 3.3 ${name}`,i*words));
    // Resolve reads and writes only its own cell. With buffer storage the
    // canonical resolved values can safely overwrite the A input in place.
    this.offsets.set("Uniform Sec. 3.3 resolved FIM values",0);
    const fineCells=Math.ceil(dims.reduce((n,d)=>n*(d+2),1)/4)*4;
    // Finest V is dead after coefficient baking. The four solver scalars
    // below are first written by cycles, after that bake. Raw/continued phi
    // are also dead at this level, so they become backup/accepted pressure.
    const finest:Record<string,number>={"V A":0,"pressure B":0,"rhs B":fineCells,
      "residual A":2*fineCells,"p-min B":3*fineCells,"phi A":4*fineCells,
      "phi B":5*fineCells,"rhs A":6*fineCells,"p-min A":7*fineCells,"coefficients":8*fineCells};
    for(const [name,offset] of Object.entries(finest))this.offsets.set(`Uniform CM11a L0 ${name}`,offset);
    this.offsets.set("Uniform CM11a Full-Cycle p_tmp",4*fineCells);
    this.offsets.set("Uniform CM11a accepted pressure",5*fineCells);
    let pressureWords=12*fineCells;
    planUniformCM11aHierarchy(dims).levelDimensions.slice(1).forEach((d,index)=>{
      const cells=d.reduce((n,x)=>n*(x+2),1);
      for(const [name,count] of Object.entries({"pressure A":1,"pressure B":1,"rhs A":1,"rhs B":1,
        "phi A":1,"phi B":1,"V A":4,"residual A":1,"p-min A":1,"p-min B":1,"coefficients":4})){
        pressureWords=Math.ceil(pressureWords/4)*4;
        this.offsets.set(`Uniform CM11a L${index+1} ${name}`,pressureWords);pressureWords+=cells*count;
      }
    });
    const extensionEnd=words*4;
    this.offsets.set("Uniform Sec. 3.3 resolved FIM distances",2*words);
    this.offsets.set("Total surface volume corrected phi",0);
    if(retainDiagnostics)this.offsets.delete("Uniform Sec. 3.3 resolved FIM distances");
    this.edgeBytes=edgeBytes;
    this.donorOffset=Math.ceil(edgeBytes/256)*256;
    this.donorBytes=dims.reduce((n,d)=>n*d,1)*24;
    this.buffer=device.createBuffer({label:"Uniform shared stage scratch",size:Math.max(extensionEnd*4,pressureWords*4,this.donorOffset+this.donorBytes),
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  }
  offset(label:string):number|undefined{return this.offsets.get(label);}
  destroy():void{this.buffer.destroy();}
}

/** Keep the original f32 loads/stores and bounds behavior; only their backing
 * changes. w uses its high bit for scratch, low two bits for component count. */
export function uniformScratchAccessWGSL(name:string, metadata:string, type:string, atomic=false, components?:number):string {
  const uint=type.includes('uint')||type.includes('u32');
  const vector=uint?'vec4u':'vec4f';
  const raw=(i:number)=>atomic?`atomicLoad(&uniformScratch[at+${i}u])`:`uniformScratch[at+${i}u]`;
  const load=(i:number)=>uint?raw(i):`bitcast<f32>(${raw(i)})`;
  const store=(i:number)=>uint?`value[${i}]`:`bitcast<u32>(value[${i}])`;
  const address=`let fieldLayout=${metadata};let count=${components?`${components}u`:"select(1u,4u,(fieldLayout.w&3u)==3u)"};let at=(fieldLayout.w&0x7ffffffcu)+(u32(p.x)+fieldLayout.x*(u32(p.y)+fieldLayout.y*u32(p.z)))*count;`;
  const assignment=(i:number)=>atomic?`atomicStore(&uniformScratch[at+${i}u],${store(i)});`:`uniformScratch[at+${i}u]=${store(i)};`;
  const guard=`any(p<vec3i(0))||any(p>=vec3i(${metadata}.xyz))`;
  return type.startsWith('texture_storage')
    ? `if((${metadata}.w&0x80000000u)!=0u){if(${guard}){return;}${address}${assignment(0)}if(count==4u){${[1,2,3].map(assignment).join('')}}return;}`
    : `if((${metadata}.w&0x80000000u)!=0u){if(${guard}){return ${vector}(0);}${address}if(count==1u){return ${vector}(${load(0)},0,0,1);}return ${vector}(${[0,1,2,3].map(load).join(',')});}`;
}

export function uniformPressureScratchShader(source:string):string {
  const declarations=[...source.matchAll(/@group\(1\) @binding\((\d+)\) var (mg\w+): (texture[^;]+);/g)];
  const fields=new Map(declarations.map(m=>[m[2]!,{binding:Number(m[1]),type:m[3]!}]));
  let helpers='';
  for(const [name,{binding,type}] of fields){
    const metadata=`mg.fieldDims[${binding}]`;
    const readWrite=type.includes('read_write');
    if(!type.startsWith('texture_storage')||readWrite){
      const code=uniformScratchAccessWGSL(name,metadata,'texture_3d<f32>',false,[7,8,14,15].includes(binding)?4:1);
      helpers+=`fn ${name}Load(p:vec3i)->vec4f{${code}return textureLoad(${name},p${readWrite?'':',0'});}\n`;
    }
    if(type.startsWith('texture_storage')){
      const code=uniformScratchAccessWGSL(name,metadata,type,false,[7,8,14,15].includes(binding)?4:1);
      helpers+=`fn ${name}Store(p:vec3i,value:vec4f){${code}textureStore(${name},p,value);}\n`;
    }
  }
  let result=source.replace('  control: vec4u,','  control: vec4u,\n  fieldDims: array<vec4u,17>,');
  // read_write textures use a two-argument textureLoad. Normalize it for the
  // existing call rewriter before replacing the accesses with buffer helpers.
  result=result.replace(/textureLoad\(mgPressureRW,\s*([^,()]+)\)/g,'textureLoad(mgPressureRW,$1,0)');
  result=result.replaceAll('textureBarrier();','storageBarrier();textureBarrier();');
  return rewritePressureTextureCalls(result,fields)+helpers;
}

/** The same 40-byte stencil rows in the shared arena. Donor accumulation
 * binds a disjoint range as atomics; ordinary field accesses remain f32 loads. */
export function uniformVolumeScratchShader(source:string,sharpenBaseWords:number,conditioningBytes:number):string {
  // Once decoding finishes, no pass needs the exact limbs again until the
  // next clear. Store each rounded sum in its own low limb, in place.
  let result=source.replace("atomicStore(&sharpenDeposits[i],bitcast<i32>(uvDonorSum(i)))",
      "atomicStore(&rigidExchange[i],bitcast<i32>(uvDonorSum(i)))")
    .replace("if(atomicLoad(&sharpenDeposits[i])==0){uvEdges", "if(atomicLoad(&rigidExchange[i])==0){uvEdges")
    .replace("let sum=bitcast<f32>(atomicLoad(&sharpenDeposits[donor]));", "let sum=bitcast<f32>(atomicLoad(&rigidExchange[donor]));")
    .replace("fn uvCoarseBase()->u32{return cellCount();}","fn uvCoarseBase()->u32{return 0u;}")
    .replace("return cellCount()+4u*uvCoarseCount()+plane*uvCoarseCount();", "return 4u*uvCoarseCount()+plane*uvCoarseCount();")
    .replaceAll('2u*cellCount()+UV_SHARPEN',`${sharpenBaseWords}u+UV_SHARPEN`)
    .replace('fn uvBalanceBase()->u32{return 3u*cellCount();}',`fn uvBalanceBase()->u32{return ${conditioningBytes/4}u;}`)
    .replace(/@group\(0\) @binding\(33\) var<storage,read_write> uvEdges:array<UVEdges>;/,'');
  const weight=/uvEdges\[([^\]]+)\]\.weight\[([^\]]+)\]/g;
  result=result.replace(/uvEdges\[([^\]]+)\]\.weight\[([^\]]+)\](\/=|=(?!=))([^;]+);/g,
    (_m,i,k,op,value)=>`uvSetWeight(${i},${k},${op==='/='?`uvWeight(${i},${k})/(${value})`:value});`);
  result=result.replace(/uvEdges\[([^\]]+)\]\.base(\|=|&=|=(?!=))([^;]+);/g,
    (_m,i,op,value)=>`uvSetBase(${i},${op==='='?value:`uvBase(${i})${op[0]}(${value})`});`);
  result=result.replace(weight,(_m,i,k)=>`uvWeight(${i},${k})`)
    .replace(/uvEdges\[([^\]]+)\]\.base/g,(_m,i)=>`uvBase(${i})`);
  return result+`
fn uvBase(i:u32)->u32{return uniformScratch[10u*i];}
fn uvSetBase(i:u32,value:u32){uniformScratch[10u*i]=value;}
fn uvWeight(i:u32,k:u32)->f32{return bitcast<f32>(uniformScratch[10u*i+1u+k]);}
fn uvSetWeight(i:u32,k:u32,value:f32){uniformScratch[10u*i+1u+k]=bitcast<u32>(value);}
`;
}
