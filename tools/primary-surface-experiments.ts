import assert from "node:assert/strict";
import { meshVolumeWGSL, meshVolumeUniformMaterialWGSL } from "../lib/core/imported-assets/mesh-volume.wgsl";
import { nativeLodStudy } from "./benchmark-native-lod";
import { meshAssetVolume } from "../lib/core/imported-assets/asset-library";
import { sampleMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { nativeIndexVariant, nativeIndexRangePatchWGSL } from "./native-index-experiments";
import { SVO_DRY_VISIBILITY_FLAGS } from "../lib/svo/features/shading/program";
import { SVO_SURFACE_MESH_STATE, svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { primaryNormalPartition } from "./primary-normal-partition";

interface Experiment {
  name: string;
  pipeline?: string;
  transform?(code: string): string;
  omitIdentity?: boolean;
  cachedNormals?: boolean;
  indexStrategy?: "zero-error" | "appearance";
  selector?: boolean;
  splitNormals?: number;
  optionalTransform?: boolean;
  partitionNormals?: boolean;
  integratedPartition?: boolean;
}
const name = "dryFieldProgramBlockVolume", load = "drySceneWord";
const volume = meshVolumeWGSL(name, load).replace(meshVolumeUniformMaterialWGSL,"");
const tetra = volume.slice(0, volume.indexOf(`fn ${name}(base`)) + volume.slice(volume.indexOf(`fn ${name}(base`), volume.indexOf("  let c=cell&", volume.indexOf(`fn ${name}(base`))) + `
  let c=cell&vec3u(7);let f=q-vec3f(cell);
  var a=0u;var bAxis=1u;var cAxis=2u;
  if(f[a]<f[bAxis]){let t=a;a=bAxis;bAxis=t;}
  if(f[bAxis]<f[cAxis]){let t=bAxis;bAxis=cAxis;cAxis=t;}
  if(f[a]<f[bAxis]){let t=a;a=bAxis;bAxis=t;}
  let corners=array<u32,4>(0u,1u<<a,(1u<<a)|(1u<<bAxis),7u);
  let weights=array<f32,4>(1.0-f[a],f[a]-f[bAxis],f[bAxis]-f[cAxis],f[cAxis]);
  var normal=vec3f(0);var ids:array<u32,4>;
  for(var i=0u;i<4u;i++){
    let corner=corners[i];let n=c+vec3u(corner&1u,(corner>>1u)&1u,corner>>2u);
    let identity=${load}(base+relative+2u*(n.x+9u*(n.y+9u*n.z))+1u);
    ids[i]=identity>>16u;if(ids[i]!=0u){normal+=weights[i]*${name}Normal(identity);}
  }
  var best=-1.0;
  ${Array.from({length:4},(_,i)=>`if(ids[${i}]!=0u){var vote=0.0;${Array.from({length:4},(_,j)=>`if(ids[${j}]==ids[${i}]){vote+=weights[${j}];}`).join("")}if(vote>best){best=vote;result.material=ids[${i}];}}`).join("\n")}
  if(dot(normal,normal)>1e-12){result.normal=normalize(normal);}return result;
}
`;

function cachedNormals(code:string,adaptive:boolean):string {
  return code.replace("  @location(2) @interpolate(flat) primitive:u32,", "  @location(2) @interpolate(flat) primitive:u32,\n  @location(3) appearance:vec4f,")
    .replace("@location(1) primitive:u32)->ImportedSurfaceVertex", "@location(1) primitive:u32,@location(2) appearance:vec4f)->ImportedSurfaceVertex")
    .replace("ImportedSurfaceVertex(clip,local,world,primitive)","ImportedSurfaceVertex(clip,local,world,primitive,appearance)")
    .replace("let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,input.local);", `
  var surface:${name}Value;
  ${adaptive ? "let footprint=max(length(dpdx(input.local)),length(dpdy(input.local)));\n  let useCached=input.appearance.w<=footprint*1.5&&dot(input.appearance.xyz,input.appearance.xyz)>.25;" : "let useCached=dot(input.appearance.xyz,input.appearance.xyz)>.25;"}
  if(useCached){
    let original=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,input.local);
    surface=${name}Value(0.0,normalize(input.appearance.xyz),original.material);
  }else{surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,input.local);}`);
}

function adaptiveVoxelNormals(code:string,threshold:number):string {
  return code.replace(/  var boundaryNormal=vec3f\(0\);[\s\S]*?  if\(dot\(boundaryNormal,boundaryNormal\)>1e-12\)\{normal=normalize\(boundaryNormal\);receiverNormal=normal;\}/,block=>`
  let footprint=max(min(length(dpdx(input.world)),length(dpdy(input.world))),1e-8);
  let pixelsPerCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))/footprint;
  let sourceWeight=smoothstep(${threshold.toFixed(1)},${(threshold*2).toFixed(1)},pixelsPerCell);
  if(sourceWeight>0.0){
${block.replace("normal=normalize(boundaryNormal);receiverNormal=normal;","normal=normalize(mix(normal,normalize(boundaryNormal),sourceWeight));receiverNormal=normalize(mix(receiverNormal,normalize(boundaryNormal),sourceWeight));")}
  }`);
}

function splitVoxelNormals(code:string,threshold:number,far:boolean,partitioned=false):string {
  if(partitioned&&far)return code.replace(/  var boundaryNormal=vec3f\(0\);[\s\S]*?  var shading=dryShadingNormal/,"  var shading=dryShadingNormal");
  const footprint=`let camera=dryRasterPrimaryCamera();
  let halfExtent=vec3f(meshQuadExtent(quad))*dry.mapping.cellSize*.5;
  let centre=dry.mapping.worldOrigin+vec3f(meshQuadOrigin(quad))*dry.mapping.cellSize+halfExtent;
  let depth=max(dot(centre-camera[0],camera[1])-dot(halfExtent,abs(camera[1])),DRY_REVERSED_Z_NEAR_M);
  let cellSize=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))*f32(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
  let projectedCell=cellSize*uniforms.viewport.y/(2.0*cameraTanHalfFov()*depth);
  ${partitioned?"":`if(${far?"projectedCell>":"projectedCell<="}${threshold.toFixed(1)}){position=vec4f(0,0,0,1);}`}`;
  code=code.replace("  var cellPixels=0.0;",`${footprint}\n  var cellPixels=projectedCell;`);
  if(far)return code.replace(/  var boundaryNormal=vec3f\(0\);[\s\S]*?  var shading=dryShadingNormal/,"  var shading=dryShadingNormal");
  return adaptiveVoxelNormals(code,threshold)
    .replace(/  let footprint=max\(min\(length\(dpdx\(input.world\)\),length\(dpdy\(input.world\)\)\),1e-8\);\n  let pixelsPerCell=.*?;/,"  let pixelsPerCell=input.cellPixels;");
}

export const primarySurfaceExperiments: Experiment[] = [
  {name:"lean-attachments",omitIdentity:true},
  {name:"homogeneous-material",pipeline:"Cached native asset surfaces",transform:code=>code.replace(volume,volume.replace("var best=-1.0;",`${meshVolumeUniformMaterialWGSL}var best=-1.0;`))},
  {name:"exact-combined",omitIdentity:true,optionalTransform:true,transform:code=>code.replace(volume,volume.replace("var best=-1.0;",`${meshVolumeUniformMaterialWGSL}var best=-1.0;`))},
  {name:"raw-oct-normals",pipeline:"Cached native asset surfaces",transform:code=>code.replace(volume,volume.replace("  return normalize(n);","  return n;"))},
  {name:"tetra-native",pipeline:"Cached native asset surfaces",transform:code=>code.replace(volume,tetra)},
  {name:"voxel-vertex-normals",pipeline:"Opaque voxel surface triangles",transform:code=>code.replace(/  var boundaryNormal=vec3f\(0\);[\s\S]*?  var shading=dryShadingNormal/,"  var shading=dryShadingNormal")},
  {name:"native-sampling-ceiling",pipeline:"Cached native asset surfaces",transform:code=>code.replace("let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,input.local);",`let surface=${name}Value(0.0,vec3f(0,1,0),0u);`)},
  {name:"native-vertex-normals",pipeline:"Cached native asset surfaces",cachedNormals:true,transform:code=>cachedNormals(code,false)},
  {name:"native-adaptive-normals",pipeline:"Cached native asset surfaces",cachedNormals:true,transform:code=>cachedNormals(code,true)},
  {name:"native-zero-error",indexStrategy:"zero-error"},
  {name:"native-appearance-lod",indexStrategy:"appearance"},
  {name:"native-box-depth",selector:true,transform:code=>code.replace("let depth=max(dot(centre-camera[0],camera[1])-radius,DRY_REVERSED_Z_NEAR_M);",`let localForward=svoQuaternionRotate(vec4f(-record.orientation.xyz,record.orientation.w),camera[1]);
  let extent=svoPrimitiveLocalExtent_m(svoPrimitiveKind(record),svoPrimitiveDimensions_m(record));
  let depth=max(z-min(radius,dot(abs(localForward),extent)),DRY_REVERSED_Z_NEAR_M);`)},
  ...[1,2,4].map(threshold=>({name:`voxel-adaptive-${threshold}`,pipeline:"Opaque voxel surface triangles",transform:(code:string)=>adaptiveVoxelNormals(code,threshold)})),
  ...[2,4].map(threshold=>({name:`voxel-split-${threshold}`,pipeline:"Opaque voxel surface triangles",splitNormals:threshold,transform:(code:string)=>splitVoxelNormals(code,threshold,false)})),
  ...[2,4].map(threshold=>({name:`voxel-partition-${threshold}`,pipeline:"Opaque voxel surface triangles",splitNormals:threshold,partitionNormals:true,transform:(code:string)=>splitVoxelNormals(code,threshold,false,true)})),
  ...[2,4].map(threshold=>({name:`voxel-partition-hard-${threshold}`,pipeline:"Opaque voxel surface triangles",splitNormals:threshold,partitionNormals:true,transform:(code:string)=>code+"\n// Full source normals in the near partition.\n"})),
  {name:"voxel-integrated-partition",pipeline:"Opaque voxel surface triangles",integratedPartition:true,transform:(code:string)=>{
    const group=/@group\((\d+)\) @binding\(30\) var<storage,read_write> meshState/.exec(code);
    if(!group)return code;
    const args=[Number(group[1]),SVO_DRY_VISIBILITY_FLAGS.flatVoxelNormals,true,true,false,true,code.includes("enable subgroups;")] as const;
    const original=svoSurfaceMeshWGSL(...args);
    assert.ok(code.includes(original),"Integrated normal experiment requires the production culling configuration");
    return code.replace(original,svoSurfaceMeshWGSL(...args,true));
  }},
  {name:"voxel-boundary-gradient",pipeline:"Opaque voxel surface triangles",transform:(code:string)=>code.replace("meshBoundaryNormal(input.world,boundary)","boundary.xyz")},
  {name:"voxel-source-specialized",pipeline:"Opaque voxel surface triangles"},
];

/** Isolated renderer experiments; every draw and resource is production-owned.
 * Variants replay immutable bundles, changing only named shaders/attachments. */
export async function primarySurfaceStudy(device: GPUDevice, requested: string[], sources:{terrain:boolean;boundary:boolean}) {
  const base=nativeLodStudy(device,[.25]);
  const experiments=requested.map(name=>{const experiment=primarySurfaceExperiments.find(e=>e.name===name);assert.ok(experiment,`Unknown primary experiment ${name}`);
    if(name!=="voxel-source-specialized")return experiment;
    return {...experiment,transform:(code:string)=>{
      if(!sources.terrain)code=code.replace("let terrain=meshAuthoredTerrainNormal(world,material,face);","let terrain=vec3f(0);");
      if(!sources.boundary)code=code.replaceAll(`meshHeader[${SVO_SURFACE_MESH_STATE.hostSolidBoundary}u]`,`0u`);
      return code;
    }};
  });
  const variants:Experiment[]=[{name:"reference"},...experiments];
  const partition=primaryNormalPartition(device,experiments.filter(e=>e.partitionNormals).map(e=>e.splitNormals!));
  const modules=new WeakMap<GPUShaderModule,Map<string,GPUShaderModule>>();
  const pipelines=new WeakMap<GPURenderPipeline,GPURenderPipeline[]>();
  const computePipelines=new WeakMap<GPUComputePipeline,GPUComputePipeline[]>();
  const farPipelines=new WeakMap<GPURenderPipeline,Map<string,GPURenderPipeline>>();
  const bundles=new WeakMap<GPURenderBundle,GPURenderBundle[]>();
  const appearances=new WeakMap<GPUBuffer,GPUBuffer>();
  const indices=new WeakMap<GPUBuffer,Map<string,{buffer:GPUBuffer;levels:ReturnType<typeof nativeIndexVariant>["levels"]}>>();
  const indexGroups=new Map<string,GPUBindGroup>();
  const rangeBuffers:GPUBuffer[]=[];
  const patch=experiments.some(e=>e.indexStrategy)?await device.createComputePipelineAsync({label:"Experimental native LOD ranges",layout:"auto",compute:{module:device.createShaderModule({code:nativeIndexRangePatchWGSL}),entryPoint:"main"}}):undefined;
  let selected=0;
  let identityOptional=false;
  const wrapped=new Proxy(base.device,{
    get(target,property){
      if(property==="createPipelineLayout")return (descriptor:GPUPipelineLayoutDescriptor)=>{const layout=target.createPipelineLayout(descriptor);partition.layout(layout,descriptor);return layout;};
      if(property==="createBindGroupLayout")return (descriptor:GPUBindGroupLayoutDescriptor)=>{const layout=target.createBindGroupLayout(descriptor);partition.groupLayout(layout,descriptor);return layout;};
      if(property==="createBindGroup")return (descriptor:GPUBindGroupDescriptor)=>{const group=target.createBindGroup(descriptor);partition.bindGroup(group,descriptor);return group;};
      if(property==="createBuffer"&&experiments.some(e=>e.cachedNormals||e.indexStrategy))return (descriptor:GPUBufferDescriptor)=>{
        const original=target.createBuffer(descriptor);
        if(experiments.some(e=>e.cachedNormals)&&descriptor.label?.endsWith(" native surface vertices")&&descriptor.mappedAtCreation){
          const getMappedRange=original.getMappedRange.bind(original),unmap=original.unmap.bind(original);
          let mapped:ArrayBuffer|undefined;
          original.getMappedRange=(offset,size)=>{assert.equal(offset??0,0);mapped=getMappedRange(offset,size);return mapped;};
          original.unmap=()=>{
            const points=new Float32Array(mapped??getMappedRange()),asset=descriptor.label!.slice(0,-" native surface vertices".length),volume=meshAssetVolume(asset);
            const data=new Float32Array(points.length/3*4),h=volume.header.cellSize;
            for(let i=0;i<points.length;i+=3){
              const p=volume.header.origin.map((v,a)=>v+volume.header.dimensions[a]!*h/2+points[i+a]!) as [number,number,number];
              data.set([...sampleMeshVolume(volume,p).shadingNormal,h],i/3*4);
            }
            const buffer=target.createBuffer({label:`${asset} cached native appearance`,size:Math.max(4,data.byteLength),usage:GPUBufferUsage.VERTEX,mappedAtCreation:true});
            new Float32Array(buffer.getMappedRange()).set(data);buffer.unmap();appearances.set(original,buffer);
            original.getMappedRange=getMappedRange;original.unmap=unmap;unmap();
            const destroy=original.destroy.bind(original);original.destroy=()=>{buffer.destroy();destroy();};
          };
        }
        if(experiments.some(e=>e.indexStrategy)&&descriptor.label?.endsWith(" native surface indices")&&descriptor.mappedAtCreation){
          const unmap=original.unmap.bind(original);
          original.unmap=()=>{
            unmap();original.unmap=unmap;
            const alternatives=new Map<string,{buffer:GPUBuffer;levels:ReturnType<typeof nativeIndexVariant>["levels"]}>();
            const volume=meshAssetVolume(descriptor.label!.slice(0,-" native surface indices".length));
            for(const experiment of experiments)if(experiment.indexStrategy){
              const result=nativeIndexVariant(volume,experiment.indexStrategy);
              const buffer=target.createBuffer({...descriptor,label:`${descriptor.label} ${experiment.name}`,size:Math.max(4,result.indices.byteLength),mappedAtCreation:true});
              new Uint32Array(buffer.getMappedRange()).set(result.indices);buffer.unmap();alternatives.set(experiment.name,{buffer,levels:result.levels});
            }
            indices.set(original,alternatives);
            const destroy=original.destroy.bind(original);original.destroy=()=>{for(const a of alternatives.values())a.buffer.destroy();destroy();};
          };
        }
        return original;
      };
      if(property==="createShaderModule")return (descriptor:GPUShaderModuleDescriptor)=>{
        // Keep the campaign's pre-optimization reference after promoting the
        // uniform-material fast path into production.
        descriptor={...descriptor,code:descriptor.code.replaceAll(meshVolumeUniformMaterialWGSL,"")};
        const original=target.createShaderModule(descriptor), alternatives=new Map<string,GPUShaderModule>();
        partition.shader(original,descriptor.code);
        for(const experiment of experiments){const code=experiment.transform?.(descriptor.code)??descriptor.code;if(code!==descriptor.code)alternatives.set(experiment.name,target.createShaderModule({...descriptor,code,label:`${descriptor.label} ${experiment.name}`}));
          if(experiment.splitNormals&&code!==descriptor.code)alternatives.set(`${experiment.name}-far`,target.createShaderModule({...descriptor,code:splitVoxelNormals(descriptor.code,experiment.splitNormals,true,experiment.partitionNormals),label:`${descriptor.label} ${experiment.name} far`}));
        }
        modules.set(original,alternatives);return original;
      };
      if(property==="createRenderPipelineAsync")return async(descriptor:GPURenderPipelineDescriptor)=>{
        const original=await target.createRenderPipelineAsync(descriptor);
        const primary=descriptor.fragment && Array.from(descriptor.fragment.targets).length===4 && descriptor.depthStencil;
        if(!primary)return original;
        const alternatives=[original];
        for(const experiment of experiments){
          if(experiment.indexStrategy||experiment.selector){alternatives.push(original);continue;}
          if(experiment.name==="voxel-source-specialized"&&sources.terrain&&sources.boundary){alternatives.push(original);continue;}
          if(experiment.pipeline&&experiment.pipeline!==descriptor.label){alternatives.push(original);continue;}
          const fragment=descriptor.fragment!;
          const module=modules.get(fragment.module)?.get(experiment.name);
          if(experiment.transform&&(!experiment.optionalTransform||descriptor.label==="Cached native asset surfaces"))assert.ok(module,`Experiment ${experiment.name} did not match ${descriptor.label}`);
          const targets=experiment.omitIdentity?Array.from(fragment.targets).map((target,i)=>i<2?null:target):fragment.targets;
          alternatives.push(await target.createRenderPipelineAsync({...descriptor,label:`${descriptor.label} ${experiment.name}`,
            vertex:{...descriptor.vertex,module:modules.get(descriptor.vertex.module)?.get(experiment.name)??descriptor.vertex.module,
              ...(experiment.cachedNormals?{buffers:[...Array.from(descriptor.vertex.buffers??[]),{arrayStride:16,attributes:[{shaderLocation:2,offset:0,format:"float32x4" as GPUVertexFormat}]}]}:{})},
            fragment:{...fragment,module:module??fragment.module,targets}}));
          if(experiment.splitNormals||experiment.integratedPartition){
            const far=experiment.integratedPartition?module:modules.get(fragment.module)?.get(`${experiment.name}-far`);assert.ok(far);
            const map=farPipelines.get(original)??new Map<string,GPURenderPipeline>();
            map.set(experiment.name,await target.createRenderPipelineAsync({...descriptor,label:`${descriptor.label} ${experiment.name} far`,vertex:{...descriptor.vertex,module:far,...(experiment.integratedPartition?{entryPoint:"surfaceMeshFarVertex"}:{})},fragment:{...fragment,module:far,...(experiment.integratedPartition?{entryPoint:"surfaceMeshFarFragment"}:{})}}));
            farPipelines.set(original,map);
          }
        }
        pipelines.set(original,alternatives);return original;
      };
      if(property==="createComputePipelineAsync")return async(descriptor:GPUComputePipelineDescriptor)=>{
        const original=await target.createComputePipelineAsync(descriptor);
        await partition.compile(original,descriptor);
        const native=descriptor.label==="Native asset screen-error selection";
        if(!native&&!descriptor.label?.startsWith("surfaceMesh"))return original;
        const alternatives=[original];
        for(const experiment of experiments){
          if(!(native?experiment.selector:experiment.integratedPartition)){alternatives.push(original);continue;}
          const module=modules.get(descriptor.compute.module)?.get(experiment.name);assert.ok(module,`Selector ${experiment.name} must match`);
          alternatives.push(await target.createComputePipelineAsync({...descriptor,compute:{...descriptor.compute,module}}));
        }
        computePipelines.set(original,alternatives);return original;
      };
      if(property==="createRenderBundleEncoder")return (descriptor:GPURenderBundleEncoderDescriptor)=>{
        const original=target.createRenderBundleEncoder(descriptor);
        if(descriptor.label!=="Cached primary draw commands")return original;
        const commands:{name:string;args:unknown[]}[]=[];
        return new Proxy(original,{get(bundle,property){
          if(property==="finish")return (finish?:GPURenderBundleDescriptor)=>{
            const result=bundle.finish(finish),alternatives=[result];
            for(let i=1;i<variants.length;i++){
              const experiment=variants[i]!;
              const replay=target.createRenderBundleEncoder({...descriptor,colorFormats:experiment.omitIdentity?Array.from(descriptor.colorFormats).map((format,j)=>j<2?null:format):descriptor.colorFormats});
              const ranges=experiment.indexStrategy?new Uint32Array(base.drawArgs!.size/20*32):undefined;
              let activeIndices:ReturnType<typeof nativeIndexVariant>["levels"]|undefined;
              let activePipeline:GPURenderPipeline|undefined;
              let originalDrawGroup:GPUBindGroup|undefined;
              for(const command of commands){
                if(command.name==="setPipeline")activePipeline=command.args[0] as GPURenderPipeline;
                let args=command.name==="setPipeline"?[pipelines.get(command.args[0] as GPURenderPipeline)?.[i]??command.args[0]]:command.args;
                if(experiment.partitionNormals&&activePipeline?.label==="Opaque voxel surface triangles"){
                  if(command.name==="setBindGroup"&&command.args[0]===1){
                    originalDrawGroup=command.args[1] as GPUBindGroup;
                    args=[1,partition.group(originalDrawGroup,experiment.splitNormals!,"near"),...command.args.slice(2)];
                  }
                  if(command.name==="drawIndirect")args=[partition.args(originalDrawGroup!,experiment.splitNormals!),0];
                }
                if(experiment.integratedPartition&&command.name==="drawIndirect"&&activePipeline?.label==="Opaque voxel surface triangles")args=[command.args[0],SVO_SURFACE_MESH_STATE.nearNormalDraw*4];
                if(experiment.indexStrategy&&command.name==="setIndexBuffer"){
                  const alternative=indices.get(command.args[0] as GPUBuffer)?.get(experiment.name);activeIndices=alternative?.levels;
                  if(alternative)args=[alternative.buffer,...command.args.slice(1)];
                }
                if(ranges&&activeIndices&&command.name==="drawIndexedIndirect"){
                  assert.equal(command.args[0],base.drawArgs);
                  const at=Number(command.args[1])/20*32;ranges[at]=activeIndices.length;
                  activeIndices.forEach((level,j)=>ranges.set([level.originalFirst,level.first,level.count],at+1+j*3));
                }
                Reflect.get(replay,command.name).apply(replay,args);
                if((experiment.splitNormals||experiment.integratedPartition)&&command.name==="drawIndirect"&&activePipeline?.label==="Opaque voxel surface triangles"){
                  replay.setPipeline(farPipelines.get(activePipeline)!.get(experiment.name)!);
                  if(experiment.partitionNormals)replay.setBindGroup(1,partition.group(originalDrawGroup!,experiment.splitNormals!,"far"));
                  replay.drawIndirect(args[0] as GPUBuffer,experiment.integratedPartition?SVO_SURFACE_MESH_STATE.farNormalDraw*4:experiment.partitionNormals?16:Number(args[1]));
                  if(experiment.partitionNormals)replay.setBindGroup(1,partition.group(originalDrawGroup!,experiment.splitNormals!,"near"));
                  replay.setPipeline(pipelines.get(activePipeline)![i]!);
                }
                if(experiment.cachedNormals&&command.name==="setVertexBuffer"&&command.args[0]===0){const appearance=appearances.get(command.args[1] as GPUBuffer);if(appearance)replay.setVertexBuffer(2,appearance);}
              }
              if(ranges){
                const buffer=device.createBuffer({size:ranges.byteLength,usage:GPUBufferUsage.STORAGE,mappedAtCreation:true});new Uint32Array(buffer.getMappedRange()).set(ranges);buffer.unmap();rangeBuffers.push(buffer);
                indexGroups.set(experiment.name,device.createBindGroup({layout:patch!.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:base.drawArgs!}},{binding:1,resource:{buffer}}]}));
              }
              alternatives.push(replay.finish());
            }
            bundles.set(result,alternatives);return result;
          };
          const value=Reflect.get(bundle,property,bundle);
          return typeof value==="function"?(...args:unknown[])=>{commands.push({name:String(property),args});return value.apply(bundle,args);}:value;
        }});
      };
      const value=Reflect.get(target,property,target);return typeof value==="function"?value.bind(target):value;
    },
  });
  return {device:wrapped,budgets:variants.map(e=>e.name),async census(){const experiment=variants[selected]!;return {...await base.census(),...await partition.totalCensus(),...(experiment.partitionNormals||experiment.integratedPartition?await partition.census(experiment.splitNormals):{})};},skipReason(index:number){return variants[index]!.omitIdentity&&!identityOptional?"Identity planes are required by this scene":undefined;},select(index:number){selected=index;},encode(encoder:GPUCommandEncoder){
    return new Proxy(encoder,{get(target,property){
      if(property==="beginComputePass")return (descriptor?:GPUComputePassDescriptor)=>{
        const experiment=variants[selected]!,partitioned=experiment.partitionNormals&&descriptor?.label==="Voxel mesh culling";
        if(partitioned)partition.reset(target,experiment.splitNormals!);
        const pass=target.beginComputePass(descriptor);
        return new Proxy(pass,{get(target,property){
          if(property==="setPipeline")return (pipeline:GPUComputePipeline)=>target.setPipeline(partitioned?partition.pipeline(pipeline,experiment.splitNormals!):computePipelines.get(pipeline)?.[selected]??pipeline);
          if(property==="setBindGroup"&&partitioned)return (index:number,group:GPUBindGroup)=>target.setBindGroup(index,partition.group(group,experiment.splitNormals!,"compute"));
          const value=Reflect.get(target,property,target);return typeof value==="function"?value.bind(target):value;
        }});
      };
      if(property==="beginRenderPass")return (descriptor:GPURenderPassDescriptor)=>{
        if(descriptor.label!=="Voxel surface mesh rasterization")return target.beginRenderPass(descriptor);
        const experiment=variants[selected]!;
        identityOptional=Array.from(descriptor.colorAttachments).slice(0,2).every(a=>a?.storeOp==="discard");
        if(experiment.indexStrategy){
          const group=indexGroups.get(experiment.name);assert.ok(group,"Experimental index ranges must be published");
          const pass=target.beginComputePass({label:"Experimental native LOD ranges"});pass.setPipeline(patch!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(base.drawArgs!.size/20/64));pass.end();
        }
        if(experiment.omitIdentity)assert.ok(Array.from(descriptor.colorAttachments).slice(0,2).every(a=>a?.storeOp==="discard"),"Lean attachment experiment requires unused identity planes");
        const pass=target.beginRenderPass(experiment.omitIdentity?{...descriptor,colorAttachments:Array.from(descriptor.colorAttachments).map((attachment,i)=>i<2?null:attachment)}:descriptor);
        return new Proxy(pass,{get(target,property){
          if(property==="executeBundles")return (input:Iterable<GPURenderBundle>)=>target.executeBundles(Array.from(input,bundle=>bundles.get(bundle)?.[selected]??bundle));
          const value=Reflect.get(target,property,target);return typeof value==="function"?value.bind(target):value;
        }});
      };
      const value=Reflect.get(target,property,target);return typeof value==="function"?value.bind(target):value;
    }});
  }};
}
