import {gpuCompilationManagerFor} from "../../lib/core/gpu-compilation-manager";

/** Test-only scoped compiler instrumentation. The solver's initialization
 * device owns detail storage by identity, so preserve that capability and
 * change only the selected surface pipeline constant. */
export async function withExactSolidSupport<T>(device:GPUDevice,enabled:boolean,run:()=>Promise<T>):Promise<T>{
 const manager=gpuCompilationManagerFor(device),modules=new WeakSet<GPUShaderModule>();
 let overridden=0;
 const create=manager.createShaderModule.bind(manager),compile=manager.compileComputePipeline.bind(manager);
 manager.createShaderModule=(descriptor)=>{
  const module=create(descriptor);
  if(/override\s+umExactSolidSupport\s*:/.test(descriptor.code))modules.add(module);
  return module;
 };
 manager.compileComputePipeline=(descriptor,options)=>{
  if(modules.has(descriptor.compute.module)){overridden++;descriptor={...descriptor,compute:{...descriptor.compute,constants:{...descriptor.compute.constants,umExactSolidSupport:Number(enabled),umExactLocalSolidTransport:Number(enabled)}}};}
  return compile(descriptor,options);
 };
 try{const result=await run();if(overridden===0)throw new Error("Exact-solid-support reference did not override any surface pipeline");return result;}finally{manager.createShaderModule=create;manager.compileComputePipeline=compile;}
}
