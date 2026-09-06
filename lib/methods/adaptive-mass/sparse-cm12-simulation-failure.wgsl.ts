import { CM12_FAILURE_WORDS, cm12FailureKernelId } from "./sparse-cm12-simulation-failure";

export const cm12SimulationFailureWGSL = /* wgsl */ `
var<private>cm12FailureKernel:u32;
fn cm12FailureBase()->u32{return arrayLength(&topologyArena)-${CM12_FAILURE_WORDS}u;}
fn cm12RecordFailure(code:u32,owner:u32,operands:vec4u){
  let base=cm12FailureBase();
  // atomicAdd cannot fail spuriously. Only the first reporter owns the record.
  if(atomicAdd(&topologyArena[base],1u)!=0u){return;}
  atomicStore(&topologyArena[base+1u],code);
  atomicStore(&topologyArena[base+2u],cm12FailureKernel);
  atomicStore(&topologyArena[base+3u],atomicLoad(&activity[0]));
  atomicStore(&topologyArena[base+4u],cm12FCCandidateGeneration());
  atomicStore(&topologyArena[base+5u],owner);
  for(var i=0u;i<4u;i+=1u){atomicStore(&topologyArena[base+6u+i],operands[i]);}
}
fn cm12ConservedValueValid(value:f32)->bool{
  return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u&&value>=0.0;
}
`;

/** The GPU copies the sticky fault count to uniform parameters between stages.
 * Uniform entry guards keep existing kernel barriers legal without extra barriers.
 */
export function guardCM12SimulationDispatches(source: string): string {
  return source.replace(/(@compute\s+@workgroup_size\([^)]*\)\s*fn\s+(\w+)\s*\()([\s\S]*?)(\)\s*\{)/g,
    (_match, prefix: string, name: string, parameters: string, suffix: string) =>
      `${prefix}${parameters}${suffix}\n  cm12FailureKernel=${cm12FailureKernelId(name)}u;\n  if(p.failure.x!=0u){return;}\n`);
}
