import {inflowBoundaryWGSL} from "../../core/inflow-boundary";
import {uniformDropSourceWGSL,uniformExtrusionParamsWGSL,uniformSourcePhiWGSL} from "./uniform-source.wgsl";
export {UNIFORM_PARAMS_BYTES} from "./uniform-source.wgsl";

/**
 * ABI of the host's existing scene uniforms; borrowed without copies. Only the
 * source fields are named: the six vec4f between them and the extrusion tail
 * belong to the host's other stages. Bind UNIFORM_PARAMS_BYTES of it.
 */
export function uniformMixedSourceWGSL(binding:number):string{
 const code=/* wgsl */`
struct UMSourceParams {
 dimsDt:vec4f,cellGravity:vec4f,container:vec4f,physical:vec4f,boundary:vec4f,
 inflowPositionRadius:vec4f,inflowVelocityLength:vec4f,inflowTiming:vec4f,tuning:vec4f,drop:vec4f,dropExtent:vec4f,
 hostStages:array<vec4f,6>,${uniformExtrusionParamsWGSL}
}
@group(1) @binding(${binding}) var<uniform> umSourceParams:UMSourceParams;
fn inflowGridDims()->vec3i{return vec3i(UM_D);}
fn traceWorld(p:vec3f)->vec3f{return vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z)+p*params.cellGravity.xyz;}
${inflowBoundaryWGSL}
${uniformDropSourceWGSL}
${uniformSourcePhiWGSL}
`;
 return code.replace(/\bparams\b/g,"umSourceParams").replace(/\b(?:inflow[A-Z]\w*|isInflow\w*|applyInflow\w*|dropSource|dropDistance|dropOutlineDistance|uvSourcePhi|traceWorld)\b/g,name=>"umSource"+name);
}
