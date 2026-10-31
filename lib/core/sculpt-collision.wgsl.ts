import { sculptSubstrateWGSL } from "../svo/features/scene-publication/svo-sculpt-substrate";

/** Last four words of the solver scratch: sculpt pointer/tag, then the existing
 * procedural boundary pointer/tag. Older and test buffers remain valid. */
export const SCULPT_COLLISION_MAGIC = 0x53434f4c;
export function sculptCollisionWGSL(buffer: string): string {
  return /* wgsl */ `
fn collisionSculptBase()->u32{
 let count=arrayLength(&${buffer});
 if(count<4u||${buffer}[count-3u]!=${SCULPT_COLLISION_MAGIC}u){return 0u;}
 return ${buffer}[count-4u];
}
fn collisionSculptWord(i:u32)->u32{return ${buffer}[collisionSculptBase()+i];}
struct SvoSculptCutSample { distance:f32,normal:vec3f }
${sculptSubstrateWGSL("collisionSculptWord", false)}
fn collisionSculptActive()->bool{return collisionSculptBase()!=0u&&collisionSculptWord(0u)!=0u;}
fn collisionSculptSample(world:vec3f)->SvoSculptCutSample{
 if(!collisionSculptActive()){return SvoSculptCutSample(1e20,vec3f(0));}
 return sculptFieldTrilinear(world);
}
fn collisionSculptNear(world:vec3f)->bool{
 if(!collisionSculptActive()){return false;}
 let page=vec3i(floor(((world-sculptFieldOrigin())/sculptFieldCell()-0.5)/8.0));
 return sculptFieldSlot(page)!=0xffffffffu;
}
`;
}
