/** Shared RK2 characteristic in lattice coordinates. Collision handling, when
 * needed, follows this endpoint in the native caller. */
export function uniformMidpointTraceWGSL(sampleVelocity: string): string {
  return /* wgsl */ `
  let h=params.cellGravity.xyz;
  let mid=clamp(p-0.5*dt*${sampleVelocity}(p)/h,vec3f(0),vec3f(dims()));
  let end=clamp(p-dt*${sampleVelocity}(mid)/h,vec3f(0),vec3f(dims()));`;
}
