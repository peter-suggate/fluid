/** Native velocity characteristic: one RK2 midpoint step of the whole dt,
 * whatever its length (no substep or CFL limit). The sampler uses lattice
 * coordinates and physical velocity; the clamp callback retains the open-top
 * policy. */
export function uniformVelocityDepartureWGSL(sample: string, clamp: string): string {
  return /* wgsl */ `
  // The start sample, then the midpoint, at one call site: Metal inlines
  // every call, so a sampler written twice compiles twice.
  if(abs(dt)<=1e-7){return position;}
  var at=position;
  for(var stage=0u;stage<2u;stage++){
    let sampled=${sample}(at);
    at=${clamp}(position-select(0.5,1.0,stage==1u)*sampled*dt/h);
  }
  return at;`;
}
