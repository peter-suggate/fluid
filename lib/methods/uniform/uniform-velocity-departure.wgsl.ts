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

/** The same characteristic in RK2 midpoint segments (momentumTrace
 * "substep"): each segment travels at most `reach` lattice cells at its start
 * velocity (the face's own cell size), the last takes what remains of dt, and
 * the time step is unchanged. Past `segments` segments the trace is cut (the
 * last takes the remaining time) and `capped` runs: so long a travel is past
 * the residency closure's reach, which raises its own fatal on a live pool,
 * so the count is a diagnostic that should stay zero. One sampler call site. */
export function uniformVelocitySegmentedDepartureWGSL(sample: (at: string) => string, clamp: (p: string) => string, reach: string, segments: string, capped: string): string {
  return /* wgsl */ `
  // A segment's start sample, then its midpoint, at one call site.
  if(abs(dt)<=1e-7){return position;}
  var at=position;var p=position;var left=abs(dt);let sign=select(-1.0,1.0,dt>=0.0);
  var span=0.0;var count=0u;var middle=false;
  loop{
   let cells=${sample("at")}/h;
   if(!middle){
    let speed=max(abs(cells.x),max(abs(cells.y),abs(cells.z)));count++;
    span=min(left,${reach}/max(speed,1e-30));
    if(count>=${segments}&&span<left){span=left;${capped}}
    at=${clamp("p-0.5*sign*span*cells")};middle=true;
   }else{
    p=${clamp("p-sign*span*cells")};left=left-span;at=p;middle=false;
    if(left<=1e-7*abs(dt)){return p;}
   }
  }`;
}
