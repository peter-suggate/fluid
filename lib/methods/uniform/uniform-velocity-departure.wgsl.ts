/** Native CM11b velocity characteristic. Both samplers use lattice coordinates
 * and physical velocity; the clamp callback retains the open-top policy. */
export function uniformVelocityDepartureWGSL(sample: string, clamp: string, cellWidth?: string, iterationLimit = "32"): string {
  return /* wgsl */ `
  // Each substep's two samples (the start, then the midpoint) at one call
  // site: Metal inlines every call, so a sampler written twice compiles twice.
  var point=position;var remaining=abs(dt);let direction=select(-1.0,1.0,dt>=0.0);
  var at=position;var mid=false;var stepSeconds=0.0;var signedStep=0.0;
  for(var stage=0;stage<2*${iterationLimit};stage+=1){
    if(!mid&&remaining<=1e-7){break;}
    let sampled=${sample}(at);
    if(!mid){
      let first=sampled;let rate=max(abs(first.x)/h.x,max(abs(first.y)/h.y,abs(first.z)/h.z))${cellWidth ? `/(${cellWidth})` : ""};
      stepSeconds=min(remaining,1.5/max(rate,1e-6));signedStep=direction*stepSeconds;
      at=${clamp}(point-0.5*first*signedStep/h);mid=true;continue;
    }
    point=${clamp}(point-sampled*signedStep/h);remaining-=stepSeconds;at=point;mid=false;
  }
  return point;`;
}
