/** Native CM11b velocity characteristic. Both samplers use lattice coordinates
 * and physical velocity; the clamp callback retains the open-top policy. */
export function uniformVelocityDepartureWGSL(sample: string, clamp: string, cellWidth?: string, iterationLimit = "32"): string {
  return /* wgsl */ `
  var point=position;var remaining=abs(dt);let direction=select(-1.0,1.0,dt>=0.0);
  for(var step=0;step<${iterationLimit};step+=1){
    if(remaining<=1e-7){break;}
    let first=${sample}(point);let rate=max(abs(first.x)/h.x,max(abs(first.y)/h.y,abs(first.z)/h.z))${cellWidth ? `/(${cellWidth})` : ""};
    let stepSeconds=min(remaining,1.5/max(rate,1e-6));let signedStep=direction*stepSeconds;
    let midpoint=${clamp}(point-0.5*first*signedStep/h);
    point=${clamp}(point-${sample}(midpoint)*signedStep/h);remaining-=stepSeconds;
  }
  return point;`;
}
