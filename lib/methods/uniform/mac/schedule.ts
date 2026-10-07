/** Device-written dispatch records. Every record is a padded u32 xyz triple. */
export const MAC_LAUNCH = { cells: 0, vertices: 16, one: 32, pressure: 48, pressureOne: 64, projected: 80, publish: 96, publishPhi: 112 } as const;
export const MAC_RECEIPT_BYTES = 128;
export const MAC_PRESSURE_BATCH = 32;

/** Capacity only. The GPU chooses every dt and suppresses unused slots. */
export function macSubstepSlots(duration: number, maxStep: number): number {
  return Math.max(1, Math.ceil(duration / maxStep - 1e-8)) + 2;
}

export const MAC_SCHEDULE_SHADER = /* wgsl */ `
// scalars[12] sticky failure; [15] previous dt; [16] current dt;
// [17] elapsed this frame; [18] frame substeps; [19] lifetime substeps;
// [20] pressure accepted; [21] active step; [22] frame accepted;
// [23] initial volume; [24] frame pressure iterations;
// [25] submitted pressure tolerance; [26] submitted iteration limit.
@group(0) @binding(20) var<storage,read_write> launches:array<vec4u>;
fn launch(n:u32)->vec4u{let g=(n+63u)/64u;return vec4u(min(g,65535u),(g+65534u)/65535u,1,0);}
fn noLaunch()->vec4u{return vec4u(0,1,1,0);}
fn finiteState()->bool{return scalars[11]==0.0&&scalars[6]<1e30&&abs(scalars[7])<1e30&&abs(scalars[8])<1e30&&scalars[9]<1e30;}
@compute @workgroup_size(1) fn seedState(){
 scalars[23]=scalars[7];scalars[25]=p.material.z;scalars[26]=p.solve.z;
 if(!finiteState()){scalars[12]=3.0;}
}
@compute @workgroup_size(1) fn beginFrame(){
 scalars[25]=p.material.z;scalars[26]=p.solve.z;
 scalars[17]=0.0;scalars[18]=0.0;scalars[22]=0.0;scalars[24]=0.0;
 for(var i=0u;i<8u;i++){launches[i]=noLaunch();}
}
@compute @workgroup_size(1) fn prepareStep(){
 for(var i=0u;i<6u;i++){launches[i]=noLaunch();}
 scalars[20]=0.0;scalars[21]=0.0;
 let remaining=p.solve.x-scalars[17];
 if(scalars[12]>0.0||remaining<=max(1e-10,p.solve.x*1e-6)){return;}
 let h=min(p.hdt.x,min(p.hdt.y,p.hdt.z));let distance=p.solve.y*h;
 let speed=sqrt(3.0)*scalars[6];let acceleration=length(p.gravityNu.xyz);
 var dt=min(remaining,min(p.hdt.w,2.0*distance/max(speed+sqrt(speed*speed+2.0*acceleration*distance),1e-30)));
 if(p.gravityNu.w>0.0){dt=min(dt,0.45/(p.gravityNu.w*dot(1.0/(p.hdt.xyz*p.hdt.xyz),vec3f(1))));}
 if(p.material.y>0.0){dt=min(dt,0.5*sqrt(p.material.x*h*h*h/(3.14159265359*p.material.y)));}
 if(!(dt>1e-9)){scalars[12]=4.0;return;}
 scalars[16]=dt;scalars[21]=1.0;scalars[10]=0.0;scalars[11]=0.0;
 for(var i=0u;i<6u;i++){scalars[i]=0.0;}
 launches[0]=launch(count());launches[1]=launch(vertices());launches[2]=vec4u(1,1,1,0);
 launches[3]=launches[0];launches[4]=launches[2];
}
@compute @workgroup_size(1) fn pressureVerdict(){
 if(scalars[21]>0.0&&scalars[20]==0.0){scalars[12]=1.0;}
 launches[5]=noLaunch();launches[2]=noLaunch();
 if(scalars[21]>0.0&&scalars[12]==0.0){launches[5]=launch(count());launches[2]=vec4u(1,1,1,0);}
}
@compute @workgroup_size(1) fn finishStep(){
 if(scalars[21]==0.0||scalars[12]>0.0){return;}
 if(!finiteState()){
  scalars[12]=3.0;return;
 }
 scalars[17]=min(p.solve.x,scalars[17]+scalars[16]);scalars[15]=scalars[16];
 scalars[18]+=1.0;scalars[19]+=1.0;scalars[24]+=scalars[10];
}
@compute @workgroup_size(1) fn endFrame(){
 if(scalars[12]==0.0&&p.solve.x-scalars[17]>max(1e-10,p.solve.x*1e-6)){scalars[12]=2.0;}
 if(scalars[12]==0.0){scalars[22]=1.0;launches[6]=launch(count());launches[7]=launch(vertices());}
}
`;
