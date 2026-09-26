/** Native relaxation: half the excess in 1/30 s, invariant to subdivision. */
export const uniformVolumeCorrectionWGSL = /* wgsl */ `
fn uvVolumeCorrectionFractionAt(dt:f32)->f32{
 let steps=max(dt,0.0)*30.0;let x=steps*0.6931471805599453;
 if(x<0.01){return x*(1.0-x*(0.5-x/6.0));}
 return 1.0-exp2(-steps);
}
fn uvVolumeCorrectionAmountAt(v:f32,cap:f32,dt:f32)->f32{
 return min(uvVolumeCorrectionFractionAt(dt)*max(0.0,v-cap),cap);
}
`;
