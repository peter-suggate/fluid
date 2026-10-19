import {UNIFORM_SURFACE_TRANSPORT_ERROR_H} from "./uniform-surface-transport.wgsl";

/** Eight f32 words per tile, produced with the extended-velocity certificate:
 * mean.xyz, fine reconstruction defect (m/s), max component speed (m/s),
 * block identity (u32 bits), owner width, final defect (m/s).
 * Four additional words per tile follow the records as a compact FLIP copy:
 * speed, defect, identity, width. Four further words per tile hold
 * normal surface rate, momentum acceleration, reader class, velocity gradient.
 * Reader classes: 0 unused, 1 fully submerged, 2 surface/unknown. Taking
 * their maximum only certifies submerged when every physical cell agrees.
 * Fine defect compares all 64 native MAC samples with their trilinear corner
 * reconstruction. Coarse tiles use neighbouring means' second differences.
 * This is an update-error estimator, NOT a geometry/refinement decision or a
 * certificate that pressure can be omitted. Unknown/recycled evidence fails hot.
 * The heat consumes the preceding certificate, like regionVelocity; body,
 * source, topology and unknown-pressure invalidations still take precedence. */
export const UNIFORM_TILE_UPDATE_WORDS=8;
export const UNIFORM_TILE_UPDATE_BUDGET_H=4*UNIFORM_SURFACE_TRANSPORT_ERROR_H;
/** Uses the current workgroup's 4^3 MAC samples. Exported so numerical
 * regression tests execute the same estimator as the frame certificate. */
export const uniformTileFineDefectWGSL=/* wgsl */`
fn tuFineDefect(lane:u32)->f32{
 let q=vec3f(umCorner(lane,4u))/3.0;var reconstructed=vec3f(0.0);
 for(var k=0u;k<8u;k++){let c=umCorner(k,2u);let w=select(1.0-q,q,c!=vec3u(0));reconstructed+=w.x*w.y*w.z*planValues[3u*c.x+12u*c.y+48u*c.z];}
 let d=abs(planValues[lane]-reconstructed);return d.x+d.y+d.z;
}`;
export function uniformTileUpdateWGSL(group:number,binding:number,name="tileUpdate"):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> ${name}:array<f32>;
fn tuValid(t:u32)->bool{return umKeyStored(t)&&bitcast<u32>(${name}[8u*t+5u])==umBlockWord(t>>9u)&&${name}[8u*t+6u]>0.0;}
fn tuError(t:u32)->f32{if(!tuValid(t)){return 3.402823e38;}return ${name}[8u*t+7u];}
fn tuPhysical(t:u32)->vec4f{if(!tuValid(t)){return vec4f(3.402823e38,3.402823e38,2.0,3.402823e38);}let p=12u*UM_TILES+4u*t;return vec4f(${name}[p],${name}[p+1u],${name}[p+2u],${name}[p+3u]);}
fn tuSpeed(t:u32)->f32{if(!tuValid(t)){return 3.402823e38;}return ${name}[8u*t+4u];}
`;}

/** Pure reference for the tier-specific hold contract; also used by CPU tests.
 * Packing keeps both deadlines in one word: hot count low 16 bits,
 * warm count high 16 bits. Current evidence renews only its own timer. */
export function advanceTierHold(packed:number,hot:boolean,warm:boolean,frames:number):{packed:number;hot:boolean;warm:boolean}{
 const h=packed&65535,w=packed>>>16;
 return {packed:((hot?frames:Math.max(0,h-1))|((warm?frames:Math.max(0,w-1))<<16))>>>0,hot:h>0,warm:w>0};
}
export const uniformTierHoldWGSL=/* wgsl */`
// Independent expirations: WARM evidence can never renew a HOT hold.
let held=state[at+8u];let hotLeft=held&65535u;let warmLeft=held>>16u;
if(hotLeft>0u){trigger|=512u;}if(warmLeft>0u){trigger|=1024u;}
let nextHot=select(hotLeft-min(hotLeft,1u),RH_HOLD_FRAMES,(trigger&RH_HELD_HOT)!=0u);
let nextWarm=select(warmLeft-min(warmLeft,1u),RH_HOLD_FRAMES,(trigger&RH_HELD_WARM)!=0u);
state[at+8u]=nextHot|(nextWarm<<16u);
`;

/** Current trilinear cell: maximum directional phi derivative and whether
 * liquid or the seven-cell surface-velocity collar can read this cell. */
export const uniformTileSurfaceRateWGSL=/* wgsl */`
// Stay outside the surface's seven-h sampling collar, plus three frames of
// characteristic travel (the previous certificate and the receipt latency).
// A flat/stationary surface is still class 2: zero normal rate is not depth.
fn tuReaderClass(corners:array<f32,8>,h:f32,dt:f32,speed:f32,reader:bool)->f32{
 if(!reader){return 0.0;}
 let margin=7.0*h+3.0*dt*speed;
 for(var k=0u;k<8u;k++){if(!(corners[k]<-margin)){return 2.0;}}
 return 1.0;
}
fn tuSurfaceRate(corners:array<f32,8>,velocity:vec3f,h:f32,width:u32,open:bool)->vec2f{
 var lo=3.402823e38;var hi=-3.402823e38;
 for(var k=0u;k<8u;k++){lo=min(lo,corners[k]);hi=max(hi,corners[k]);}
 if(!open||lo>7.0*h){return vec2f(0.0);}
 var rate=0.0;
 if(hi>=-7.0*h){
  // Preserve the dot product: tangential translation of a plane causes no
  // normal transport. Corner derivatives bound the trilinear polynomial.
  for(var k=0u;k<8u;k++){
   let g=vec3f(corners[k|1u]-corners[k&6u],corners[k|2u]-corners[k&5u],corners[k|4u]-corners[k&3u])/(f32(width)*h);
   rate=max(rate,abs(dot(g,velocity)));
  }
 }
 return vec2f(rate,1.0);
}
`;
