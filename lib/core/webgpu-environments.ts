/**
 * Shared art-directed environment library for both water presentation paths.
 * The dry room is ray-intersected in world space, which means its architecture
 * is present in the scene texture and therefore bends through the raster
 * optics pass instead of behaving like a screen-space backplate.
 *
 * The including shader must expose the standard `u: Uniforms` binding with an
 * `environment: vec4f` field. `environment.x` is the preset index.
 */
export const environmentShaderLibrary = /* wgsl */ `
struct EnvironmentSample { color:vec3f, depth:f32 }
struct EnvironmentPropHit { t:f32, normal:vec3f, color:vec3f, emission:f32 }

fn environmentIndex()->i32{return i32(round(u.environment.x));}

// --- Scene terrain heightfield -------------------------------------------
// Mirrors lib/terrain.ts exactly (same falloff and p-norm basin union) so the
// rendered ground is the surface the solver's solid fractions were baked
// from. terrainMeta = (enabled, baseHeight_m, featureCount, unionExponent);
// each feature is two vec4s: (cx, cz, rx, rz) and (signedAmount, rotation,
// flat, 0) with mounds positive and basins negative.
fn envTerrainEnabled()->bool{return u.terrainMeta.x>0.5;}
fn envTerrainHeightAt(x:f32,z:f32)->f32{
  if(!envTerrainEnabled()){return 0.0;}
  var mounds=0.0;var carvePower=0.0;let exponent=max(u.terrainMeta.w,1.0);
  let count=i32(round(u.terrainMeta.z));
  for(var i=0;i<count;i+=1){
    let a=u.terrainFeatures[2*i];let b=u.terrainFeatures[2*i+1];
    let cs=cos(b.y);let sn=sin(b.y);
    let dx=x-a.x;let dz=z-a.y;
    let localX=(cs*dx+sn*dz)/a.z;let localZ=(-sn*dx+cs*dz)/a.w;
    let d=length(vec2f(localX,localZ));
    var w=0.0;
    if(d<=b.z){w=1.0;}
    else if(d<1.0){let ss=1.0-(d-b.z)/(1.0-b.z);w=ss*ss*(3.0-2.0*ss);}
    if(b.x>=0.0){mounds+=b.x*w;}
    else{carvePower+=pow(-b.x*w,exponent);}
  }
  var carve=0.0;
  if(carvePower>0.0){carve=pow(carvePower,1.0/exponent);}
  return max(0.0,u.terrainMeta.y+mounds-carve);
}
// The garden lawn continues past the simulated footprint: outside the
// container the analytic features have decayed to the base level, and a soft
// meadow swell (zero at the boundary) keeps the horizon from reading flat.
fn envGardenGroundY(p:vec2f)->f32{
  var h=envTerrainHeightAt(p.x,p.y);
  let outside=max(max(abs(p.x)-.5*u.container.x,abs(p.y)-.5*u.container.z),0.0);
  let swell=smoothstep(0.0,1.2,outside);
  h+=.10*u.terrainMeta.y*swell*sin(p.x*1.15+2.1)*sin(p.y*1.45+0.7);
  return h;
}
fn envTerrainCeiling()->f32{
  var top=u.terrainMeta.y*1.15+.05;
  let count=i32(round(u.terrainMeta.z));
  for(var i=0;i<count;i+=1){let amount=u.terrainFeatures[2*i+1].x;if(amount>0.0){top+=amount;}}
  return top;
}
fn envTerrainNormal(p:vec2f)->vec3f{
  let e=.02;
  let gx=(envGardenGroundY(p+vec2f(e,0))-envGardenGroundY(p-vec2f(e,0)))/(2.0*e);
  let gz=(envGardenGroundY(p+vec2f(0,e))-envGardenGroundY(p-vec2f(0,e)))/(2.0*e);
  return normalize(vec3f(-gx,1.0,-gz));
}
// Sphere-trace substitute for a heightfield: graded fixed march (denser near
// the camera) plus a short bisection refine on the crossing interval.
fn envTerrainTrace(ro:vec3f,rd:vec3f)->f32{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let ceiling=envTerrainCeiling();
  var t0=.005;
  if(ro.y>ceiling){
    if(rd.y>=-.0005){return -1.0;}
    t0=(ceiling-ro.y)/rd.y;
  }
  var t1=t0+10.0*s;
  if(rd.y<-.0005){t1=min(t1,(-.02-ro.y)/rd.y);}
  else if(rd.y>.0005){t1=min(t1,max(t0,(ceiling-ro.y)/rd.y));}
  if(t1<=t0){return -1.0;}
  var previousT=t0;
  let startP=ro+rd*t0;
  if(startP.y-envGardenGroundY(startP.xz)<=0.0){return t0;}
  for(var i=1;i<=56;i+=1){
    let t=t0+(t1-t0)*pow(f32(i)/56.0,1.4);
    let p=ro+rd*t;
    if(p.y-envGardenGroundY(p.xz)<=0.0){
      var a=previousT;var b=t;
      for(var j=0;j<6;j+=1){let m=.5*(a+b);let pm=ro+rd*m;if(pm.y-envGardenGroundY(pm.xz)>0.0){a=m;}else{b=m;}}
      return .5*(a+b);
    }
    previousT=t;
  }
  return -1.0;
}
fn envHash21(p:vec2f)->f32{return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453);}
fn envHash31(p:vec3f)->f32{return fract(sin(dot(p,vec3f(127.1,311.7,74.7)))*43758.5453);}
// Width is expressed as the retained cell interior (0.46 is a 0.04-wide
// architectural seam). Keeping this convention makes material call sites read
// as panel/tile coverage rather than inverted line thickness.
fn envLine(value:f32,width:f32)->f32{let d=abs(fract(value)-.5);let halfWidth=max(.002,.5-width);return 1.0-smoothstep(halfWidth,halfWidth+.008,d);}
fn envGrid(value:vec2f,width:f32)->f32{return max(envLine(value.x,width),envLine(value.y,width));}
fn envBoxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let n=min(a,b);let f=max(a,b);return vec2f(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
// Slab test with a caller-hoisted reciprocal: the lab's occlusion sweep issues
// many box tests per ray, and the divides dominate otherwise.
fn envSlabHit(ro:vec3f,inv:vec3f,mn:vec3f,mx:vec3f)->vec2f{let a=(mn-ro)*inv;let b=(mx-ro)*inv;let n=min(a,b);let f=max(a,b);return vec2f(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
fn envSlabBlocks(ro:vec3f,inv:vec3f,mn:vec3f,mx:vec3f,maxT:f32)->bool{let hit=envSlabHit(ro,inv,mn,mx);return hit.x<=hit.y&&hit.x>.001&&hit.x<maxT;}
fn envMiss()->EnvironmentPropHit{return EnvironmentPropHit(1e20,vec3f(0,1,0),vec3f(0),0.0);}
fn envEllipsoidPrimitive(ro:vec3f,rd:vec3f,center:vec3f,radius:vec3f,color:vec3f,emission:f32)->EnvironmentPropHit{
  let o=(ro-center)/radius;let d=rd/radius;let a=dot(d,d);let b=dot(o,d);let c=dot(o,o)-1.0;let disc=b*b-a*c;if(disc<0.0){return envMiss();}var t=(-b-sqrt(disc))/a;if(t<=.001){t=(-b+sqrt(disc))/a;}if(t<=.001){return envMiss();}let p=ro+rd*t-center;return EnvironmentPropHit(t,normalize(p/(radius*radius)),color,emission);
}

// Ground shading for the porcelain garden: the same structure as before —
// pebbled pool liner below the waterline, a pale collar, then the lawn with
// mow stripes, tone noise, matte patches and bright white flecks — but in a
// monochrome white-clay palette. The dark liner is what lets the water read
// as deep. The hollow self-occludes with depth.
fn gardenGroundMaterial(p:vec3f)->vec3f{
  let base=u.terrainMeta.y;
  let waterline=u.container.w;
  let cell=floor(p.xz*26.0);
  let jitter=vec2f(envHash21(cell),envHash21(cell+19.7))-.5;
  let pebbleDistance=length(fract(p.xz*26.0)-.5-jitter*.55);
  let pebbleTone=.55+.45*envHash21(cell+7.3);
  let liner=mix(vec3f(.135,.13,.125),vec3f(.44,.435,.42)*pebbleTone,smoothstep(.44,.18,pebbleDistance));
  let sand=vec3f(.56,.55,.52)*(.9+.2*envHash21(floor(p.xz*40.0)));
  let stripe=.5+.5*sin((p.x*.9+p.z*.35)*4.4);
  var lawn=mix(vec3f(.46,.455,.435),vec3f(.66,.65,.62),.5*stripe+.5*envHash21(floor(p.xz*90.0)));
  let clover=step(.962,envHash21(floor(p.xz*14.0)));
  lawn=mix(lawn,vec3f(.58,.575,.55),clover*.55);
  let daisy=step(.986,envHash21(floor(p.xz*24.0)+3.1));
  lawn=mix(lawn,vec3f(.95,.94,.90),daisy*.85);
  let sandBand=smoothstep(waterline-.02,waterline+.04,p.y);
  var c=mix(liner,sand,sandBand);
  c=mix(c,lawn,smoothstep(base-.05,base-.008,p.y));
  // Hollow self-occlusion: strongest at the pool bed, easing out through the
  // banks so the carve reads even from a high camera.
  let hollow=smoothstep(0.0,max(base,1e-3),p.y);
  return c*(.38+.62*hollow*hollow);
}

fn envRoomHalf()->vec3f{
  let span=max(max(u.container.x,u.container.y),u.container.z);
  return vec3f(max(u.container.x*2.8,span*2.25),max(u.container.y*1.85,span*1.8),max(u.container.z*2.8,span*2.25));
}
// The lab drops its floor so the tank can rest on a bench at y = 0; every
// other preset keeps the floor just under the tank base.
fn environmentFloorY()->f32{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  return select(-.025,-.72*s,environmentIndex()==2);
}
fn labTableHalf()->vec2f{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  return vec2f(u.container.x*.5+.30*s,u.container.z*.5+.26*s);
}
fn labBenchZ()->f32{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  return -envRoomHalf().z+.36*s;
}
fn labLampPosition()->vec3f{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let th=labTableHalf();
  return vec3f(-(th.x-.17*s),0.0,th.y-.20*s);
}
// Shadow rays only need an any-hit answer against the handful of broad
// occluders (slabs approximate the seat and lamp shade), so they never touch
// the full primitive list. Early-outs keep the common lit case at one test.
fn labShadowVisibility(ro:vec3f,rd:vec3f,maxT:f32)->f32{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let floorY=environmentFloorY();let th=labTableHalf();let zb=labBenchZ();let lamp=labLampPosition();
  let inv=1.0/rd;
  if(envSlabBlocks(ro,inv,vec3f(-th.x,-.040*s,-th.y),vec3f(th.x,-.002*s,th.y),maxT)){return .16;}
  if(envSlabBlocks(ro,inv,vec3f(-(th.x-.13*s),floorY+.147*s,-(th.y-.13*s)),vec3f(th.x-.13*s,floorY+.173*s,th.y-.13*s),maxT)){return .16;}
  let stoolX=th.x+.45*s;
  if(envSlabBlocks(ro,inv,vec3f(stoolX-.17*s,floorY+.446*s,-.07*s),vec3f(stoolX+.17*s,floorY+.494*s,.27*s),maxT)){return .16;}
  if(envSlabBlocks(ro,inv,vec3f(-1.42*s,floorY+1.604*s,zb-.22*s),vec3f(1.02*s,floorY+1.636*s,zb+.12*s),maxT)){return .16;}
  if(envSlabBlocks(ro,inv,vec3f(-1.80*s,floorY+.840*s,zb-.34*s),vec3f(1.80*s,floorY+.884*s,zb+.34*s),maxT)){return .16;}
  if(envSlabBlocks(ro,inv,vec3f(.65*s,floorY+.98*s,zb+.036*s),vec3f(1.25*s,floorY+1.36*s,zb+.064*s),maxT)){return .16;}
  if(envSlabBlocks(ro,inv,vec3f(lamp.x-.098*s,.558*s,lamp.z-.098*s),vec3f(lamp.x+.098*s,.702*s,lamp.z+.098*s),maxT)){return .16;}
  return 1.0;
}
// The polished floor only picks up the light fixtures: one slab crossing for
// the troffer plane (footprint-tested at the crossing), the lamp bulb and the
// monitor screen. Everything else falls back to the ambient gradient.
fn labEmissiveReflection(ro:vec3f,rd:vec3f)->vec3f{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let floorY=environmentFloorY();let zb=labBenchZ();let ceilY=floorY+2.0*envRoomHalf().y;let lamp=labLampPosition();
  let inv=1.0/rd;
  let tro=envSlabHit(ro,inv,vec3f(-1.50*s,ceilY-.047*s,-.50*s),vec3f(1.50*s,ceilY-.023*s,1.15*s));
  if(tro.x<=tro.y&&tro.x>.001){
    let p=ro+rd*tro.x;
    if(abs(abs(p.x)-.95*s)<.55*s&&(abs(p.z+.30*s)<.20*s||abs(p.z-.95*s)<.20*s)){return vec3f(.92,.93,.90)*2.2;}
  }
  if(envEllipsoidPrimitive(ro,rd,vec3f(lamp.x,.585*s,lamp.z),vec3f(.046*s),vec3f(1.0),1.0).t<1e19){return vec3f(1.0,.78,.45)*2.9;}
  if(envSlabBlocks(ro,inv,vec3f(.685*s,floorY+1.015*s,zb+.064*s),vec3f(1.215*s,floorY+1.325*s,zb+.072*s),1e19)){return vec3f(.25,.45,.58)*2.6;}
  return environmentLight(rd);
}
// Two ceiling troffers (with real shadow rays on upward-facing surfaces), a
// warm task lamp and faint cool monitor spill. The troffer sample points sit
// below the emissive fixture boxes so their own geometry never occludes them.
fn labKeyLights(p:vec3f,n:vec3f,withShadow:bool)->vec3f{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let ceilY=environmentFloorY()+2.0*envRoomHalf().y;
  var sum=vec3f(0.0);
  // One visibility ray toward the fixture midpoint stands in for both
  // troffers; the individual penumbrae were nearly coincident anyway.
  var vis=1.0;
  if(withShadow){
    let M=vec3f(0.0,ceilY-.16*s,-.30*s);
    let dm=M-p;let distM=max(length(dm),1e-4);
    vis=labShadowVisibility(p+dm*(.02*s/distM),dm/distM,distM-.08*s);
  }
  for(var i=-1;i<=1;i+=2){
    let L=vec3f(f32(i)*.95*s,ceilY-.16*s,-.30*s);
    let d=L-p;let dist=max(length(d),1e-4);let w=d/dist;
    sum+=vec3f(1.0,.96,.88)*(max(dot(n,w),0.0)*1.15/(1.0+dist*dist/(1.4*s*s)))*vis;
  }
  let lamp=labLampPosition();
  let L2=vec3f(lamp.x+.07*s,.52*s,lamp.z+.07*s);
  let d2=L2-p;let dist2=max(length(d2),1e-4);
  sum+=vec3f(1.0,.58,.24)*(max(dot(n,d2/dist2),0.0)*1.4/(1.0+dist2*dist2/(.42*s*s)));
  let M=vec3f(.95*s,environmentFloorY()+1.17*s,labBenchZ()+.10*s);
  let d3=M-p;let dist3=max(length(d3),1e-4);
  sum+=vec3f(.22,.42,.55)*(max(dot(n,d3/dist3),0.0)*.5/(1.0+dist3*dist3/(.60*s*s)));
  return sum;
}
// Soft occlusion the glass tank casts on the bench top, nudged away from the
// troffer pair.
fn labTankShadow(p:vec3f,n:vec3f)->f32{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  if(n.y<.6||abs(p.y)>.09*s){return 1.0;}
  let sh=envFootprintShadow(p,vec2f(.10*s,.07*s),vec2f(u.container.x,u.container.z)*.85);
  return 1.0-.40*sh;
}
fn labWindowMask(p:vec3f,n:vec3f)->f32{
  if(n.z<.5){return 0.0;}
  let s=max(max(u.container.x,u.container.y),u.container.z);
  let floorY=environmentFloorY();
  return step(abs(p.x),1.62*s)*step(floorY+1.05*s,p.y)*step(p.y,floorY+2.15*s);
}
// Room surfaces: ambient plus the shadowed key lights; the polished floor also
// traces one reflection ray so the troffers and lamp smear across it. The
// night window keeps its own luminance instead of being lit as paint.
fn labRoomShade(albedo:vec3f,p:vec3f,n:vec3f,rd:vec3f)->vec3f{
  let s=max(max(u.container.x,u.container.y),u.container.z);
  var c=albedo*(vec3f(.055,.060,.072)+labKeyLights(p,n,n.y>.7))*environmentContactShadow(p,n);
  if(n.y>.7){
    let refl=labEmissiveReflection(p+vec3f(0.0,.01*s,0.0),reflect(rd,n));
    let fres=.04+.34*pow(1.0-max(dot(-rd,n),0.0),3.0);
    c+=refl*fres;
  }
  return mix(c,albedo*1.35,labWindowMask(p,n));
}

fn environmentLightDirection()->vec3f{
  let e=environmentIndex();
  if(e==2){return normalize(vec3f(-.35,.90,.20));}
  if(e==3){return normalize(vec3f(.32,.82,.34));}
  if(e==4){return normalize(vec3f(-.55,.75,-.12));}
  if(e==5){return normalize(vec3f(.15,.42,.90));}
  if(e==7){return normalize(vec3f(-.42,.72,.38));}
  if(e==8){return normalize(vec3f(-.22,.94,.26));}
  return normalize(vec3f(-.45,.86,.28));
}
fn environmentLightColor()->vec3f{
  let e=environmentIndex();
  if(e==6){return vec3f(1.0,.86,.66);}
  if(e==8){return vec3f(1.0,.96,.90);}
  if(e==1){return vec3f(1.0,.77,.52);}
  if(e==2){return vec3f(1.0,.94,.80);}
  if(e==3){return vec3f(1.0,.67,.40);}
  if(e==4){return vec3f(1.0,.83,.57);}
  if(e==5){return vec3f(.42,.83,1.0);}
  if(e==7){return vec3f(1.0,.97,.90);}
  return vec3f(1.0,.86,.62);
}
fn environmentAccent()->vec3f{
  let e=environmentIndex();
  if(e==6){return vec3f(.18,.34,.31);}
  if(e==8){return vec3f(.012,.015,.022);}
  if(e==1){return vec3f(.10,.34,.44);}
  if(e==2){return vec3f(.08,.11,.15);}
  if(e==3){return vec3f(.72,.42,.22);}
  if(e==4){return vec3f(.54,.39,.25);}
  if(e==5){return vec3f(.14,.56,.68);}
  if(e==7){return vec3f(.48,.50,.53);}
  return vec3f(.24,.55,.39);
}

fn envFootprintShadow(p:vec3f,center:vec2f,radius:vec2f)->f32{
  let q=(p.xz-center)/radius;
  return 1.0-smoothstep(.18,1.0,length(q));
}
fn environmentContactShadow(p:vec3f,n:vec3f)->f32{
  if(n.y<.7){return 1.0;}
  let e=environmentIndex();let s=max(max(u.container.x,u.container.y),u.container.z);var shadow=0.0;
  if(e==0){shadow=max(envFootprintShadow(p,vec2f(-1.18,-.70)*s,vec2f(.74,.34)*s),envFootprintShadow(p,vec2f(1.12,-.86)*s,vec2f(.52,.43)*s));}
  else if(e==1){shadow=max(envFootprintShadow(p,vec2f(-1.12,-.54)*s,vec2f(.68,.30)*s),envFootprintShadow(p,vec2f(1.12,-.68)*s,vec2f(.54,.48)*s));}
  else if(e==2){let th=labTableHalf();let zb=labBenchZ();shadow=max(envFootprintShadow(p,vec2f(0.0,0.0),vec2f(th.x+.55*s,th.y+.55*s)),max(envFootprintShadow(p,vec2f(0.0,zb),vec2f(2.0*s,.72*s)),envFootprintShadow(p,vec2f(th.x+.45*s,.10*s),vec2f(.40*s,.40*s))));}
  else if(e==3){shadow=max(envFootprintShadow(p,vec2f(-1.08,-.42)*s,vec2f(.72,.29)*s),envFootprintShadow(p,vec2f(1.08,-.86)*s,vec2f(.48,.42)*s));}
  else if(e==4){shadow=max(envFootprintShadow(p,vec2f(-.70,-1.02)*s,vec2f(.38,.34)*s),max(envFootprintShadow(p,vec2f(-1.08,-.42)*s,vec2f(.38,.33)*s),envFootprintShadow(p,vec2f(1.06,-.70)*s,vec2f(.35,.30)*s)));}
  else if(e==5){shadow=max(envFootprintShadow(p,vec2f(-.76,-1.04)*s,vec2f(.46,.38)*s),max(envFootprintShadow(p,vec2f(-1.16,-.72)*s,vec2f(.48,.42)*s),envFootprintShadow(p,vec2f(1.16,-.72)*s,vec2f(.48,.42)*s)));}
  return 1.0-.34*clamp(shadow,0.0,1.0);
}

fn environmentLight(rd:vec3f)->vec3f{
  let e=environmentIndex();let t=clamp(rd.y*.5+.5,0.0,1.0);var c=vec3f(0.0);
  if(e==6){c=mix(vec3f(.012,.025,.028),vec3f(.19,.30,.29),t);}
  // The stage: STUDIO_STAGE_DRY_SCENE_LIGHTING's near-black cool hemisphere,
  // so the water reflects the same dark surround the set stands in.
  else if(e==8){c=mix(vec3f(.008,.010,.014),vec3f(.016,.020,.028),t);}
  else if(e==0){c=mix(vec3f(.035,.055,.044),vec3f(.46,.58,.43),t);}
  else if(e==1){c=mix(vec3f(.16,.16,.13),vec3f(.68,.69,.59),t);}
  else if(e==2){c=mix(vec3f(.016,.017,.020),vec3f(.065,.068,.078),t);c+=vec3f(.30,.30,.27)*pow(max(rd.y,0.0),6.0)+vec3f(.05,.09,.16)*pow(max(-rd.z,0.0),4.0)*.6;}
  else if(e==3){c=mix(vec3f(.045,.050,.048),vec3f(.30,.32,.30),t);}
  else if(e==4){c=mix(vec3f(.045,.038,.032),vec3f(.34,.31,.25),t);}
  else if(e==7){
    // Porcelain sky: a near-white horizon into a pale blue-grey zenith with
    // soft drifting cumulus, easing into a misty pale ground line so the
    // whole backdrop stays monochrome around the water.
    c=mix(vec3f(.88,.88,.86),vec3f(.52,.60,.72),pow(t,1.35));
    let q=rd.xz/max(rd.y+.16,.09);
    let cloudField=.5+.5*sin(q.x*1.25+sin(q.y*1.85))+.32*sin(q.y*3.6+q.x*2.15);
    let cloud=smoothstep(.86,1.32,cloudField)*smoothstep(.015,.22,rd.y);
    c=mix(c,vec3f(1.0,.99,.98),cloud*.85);
    c=mix(vec3f(.60,.61,.59),c,smoothstep(-.03,.05,rd.y));
  }
  else {c=mix(vec3f(.002,.012,.022),vec3f(.028,.12,.17),t);}
  let sun=max(dot(rd,environmentLightDirection()),0.0);
  c+=environmentLightColor()*(pow(sun,360.0)*2.5+pow(sun,15.0)*.22);
  return c;
}

fn conservatoryMaterial(p:vec3f,n:vec3f)->vec3f{
  if(n.y>.7){let tile=p.xz*2.25;let grout=envGrid(tile,.465);let variation=envHash21(floor(tile));return mix(vec3f(.46,.43,.32)*(0.88+.16*variation),vec3f(.18,.22,.18),grout*.72);}
  if(n.y<-.7){let rib=max(envLine(p.x*.72,.475),envLine(p.z*.72,.475));return mix(vec3f(.15,.24,.18),vec3f(.52,.54,.37),rib*.28);}
  let uv=select(p.xy,p.zy,abs(n.z)>.5);let mullion=max(envLine((uv.x+.12)*1.55,.458),envLine((uv.y-.18)*1.08,.466));
  let canopy=.5+.5*sin(uv.x*1.7+sin(uv.y*2.1));let garden=mix(vec3f(.035,.15,.095),vec3f(.24,.42,.18),canopy);
  let haze=smoothstep(.05,1.4,uv.y);return mix(garden,vec3f(.56,.62,.43),haze*.48)+mullion*vec3f(.38,.42,.31);
}

fn courtyardMaterial(p:vec3f,n:vec3f)->vec3f{
  if(n.y>.7){var q=p.xz*2.55;let row=floor(q.y);q.x+=select(0.0,.5,(i32(row)&1)==1);let grout=envGrid(q,.455);let variation=envHash21(floor(q));let clay=mix(vec3f(.42,.18,.11),vec3f(.68,.34,.21),variation);return mix(clay,vec3f(.23,.18,.14),grout*.86);}
  if(n.y<-.7){return vec3f(.55,.49,.38);}
  let uv=select(p.xy,p.zy,abs(n.z)>.5);let archCenter=vec2f(uv.x,uv.y-.82);let lowerOpening=abs(uv.x)<.42&&uv.y>.04&&uv.y<.82;let roundOpening=length(archCenter)<.42&&uv.y>=.78;let opening=abs(n.z)>.5&&(lowerOpening||roundOpening);
  let plaster=vec3f(.72,.65,.52)*(0.94+.06*envHash21(floor(uv*4.0)));let blueBand=1.0-smoothstep(.025,.055,abs(uv.y-.30));let reveal=vec3f(.025,.15,.22)+vec3f(.10,.22,.20)*clamp(uv.y,0.0,1.0);
  return mix(mix(plaster,vec3f(.035,.20,.29),blueBand*.82),reveal,select(0.0,1.0,opening));
}

fn labMaterial(p:vec3f,n:vec3f)->vec3f{
  let s=max(max(u.container.x,u.container.y),u.container.z);let floorY=environmentFloorY();
  if(n.y>.7){
    // Polished resin tiles; sheen arrives via the traced reflection ray.
    let q=p.xz/(1.05*s);let grout=envGrid(q,.488);let v=envHash21(floor(q));
    return mix(vec3f(.285,.295,.310)*(.92+.13*v),vec3f(.195,.205,.215),grout*.6);
  }
  if(n.y<-.7){let grid=envGrid(p.xz/(.62*s),.480);return vec3f(.46,.46,.45)*(1.0-.4*grid);}
  var c=vec3f(.56,.54,.51)*(.95+.05*envHash21(floor(vec2f(select(p.x,p.z,abs(n.x)>.5),p.y)*2.6)));
  c*=1.0-.40*smoothstep(floorY+1.6*s,floorY+3.1*s,p.y);
  let dadoY=floorY+.92*s;
  c=mix(c,vec3f(.225,.26,.275),smoothstep(dadoY+.015*s,dadoY-.015*s,p.y));
  c=mix(c,vec3f(.40,.42,.43),(1.0-smoothstep(.010*s,.022*s,abs(p.y-dadoY)))*.85);
  c=mix(c,vec3f(.14,.15,.16),1.0-smoothstep(.06*s,.09*s,p.y-floorY));
  if(n.z>.5){
    // Night window behind the tank: sky gradient, moon, a hashed skyline with
    // sparse lit offices, and a painted steel frame.
    let winL=floorY+1.05*s;let winT=floorY+2.15*s;
    if(abs(p.x)<1.62*s&&p.y>winL&&p.y<winT){
      var win=mix(vec3f(.008,.014,.034),vec3f(.030,.060,.120),clamp((p.y-winL)/(1.10*s),0.0,1.0));
      let lane=floor(p.x/(.16*s));let buildingTop=winL+(.10+.30*envHash21(vec2f(lane,7.0)))*s;
      let building=step(p.y,buildingTop);
      win=mix(win,vec3f(.010,.014,.021),building);
      let cellHash=envHash21(vec2f(floor(p.x/(.05*s)),floor((p.y-floorY)/(.045*s))));
      win+=vec3f(.96,.70,.34)*select(0.0,1.4,cellHash>.90&&building>.5);
      let moonD=length(p.xy-vec2f(-1.28*s,floorY+1.98*s));
      win+=vec3f(.90,.93,.88)*(1.0-smoothstep(.070*s,.105*s,moonD))*2.2+vec3f(.16,.20,.30)*pow(max(0.0,1.0-moonD/(1.1*s)),2.0)*.5;
      let frameD=min(min(1.62*s-abs(p.x),p.y-winL),winT-p.y);
      let frame=max(max(envLine(p.x/(.81*s)+.5,.458),1.0-smoothstep(.012*s,.022*s,abs(p.y-(floorY+1.62*s)))),1.0-smoothstep(.02*s,.05*s,frameD));
      return mix(win,vec3f(.15,.16,.17),frame);
    }
  }
  if(n.x>.5){
    // Whiteboard on the left wall with a few faint marker rows.
    if(p.z>-1.5*s&&p.z<-.1*s&&p.y>floorY+1.05*s&&p.y<floorY+1.95*s){
      var b=vec3f(.62,.63,.62);
      // Marker rows with hashed start, length and presence so it reads as
      // handwriting rather than ruled paper.
      let row=floor((p.y-floorY)/(.11*s));let rh=envHash21(vec2f(row,3.0));
      let z0=-1.35*s+.45*s*rh;let z1=-.35*s-.55*s*fract(rh*7.31);
      let stroke=envLine((p.y-floorY)/(.11*s),.485)*step(z0,p.z)*step(p.z,z1)*step(.30,rh)*step(floorY+1.22*s,p.y)*step(p.y,floorY+1.78*s);
      b=mix(b,select(vec3f(.16,.28,.40),vec3f(.42,.16,.14),rh>.78),stroke*.5);
      let frameD=min(min(p.z+1.5*s,-.1*s-p.z),min(p.y-(floorY+1.05*s),floorY+1.95*s-p.y));
      return mix(b,vec3f(.30,.31,.32),1.0-smoothstep(.015*s,.035*s,frameD));
    }
  }
  return c;
}

fn galleryMaterial(p:vec3f,n:vec3f)->vec3f{
  if(n.y>.7){let seam=envGrid(p.xz*.82,.488);return vec3f(.105,.115,.11)+seam*vec3f(.06,.065,.06);}
  let uv=select(p.xy,p.zy,abs(n.z)>.5);let boards=envLine((uv.x+uv.y*.08)*.72,.474);let tie=smoothstep(.045,.012,length(fract(uv*.72)-.5));var concrete=vec3f(.29,.30,.28)*(0.91+.09*envHash21(floor(uv*3.0)))+boards*.045-tie*.08;
  let portalSdf=max(abs(uv.x)-.78,max(abs(uv.y-.82)-.80,.04-uv.y));let glow=(1.0-smoothstep(-.025,.12,portalSdf))*select(0.0,1.0,abs(n.z)>.5);concrete=mix(concrete,vec3f(1.15,.54,.22),glow*.90);
  return concrete;
}

fn bathhouseMaterial(p:vec3f,n:vec3f)->vec3f{
  if(n.y>.7){let q=p.xz*3.1;let cell=floor(q);let pebble=length(fract(q)-.5);let variation=envHash21(cell);return mix(vec3f(.12,.14,.13),vec3f(.30,.29,.25),smoothstep(.48,.20,pebble)*(0.55+.35*variation));}
  if(n.y<-.7){return vec3f(.23,.18,.13);}
  let uv=select(p.xy,p.zy,abs(n.z)>.5);let batten=envLine(uv.x*2.25,.455);let rail=envLine(uv.y*1.15,.474);let paper=vec3f(.68,.59,.43)*(1.0+.12*sin(uv.y*1.7));let wood=vec3f(.25,.14,.075)*(0.86+.14*sin(uv.y*9.0));return mix(paper,wood,max(batten,rail));
}

fn stationMaterial(p:vec3f,n:vec3f)->vec3f{
  let uv=select(p.xz,select(p.xy,p.zy,abs(n.z)>.5),abs(n.y)<.7);let ribs=envLine(uv.x*.78,.462);let panels=envGrid(uv*1.25,.486);var metal=vec3f(.012,.045,.065)+ribs*vec3f(.07,.22,.26)+panels*vec3f(.02,.055,.067);if(abs(n.y)>.7){return metal*.72;}
  let portCenter=vec2f((fract(uv.x*.23+.5)-.5)*4.35,uv.y-.55);let ring=abs(length(portCenter)-.54);let port=smoothstep(.08,.025,ring);let glass=smoothstep(.46,.42,length(portCenter));metal=mix(metal,vec3f(.015,.22,.31)+vec3f(.08,.28,.29)*max(0.0,sin(uv.y*2.0)),glass*.8)+port*vec3f(.50,.36,.16);
  let instrument=envLine(uv.y*2.8,.492)*envLine(uv.x*5.0,.49);return metal+instrument*vec3f(.15,.78,.72);
}

fn sampleEnvironment(ro:vec3f,rd:vec3f)->EnvironmentSample{
  let e=environmentIndex();
  if(e==7){
    // Open-air garden: no room box. The terrain heightfield IS the ground —
    // the pool cavity, banks and lawn are the same surface the solver solids
    // were baked from. The trees, mushrooms and pots that stand on it are
    // scenery, and scenery is traced from the scene's own description.
    let terrainT=envTerrainTrace(ro,rd);
    if(terrainT>0.0){
      let p=ro+rd*terrainT;
      let n=envTerrainNormal(p.xz);
      let l=environmentLightDirection();
      let daylight=.28+.82*max(dot(n,l),0.0);
      // Soft canopy shadows under the two cloud trees and the grand mushroom.
      let s=max(max(u.container.x,u.container.y),u.container.z);
      let treeShadow=1.0-.38*max(envFootprintShadow(p,vec2f(-.40,-.08)*s,vec2f(.33,.31)*s),max(envFootprintShadow(p,vec2f(.45,-.25)*s,vec2f(.25,.24)*s),envFootprintShadow(p,vec2f(-.20,-.33)*s,vec2f(.17,.17)*s)));
      var c=gardenGroundMaterial(p)*environmentLightColor()*daylight*treeShadow;
      let fresnel=pow(1.0-max(dot(-rd,n),0.0),4.0);
      c+=vec3f(.35,.37,.40)*fresnel*.12;
      let haze=1.0-exp(-terrainT*.05);
      c=mix(c,vec3f(.80,.81,.83),haze*.35);
      return EnvironmentSample(c,terrainT);
    }
    return EnvironmentSample(environmentLight(rd),65504.0);
  }
  if(e==6){
    let t=clamp(rd.y*.5+.5,0.0,1.0);var color=mix(vec3f(.015,.027,.029),vec3f(.16,.23,.22),t);let sun=max(dot(rd,normalize(vec3f(-.45,.86,.28))),0.0);color+=vec3f(1.0,.86,.66)*pow(sun,320.0)*2.2+vec3f(.24,.31,.28)*pow(sun,12.0);
    let floorT=(-.012-ro.y)/rd.y;if(floorT>0.0){let p=ro+rd*floorT;let radial=length(p.xz);let checker=.5+.5*cos(p.x*31.4)*cos(p.z*31.4);color=mix(color,vec3f(.055,.068,.064)+checker*vec3f(.018,.025,.022),.82*exp(-radial*.22));return EnvironmentSample(color,floorT);}
    return EnvironmentSample(color,65504.0);
  }
  if(e==8){
    // The stage as a backplate, for the presentations that do not build the
    // set: a near-black surround, dark boards, and one warm pool that covers
    // the container and falls off inside the frame — the same picture the
    // authored set makes, at the fidelity a backdrop needs.
    var color=environmentLight(rd);
    let floorT=(-.012-ro.y)/rd.y;
    if(floorT>0.0){
      let p=ro+rd*floorT;let s=max(u.container.x,u.container.z);
      let pool=exp(-dot(p.xz,p.xz)/max(s*s*1.85,1e-4));
      let boards=.86+.14*cos(p.x*22.0);
      return EnvironmentSample(vec3f(.010,.010,.011)+vec3f(1.0,.94,.84)*pool*.22*boards,floorT);
    }
    return EnvironmentSample(color,65504.0);
  }
  let roomHalf=envRoomHalf();let center=vec3f(0.0,environmentFloorY()+roomHalf.y,0.0);let h=envBoxHit(ro,rd,center-roomHalf,center+roomHalf);var t=h.y;
  if(t<=0.0||t>1e19){return EnvironmentSample(environmentLight(rd),65504.0);}
  let p=ro+rd*t;let q=(p-center)/roomHalf;let aq=abs(q);var n=vec3f(0.0);
  if(aq.x>=aq.y&&aq.x>=aq.z){n=vec3f(-sign(q.x),0,0);}else if(aq.y>=aq.z){n=vec3f(0,-sign(q.y),0);}else{n=vec3f(0,0,-sign(q.z));}
  var color=vec3f(0.0);if(e==0){color=conservatoryMaterial(p,n);}else if(e==1){color=courtyardMaterial(p,n);}else if(e==2){color=labMaterial(p,n);}else if(e==3){color=galleryMaterial(p,n);}else if(e==4){color=bathhouseMaterial(p,n);}else{color=stationMaterial(p,n);}
  if(e==2){color=labRoomShade(color,p,n,rd);}
  else{let light=max(dot(n,environmentLightDirection()),0.0);let exposure=select(select(select(select(.26,.52,e==4),.44,e==3),.42,e==1),.40,e==0);color*=exposure*(.70+.30*light)*environmentContactShadow(p,n);}
  return EnvironmentSample(color,t);
}

`;
