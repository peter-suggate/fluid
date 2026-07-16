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
fn envHash21(p:vec2f)->f32{return fract(sin(dot(p,vec2f(127.1,311.7)))*43758.5453);}
fn envHash31(p:vec3f)->f32{return fract(sin(dot(p,vec3f(127.1,311.7,74.7)))*43758.5453);}
// Width is expressed as the retained cell interior (0.46 is a 0.04-wide
// architectural seam). Keeping this convention makes material call sites read
// as panel/tile coverage rather than inverted line thickness.
fn envLine(value:f32,width:f32)->f32{let d=abs(fract(value)-.5);let halfWidth=max(.002,.5-width);return 1.0-smoothstep(halfWidth,halfWidth+.008,d);}
fn envGrid(value:vec2f,width:f32)->f32{return max(envLine(value.x,width),envLine(value.y,width));}
fn envBoxHit(ro:vec3f,rd:vec3f,mn:vec3f,mx:vec3f)->vec2f{let inv=1.0/rd;let a=(mn-ro)*inv;let b=(mx-ro)*inv;let n=min(a,b);let f=max(a,b);return vec2f(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
// Slab test with a caller-hoisted reciprocal: cluster culls and occlusion
// sweeps issue many box tests per ray, and the divides dominate otherwise.
fn envSlabHit(ro:vec3f,inv:vec3f,mn:vec3f,mx:vec3f)->vec2f{let a=(mn-ro)*inv;let b=(mx-ro)*inv;let n=min(a,b);let f=max(a,b);return vec2f(max(max(n.x,n.y),n.z),min(min(f.x,f.y),f.z));}
fn envSlabBlocks(ro:vec3f,inv:vec3f,mn:vec3f,mx:vec3f,maxT:f32)->bool{let hit=envSlabHit(ro,inv,mn,mx);return hit.x<=hit.y&&hit.x>.001&&hit.x<maxT;}
fn envMiss()->EnvironmentPropHit{return EnvironmentPropHit(1e20,vec3f(0,1,0),vec3f(0),0.0);}
fn envNearest(best:EnvironmentPropHit,candidate:EnvironmentPropHit)->EnvironmentPropHit{
  if(candidate.t<best.t){return candidate;}
  return best;
}
fn envBoxPrimitive(ro:vec3f,rd:vec3f,center:vec3f,halfSize:vec3f,color:vec3f,emission:f32)->EnvironmentPropHit{
  let h=envBoxHit(ro,rd,center-halfSize,center+halfSize);let t=select(h.x,h.y,h.x<=.001);if(h.x>h.y||t<=.001){return envMiss();}let p=ro+rd*t-center;let q=abs(p/max(halfSize,vec3f(1e-5)));var n=vec3f(0,1,0);if(q.x>=q.y&&q.x>=q.z){n=vec3f(sign(p.x),0,0);}else if(q.y>=q.z){n=vec3f(0,sign(p.y),0);}else{n=vec3f(0,0,sign(p.z));}return EnvironmentPropHit(t,n,color,emission);
}
fn envBoxPrimitiveInv(ro:vec3f,rd:vec3f,inv:vec3f,center:vec3f,halfSize:vec3f,color:vec3f,emission:f32)->EnvironmentPropHit{
  let h=envSlabHit(ro,inv,center-halfSize,center+halfSize);let t=select(h.x,h.y,h.x<=.001);if(h.x>h.y||t<=.001){return envMiss();}let p=ro+rd*t-center;let q=abs(p/max(halfSize,vec3f(1e-5)));var n=vec3f(0,1,0);if(q.x>=q.y&&q.x>=q.z){n=vec3f(sign(p.x),0,0);}else if(q.y>=q.z){n=vec3f(0,sign(p.y),0);}else{n=vec3f(0,0,sign(p.z));}return EnvironmentPropHit(t,n,color,emission);
}
fn envEllipsoidPrimitive(ro:vec3f,rd:vec3f,center:vec3f,radius:vec3f,color:vec3f,emission:f32)->EnvironmentPropHit{
  let o=(ro-center)/radius;let d=rd/radius;let a=dot(d,d);let b=dot(o,d);let c=dot(o,o)-1.0;let disc=b*b-a*c;if(disc<0.0){return envMiss();}var t=(-b-sqrt(disc))/a;if(t<=.001){t=(-b+sqrt(disc))/a;}if(t<=.001){return envMiss();}let p=ro+rd*t-center;return EnvironmentPropHit(t,normalize(p/(radius*radius)),color,emission);
}
fn envCylinderPrimitive(ro:vec3f,rd:vec3f,center:vec3f,radius:f32,halfHeight:f32,color:vec3f,emission:f32)->EnvironmentPropHit{
  let o=ro-center;let a=dot(rd.xz,rd.xz);let b=dot(o.xz,rd.xz);let c=dot(o.xz,o.xz)-radius*radius;var best=envMiss();if(a>1e-7&&b*b-a*c>=0.0){let root=sqrt(b*b-a*c);var t=(-b-root)/a;if(t<=.001){t=(-b+root)/a;}let y=o.y+rd.y*t;if(t>.001&&abs(y)<=halfHeight){let p=o+rd*t;best=EnvironmentPropHit(t,normalize(vec3f(p.x,0,p.z)),color,emission);}}
  if(abs(rd.y)>1e-7){for(var side=-1.0;side<=1.0;side+=2.0){let t=(side*halfHeight-o.y)/rd.y;let p=o+rd*t;if(t>.001&&t<best.t&&dot(p.xz,p.xz)<=radius*radius){best=EnvironmentPropHit(t,vec3f(0,side,0),color,emission);}}}return best;
}
fn shadeEnvironmentProp(hit:EnvironmentPropHit,ro:vec3f,rd:vec3f)->vec3f{
  if(environmentIndex()==2){return labPropShade(hit,ro+rd*hit.t,rd);}
  let l=environmentLightDirection();let diffuse=.20+.80*max(dot(hit.normal,l),0.0);let rim=pow(1.0-max(dot(-rd,hit.normal),0.0),3.0);let spec=pow(max(dot(reflect(rd,hit.normal),l),0.0),72.0);return hit.color*diffuse+environmentAccent()*rim*.13+environmentLightColor()*(spec*.28+hit.emission);}

fn sampleEnvironmentProps(ro:vec3f,rd:vec3f)->EnvironmentPropHit{
  let e=environmentIndex();let s=max(max(u.container.x,u.container.y),u.container.z);var h=envMiss();
  if(e==0){
    // Conservatory: painted steel glazing, a slatted bench, stone planter,
    // layered foliage, and pendant glass globes.
    let frame=vec3f(.18,.28,.21);for(var i=-1;i<=1;i+=1){let x=f32(i)*1.12*s;h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,.92*s,-1.48*s),vec3f(.027*s,.92*s,.027*s),frame,0));}
    h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(0,.62*s,-1.48*s),vec3f(1.18*s,.025*s,.027*s),frame,0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(0,1.26*s,-1.48*s),vec3f(1.18*s,.025*s,.027*s),frame,0));
    let wood=vec3f(.34,.23,.12);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-1.18*s,.31*s,-.70*s),vec3f(.52*s,.055*s,.20*s),wood,0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-1.18*s,.58*s,-.87*s),vec3f(.52*s,.26*s,.045*s),wood*.82,0));for(var i=-1;i<=1;i+=2){h=envNearest(h,envBoxPrimitive(ro,rd,vec3f((-1.18+.38*f32(i))*s,.15*s,-.70*s),vec3f(.035*s,.16*s,.15*s),wood*.65,0));}
    let stone=vec3f(.38,.35,.26);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(1.12*s,.24*s,-.86*s),vec3f(.28*s,.24*s,.28*s),stone,0));let leaf=vec3f(.055,.30,.14);h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(1.12*s,.62*s,-.86*s),vec3f(.42*s,.38*s,.30*s),leaf,0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(.90*s,.82*s,-.84*s),vec3f(.26*s,.38*s,.20*s),leaf*.82,0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(1.32*s,.88*s,-.90*s),vec3f(.24*s,.42*s,.19*s),leaf*.72,0));
    for(var i=-1;i<=1;i+=1){h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(f32(i)*.56*s,1.50*s,-1.04*s),.012*s,.34*s,vec3f(.12),0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(f32(i)*.56*s,1.18*s,-1.04*s),vec3f(.095*s),vec3f(.85,.68,.38),.48));}
  }else if(e==1){
    // Courtyard: paired limestone columns, a tiled bench, citrus tree and
    // hand-thrown pots create recognisable Mediterranean depth cues.
    let limestone=vec3f(.62,.52,.38);for(var i=-1;i<=1;i+=2){let x=f32(i)*1.16*s;h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(x,.76*s,-1.34*s),.13*s,.76*s,limestone,0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,1.51*s,-1.34*s),vec3f(.19*s,.055*s,.19*s),limestone*.9,0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,.055*s,-1.34*s),vec3f(.20*s,.055*s,.20*s),limestone*.75,0));}
    let tile=vec3f(.46,.20,.12);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-1.12*s,.27*s,-.54*s),vec3f(.50*s,.065*s,.18*s),tile,0));for(var i=-1;i<=1;i+=2){h=envNearest(h,envBoxPrimitive(ro,rd,vec3f((-1.12+.38*f32(i))*s,.13*s,-.54*s),vec3f(.045*s,.14*s,.14*s),tile*.7,0));}
    h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(1.12*s,.22*s,-.68*s),.27*s,.22*s,vec3f(.56,.23,.12),0));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(1.12*s,.70*s,-.68*s),.045*s,.40*s,vec3f(.20,.12,.055),0));let citrus=vec3f(.12,.34,.10);h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(1.12*s,1.12*s,-.68*s),vec3f(.48*s,.42*s,.40*s),citrus,0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(.86*s,1.20*s,-.70*s),vec3f(.27*s),citrus*.82,0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(1.35*s,1.26*s,-.65*s),vec3f(.25*s),citrus*.9,0));for(var i=-1;i<=1;i+=2){h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f((1.12+.16*f32(i))*s,1.12*s,-.42*s),vec3f(.045*s),vec3f(.92,.46,.06),.08));}
  }else if(e==2){
    // Research lab at night: the tank rests on a steel-framed bench above a
    // dropped floor. A task lamp, stool, back counter with instruments and
    // emissive ceiling troffers are all real depth-tested geometry, so the
    // water refracts furniture instead of a backplate. The furniture is
    // grouped under three cluster slabs so a ray pays for a group's
    // primitives only when it actually enters that group's bounds.
    let floorY=environmentFloorY();let th=labTableHalf();let zb=labBenchZ();let ceilY=floorY+2.0*envRoomHalf().y;
    let steel=vec3f(.30,.32,.34);let inv=1.0/rd;
    let bench=envSlabHit(ro,inv,vec3f(-th.x,floorY,-th.y),vec3f(th.x+.63*s,.71*s,th.y));
    if(bench.x<=bench.y&&bench.y>.001&&bench.x<h.t){
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(0.0,-.021*s,0.0),vec3f(th.x,.019*s,th.y),vec3f(.35,.26,.17),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(0.0,-.074*s,0.0),vec3f(th.x-.07*s,.034*s,th.y-.07*s),vec3f(.125,.135,.145),0));
      for(var i=-1;i<=1;i+=2){for(var j=-1;j<=1;j+=2){h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(f32(i)*(th.x-.10*s),floorY+.34*s,f32(j)*(th.y-.10*s)),vec3f(.027*s,.34*s,.027*s),steel,0));}}
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(0.0,floorY+.16*s,0.0),vec3f(th.x-.13*s,.013*s,th.y-.13*s),vec3f(.20,.21,.22),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(-.35*th.x,floorY+.235*s,.15*th.y),vec3f(.15*s,.062*s,.115*s),vec3f(.10,.155,.165),0));
      let lamp=labLampPosition();
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(lamp.x,.010*s,lamp.z),.085*s,.013*s,vec3f(.095,.10,.11),0));
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(lamp.x,.31*s,lamp.z),.012*s,.29*s,vec3f(.14,.15,.16),0));
      h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(lamp.x,.63*s,lamp.z),vec3f(.098*s,.072*s,.098*s),vec3f(.055,.058,.062),0));
      h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(lamp.x,.585*s,lamp.z),vec3f(.046*s),vec3f(1.0,.78,.45),2.8));
      let stoolX=th.x+.45*s;
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(stoolX,floorY+.47*s,.10*s),.17*s,.024*s,vec3f(.235,.155,.095),0));
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(stoolX,floorY+.235*s,.10*s),.024*s,.215*s,steel*.8,0));
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(stoolX,floorY+.02*s,.10*s),.145*s,.014*s,steel*.6,0));
    }
    let counter=envSlabHit(ro,inv,vec3f(-1.82*s,floorY,zb-.36*s),vec3f(1.82*s,floorY+1.90*s,zb+.32*s));
    if(counter.x<=counter.y&&counter.y>.001&&counter.x<h.t){
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(0.0,floorY+.42*s,zb),vec3f(1.72*s,.42*s,.30*s),vec3f(.165,.195,.215),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(0.0,floorY+.862*s,zb),vec3f(1.80*s,.022*s,.34*s),vec3f(.54,.54,.52),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(.95*s,floorY+.93*s,zb),vec3f(.05*s,.055*s,.05*s),vec3f(.06,.065,.07),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(.95*s,floorY+1.17*s,zb+.05*s),vec3f(.30*s,.19*s,.014*s),vec3f(.030,.034,.040),0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(.95*s,floorY+1.17*s,zb+.068*s),vec3f(.265*s,.155*s,.004*s),vec3f(.25,.45,.58),1.0));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(.95*s,floorY+.892*s,zb+.22*s),vec3f(.19*s,.007*s,.07*s),vec3f(.085,.09,.10),0));
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(-.58*s,floorY+.966*s,zb),.070*s,.082*s,vec3f(.30,.36,.39),.02));
      h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(-.86*s,floorY+1.014*s,zb+.04*s),.046*s,.13*s,vec3f(.27,.33,.36),.02));
      h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(-1.12*s,floorY+.980*s,zb-.02*s),vec3f(.088*s,.096*s,.088*s),vec3f(.28,.35,.34),.02));
      h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(-.20*s,floorY+1.62*s,zb-.05*s),vec3f(1.22*s,.016*s,.17*s),vec3f(.29,.30,.31),0));
      for(var i=0;i<3;i+=1){let bx=(-.92+.13*f32(i))*s;var bc=vec3f(.36,.14,.10);if(i==1){bc=vec3f(.10,.235,.255);}else if(i==2){bc=vec3f(.27,.27,.26);}h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(bx,floorY+1.751*s,zb-.05*s),vec3f(.050*s,.115*s,.135*s),bc,0));}
    }
    let fixtures=envSlabHit(ro,inv,vec3f(-1.52*s,ceilY-.05*s,-.52*s),vec3f(1.52*s,ceilY-.02*s,1.17*s));
    if(fixtures.x<=fixtures.y&&fixtures.y>.001&&fixtures.x<h.t){
      for(var i=-1;i<=1;i+=2){for(var j=0;j<2;j+=1){let jz=select(-.30*s,.95*s,j==1);h=envNearest(h,envBoxPrimitiveInv(ro,rd,inv,vec3f(f32(i)*.95*s,ceilY-.035*s,jz),vec3f(.55*s,.012*s,.20*s),vec3f(.92,.93,.90),2.3));}}
    }
  }else if(e==3){
    // Gallery: board-formed portal, long bench, two plinths and contrasting
    // abstract works. All silhouettes are actual depth-tested geometry.
    let portal=vec3f(.82,.39,.17);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-.74*s,.82*s,-1.42*s),vec3f(.045*s,.82*s,.08*s),portal,.12));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(.74*s,.82*s,-1.42*s),vec3f(.045*s,.82*s,.08*s),portal,.12));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(0,1.60*s,-1.42*s),vec3f(.78*s,.045*s,.08*s),portal,.12));
    let bench=vec3f(.12,.14,.13);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-1.08*s,.27*s,-.42*s),vec3f(.55*s,.065*s,.18*s),bench,0));for(var i=-1;i<=1;i+=2){h=envNearest(h,envBoxPrimitive(ro,rd,vec3f((-1.08+.40*f32(i))*s,.14*s,-.42*s),vec3f(.04*s,.14*s,.13*s),bench*.7,0));}
    let plinth=vec3f(.43,.43,.40);h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(1.08*s,.25*s,-.86*s),vec3f(.28*s,.25*s,.28*s),plinth,0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(1.08*s,.72*s,-.86*s),vec3f(.30*s,.42*s,.23*s),vec3f(.08,.12,.11),0));h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(.94*s,.97*s,-.84*s),vec3f(.18*s,.28*s,.16*s),vec3f(.40,.17,.10),0));
  }else if(e==4){
    // Bathhouse: cedar screen structure, stone stools, a soaking bucket and
    // softly lit paper lanterns establish scale around the water vessel.
    let cedar=vec3f(.34,.20,.105);for(var i=-2;i<=2;i+=1){h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(f32(i)*.52*s,.84*s,-1.34*s),vec3f(.025*s,.84*s,.035*s),cedar,0));}for(var i=0;i<=2;i+=1){h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(0,(.22+.53*f32(i))*s,-1.34*s),vec3f(1.18*s,.025*s,.035*s),cedar*.82,0));}
    let stone=vec3f(.24,.25,.22);h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(-1.08*s,.18*s,-.42*s),.27*s,.18*s,stone,0));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(1.06*s,.16*s,-.70*s),.24*s,.16*s,stone*.82,0));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(-.70*s,.20*s,-1.02*s),.25*s,.20*s,vec3f(.42,.25,.12),0));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(-.70*s,.42*s,-1.02*s),.19*s,.035*s,cedar,0));
    for(var i=-1;i<=1;i+=2){let x=f32(i)*.68*s;h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,1.22*s,-1.18*s),vec3f(.12*s,.19*s,.12*s),vec3f(.72,.57,.34),.22));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(x,1.48*s,-1.18*s),.008*s,.09*s,cedar,0));}
  }else if(e==5){
    // Research station: pressure ribs, side consoles, exposed pipes, equipment
    // cases and emissive monitor glass make the chamber feel occupied.
    let metal=vec3f(.025,.09,.115);for(var i=-2;i<=2;i+=1){let x=f32(i)*.58*s;h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,1.35*s,-1.42*s),vec3f(.025*s,1.35*s,.05*s),metal,0));}
    for(var i=-1;i<=1;i+=2){let x=f32(i)*1.16*s;h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,.37*s,-.72*s),vec3f(.34*s,.37*s,.30*s),metal*.72,0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(x,.63*s,-.40*s),vec3f(.25*s,.13*s,.018*s),vec3f(.06,.48,.58),.30));h=envNearest(h,envCylinderPrimitive(ro,rd,vec3f(x+.25*s,.88*s,-1.06*s),.055*s,.72*s,vec3f(.12,.25,.27),0));}
    h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-.76*s,.23*s,-1.04*s),vec3f(.30*s,.23*s,.24*s),vec3f(.12,.15,.15),0));h=envNearest(h,envBoxPrimitive(ro,rd,vec3f(-.76*s,.47*s,-1.04*s),vec3f(.25*s,.018*s,.19*s),vec3f(.74,.48,.16),.12));
    for(var i=-1;i<=1;i+=1){h=envNearest(h,envEllipsoidPrimitive(ro,rd,vec3f(f32(i)*.52*s,1.36*s,-1.12*s),vec3f(.055*s),vec3f(.10,.65,.72),.36));}
  }
  return h;
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
fn labPropShade(hit:EnvironmentPropHit,p:vec3f,rd:vec3f)->vec3f{
  let n=hit.normal;
  let s=max(max(u.container.x,u.container.y),u.container.z);
  var c=hit.color*(vec3f(.060,.065,.078)+labKeyLights(p,n,n.y>.55))*labTankShadow(p,n);
  c+=hit.color*hit.emission*2.6;
  let ceilY=environmentFloorY()+2.0*envRoomHalf().y;
  let l=normalize(vec3f(select(-.95,.95,p.x>0.0)*s,ceilY-.16*s,-.30*s)-p);
  c+=vec3f(1.0,.97,.90)*pow(max(dot(reflect(rd,n),l),0.0),60.0)*.30;
  c+=vec3f(.05,.07,.09)*pow(1.0-max(dot(-rd,n),0.0),4.0);
  return c;
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
  return normalize(vec3f(-.45,.86,.28));
}
fn environmentLightColor()->vec3f{
  let e=environmentIndex();
  if(e==6){return vec3f(1.0,.86,.66);}
  if(e==1){return vec3f(1.0,.77,.52);}
  if(e==2){return vec3f(1.0,.94,.80);}
  if(e==3){return vec3f(1.0,.67,.40);}
  if(e==4){return vec3f(1.0,.83,.57);}
  if(e==5){return vec3f(.42,.83,1.0);}
  return vec3f(1.0,.86,.62);
}
fn environmentAccent()->vec3f{
  let e=environmentIndex();
  if(e==6){return vec3f(.18,.34,.31);}
  if(e==1){return vec3f(.10,.34,.44);}
  if(e==2){return vec3f(.08,.11,.15);}
  if(e==3){return vec3f(.72,.42,.22);}
  if(e==4){return vec3f(.54,.39,.25);}
  if(e==5){return vec3f(.14,.56,.68);}
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
  else if(e==0){c=mix(vec3f(.035,.055,.044),vec3f(.46,.58,.43),t);}
  else if(e==1){c=mix(vec3f(.16,.16,.13),vec3f(.68,.69,.59),t);}
  else if(e==2){c=mix(vec3f(.016,.017,.020),vec3f(.065,.068,.078),t);c+=vec3f(.30,.30,.27)*pow(max(rd.y,0.0),6.0)+vec3f(.05,.09,.16)*pow(max(-rd.z,0.0),4.0)*.6;}
  else if(e==3){c=mix(vec3f(.045,.050,.048),vec3f(.30,.32,.30),t);}
  else if(e==4){c=mix(vec3f(.045,.038,.032),vec3f(.34,.31,.25),t);}
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
  if(e==6){
    let t=clamp(rd.y*.5+.5,0.0,1.0);var color=mix(vec3f(.015,.027,.029),vec3f(.16,.23,.22),t);let sun=max(dot(rd,normalize(vec3f(-.45,.86,.28))),0.0);color+=vec3f(1.0,.86,.66)*pow(sun,320.0)*2.2+vec3f(.24,.31,.28)*pow(sun,12.0);
    let floorT=(-.012-ro.y)/rd.y;if(floorT>0.0){let p=ro+rd*floorT;let radial=length(p.xz);let checker=.5+.5*cos(p.x*31.4)*cos(p.z*31.4);color=mix(color,vec3f(.055,.068,.064)+checker*vec3f(.018,.025,.022),.82*exp(-radial*.22));return EnvironmentSample(color,floorT);}
    return EnvironmentSample(color,65504.0);
  }
  let roomHalf=envRoomHalf();let center=vec3f(0.0,environmentFloorY()+roomHalf.y,0.0);let h=envBoxHit(ro,rd,center-roomHalf,center+roomHalf);var t=h.y;
  if(t<=0.0||t>1e19){return EnvironmentSample(environmentLight(rd),65504.0);}
  // Props resolve before the room surface: a furniture hit makes the room
  // material and its shadow/reflection rays dead work, so skip them entirely.
  let prop=sampleEnvironmentProps(ro,rd);if(prop.t<t){return EnvironmentSample(shadeEnvironmentProp(prop,ro,rd),prop.t);}
  let p=ro+rd*t;let q=(p-center)/roomHalf;let aq=abs(q);var n=vec3f(0.0);
  if(aq.x>=aq.y&&aq.x>=aq.z){n=vec3f(-sign(q.x),0,0);}else if(aq.y>=aq.z){n=vec3f(0,-sign(q.y),0);}else{n=vec3f(0,0,-sign(q.z));}
  var color=vec3f(0.0);if(e==0){color=conservatoryMaterial(p,n);}else if(e==1){color=courtyardMaterial(p,n);}else if(e==2){color=labMaterial(p,n);}else if(e==3){color=galleryMaterial(p,n);}else if(e==4){color=bathhouseMaterial(p,n);}else{color=stationMaterial(p,n);}
  if(e==2){color=labRoomShade(color,p,n,rd);}
  else{let light=max(dot(n,environmentLightDirection()),0.0);let exposure=select(select(select(select(.26,.52,e==4),.44,e==3),.42,e==1),.40,e==0);color*=exposure*(.70+.30*light)*environmentContactShadow(p,n);}
  return EnvironmentSample(color,t);
}

fn envEllipse(p:vec2f,center:vec2f,radius:vec2f,angle:f32)->f32{let cs=cos(angle);let sn=sin(angle);let d=p-center;let q=vec2f(cs*d.x+sn*d.y,-sn*d.x+cs*d.y)/radius;return 1.0-smoothstep(.82,1.02,length(q));}
fn envStroke(distance:f32,width:f32)->f32{return 1.0-smoothstep(width,width+.008,abs(distance));}

fn environmentForeground(color:vec3f,ndc:vec2f)->vec3f{
  let e=environmentIndex();var c=color;
  if(e==0){
    var leaves=envEllipse(ndc,vec2f(-.94,.68),vec2f(.22,.085),-.72)+envEllipse(ndc,vec2f(-.83,.84),vec2f(.20,.078),-.22)+envEllipse(ndc,vec2f(.92,-.70),vec2f(.27,.10),.58)+envEllipse(ndc,vec2f(.82,-.88),vec2f(.22,.09),.08);leaves=clamp(leaves,0.0,1.0);let branch=envStroke(ndc.y+.78-.40*ndc.x,.012);c=mix(c,vec3f(.018,.075,.041),max(leaves*.48,branch*.30));c+=leaves*vec3f(.020,.065,.032);
  }else if(e==1){
    let corner=smoothstep(.48,.72,ndc.x)*smoothstep(.36,.64,ndc.y);let leaf=envEllipse(ndc,vec2f(.91,.78),vec2f(.18,.066),.72)+envEllipse(ndc,vec2f(.82,.91),vec2f(.15,.057),.28);let branch=envStroke(ndc.y-.53*ndc.x-.30,.009)*corner;let fruit=1.0-smoothstep(.036,.048,length(ndc-vec2f(.88,.72)));c=mix(c,vec3f(.055,.105,.048),clamp(leaf*.44+branch*.26,0.0,1.0));c=mix(c,vec3f(.88,.48,.11),fruit*.58);
  }else if(e==2){
    let edge=smoothstep(.88,1.25,max(abs(ndc.x),abs(ndc.y)));c*=1.0-.28*edge;
  }else if(e==3){
    let slab=smoothstep(.68,.74,ndc.x)*(1.0-smoothstep(-.76,-.67,ndc.y));let rail=envStroke(ndc.y+.86,.018);c=mix(c,vec3f(.025,.029,.027),max(slab*.56,rail*.42));let dust=envHash21(floor((ndc+vec2f(u.viewport.z*.002,0))*vec2f(410,260)));c+=vec3f(.18,.13,.08)*select(0.0,.08,dust>.997);
  }else if(e==4){
    let post=smoothstep(.92,.88,abs(ndc.x));let cloth=smoothstep(.68,.74,ndc.x)*smoothstep(.38,.46,ndc.y);let hem=envStroke(ndc.y-.38-.018*sin(ndc.x*14.0),.010)*smoothstep(.68,.75,ndc.x);c=mix(c,vec3f(.055,.035,.022),post*.44);c=mix(c,vec3f(.30,.20,.12),cloth*.42);c+=hem*vec3f(.22,.15,.08);
  }else if(e==5){
    let radius=length(ndc*vec2f(1.0,u.viewport.y/max(u.viewport.x,1.0)));let frame=smoothstep(.83,.91,radius);let rib=max(envStroke(abs(ndc.x)-.93,.025),envStroke(abs(ndc.y)-.91,.025));c=mix(c,vec3f(.003,.014,.024),max(frame*.58,rib*.48));let drift=envHash21(floor((ndc+vec2f(u.viewport.z*.006,-u.viewport.z*.003))*vec2f(330,210)));c+=vec3f(.20,.62,.72)*select(0.0,.20,drift>.996);
  }
  return c;
}
`;
