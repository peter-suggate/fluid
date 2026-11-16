/** NB-FLIP's particle formulas as pure WGSL (lane F; design docs/plans/nbflip-structural-2026-11-15.md §2.4).
 *
 * Every function here is ported verbatim from the uniform engine's resolved WGSL (the text the old engine compiles
 * for the gate scenes, captured at a22d512d), with its addressing taken out: a value the old code loaded through the
 * page table, a texture or a bin run is an argument here, or a call to an `nbx*` function the including kernel
 * defines (staged workgroup memory in a particle kernel, T2o in a cell kernel). Nothing in this file reads a
 * binding. The expressions keep the old operand order, so a kernel that feeds the same inputs gets the same bits
 * (`particles-formulas.test.ts` checks each against the old text on a frozen state).
 *
 * The papers each formula follows are cited where the old engine cited them; nothing here changes a formula.
 *
 * Settings are the gate scenes' (every one resolves the same NB-FLIP values): particleTransfer "flip",
 * crowdedCells "merge", surfaceSamples 8 (3 mass bits), particleDensity "graded", sampleSeparation "pairs",
 * particleBandDepth 3, heatExpiry "held". A scene with other values fails at build (`nbfParticleSupported`). */

/** The NB-FLIP values these formulas are written for. Unknown or other values must fail explicitly. */
export interface NbfParticleSettings {
  readonly particleTransfer: string; readonly crowdedCells: string; readonly surfaceSamples: number;
  readonly particleDensity: string; readonly sampleSeparation: string; readonly particleBandDepth: number;
  readonly heatExpiry: string;
}
export const NBF_PARTICLE_SETTINGS: NbfParticleSettings = {
  particleTransfer: "flip", crowdedCells: "merge", surfaceSamples: 8, particleDensity: "graded",
  sampleSeparation: "pairs", particleBandDepth: 3, heatExpiry: "held",
};
/** Throws naming the first setting the brick engine's particle kernels do not implement. */
export function nbfParticleSupported(values: Readonly<Record<string, unknown>>): void {
  for (const [key, want] of Object.entries(NBF_PARTICLE_SETTINGS)) {
    const got = values[key];
    if (got !== undefined && got !== want) throw new Error(`NB-FLIP brick engine: ${key} "${String(got)}" is not implemented (only "${String(want)}")`);
  }
}

/** The particle record (narrowBandParticleWGSL(false, true, 3)): FLIP, merge, 3 mass bits. before.w packs the
 * ballistic bit (31), the sparse age (bits 3..30) and mass - 1 (bits 0..2). */
export const nbfParticleRecordWGSL = /* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
fn nbParticleMode(p:Particle)->f32{return f32(bitcast<u32>(p.before.w)>>31u);}
fn nbParticleAge(p:Particle)->f32{return bitcast<f32>(bitcast<u32>(p.before.w)&0x7ffffff8u);}
fn nbParticleState(age:f32,mode:f32)->f32{return bitcast<f32>((bitcast<u32>(age)&0x7ffffff8u)|(u32(mode)<<31u));}
fn nbParticleMass(p:Particle)->f32{return f32((bitcast<u32>(p.before.w)&7u)+1u);}
fn nbParticleRestate(p:Particle,age:f32,mode:f32)->f32{return bitcast<f32>(bitcast<u32>(nbParticleState(age,mode))|(bitcast<u32>(p.before.w)&7u));}
fn nbWithMass(p:Particle,mass:f32)->Particle{var r=p;r.before.w=bitcast<f32>((bitcast<u32>(p.before.w)&0xfffffff8u)|((u32(mass)-1u)&7u));return r;}
fn nbLive(p:Particle)->bool{return p.position.w>0.0;}
fn nbDead()->Particle{var p=Particle();p.position=vec4f(-1.0,0.0,0.0,0.0);return p;}
// A deferred step seed's age until its first move: the largest finite f32 less seven ulps, its low three bits
// clear for a merge's mass (particleSlabWGSL).
const NB_SEED_FIRST:u32=0x7f7ffff8u;
fn nbSeedFirst()->f32{return bitcast<f32>(NB_SEED_FIRST);}
`;

/** nbfParticleFormulasWGSL with the band bandDepth cells deep (the scene's brickTiers.particleBandDepth): records
 * retire past it and seeding stops there. A depth outside [2, 4] cells, or not finite, fails. */
export function nbfParticleFormulasAt(bandDepth: number): string {
  if (!(Number.isFinite(bandDepth) && bandDepth >= 2 && bandDepth <= 4)) throw new Error(`NB-FLIP brick engine: particle band depth ${String(bandDepth)} is not in [2, 4] cells`);
  const anchor = "const NB_BAND_DEPTH:f32=3.00;";
  if (nbfParticleFormulasWGSL.split(anchor).length !== 2) throw new Error("particles: NB_BAND_DEPTH changed shape");
  return nbfParticleFormulasWGSL.replace(anchor, `const NB_BAND_DEPTH:f32=${bandDepth.toFixed(2)};`);
}

/** Constants and the pure move formulas. */
export const nbfParticleFormulasWGSL = /* wgsl */`
const NB_BAND_DEPTH:f32=3.00;
const NB_GRADED:bool=true;
const NB_MASS_MAX:f32=8.0;
// Ferstl et al. 2016 Section 3.3: no more than 2n unit samples (n = 8) a resampled cell.
const NB_CELL_MASS:f32=16.0;
// A sample's radius r = cbrt(m)/4 for the integer masses it holds: eight unit samples a cell rest half a cell apart.
const NB_SAMPLE_RADIUS=array<f32,9>(0.0,0.25,0.31498,0.36056,0.39685,0.42750,0.45428,0.47823,0.5);
fn nbSampleRadius(m:f32)->f32{return NB_SAMPLE_RADIUS[min(u32(m),8u)];}
// The cap of a deep cell (centre depth): six in a graded band's cells past 1.5. cap: particleCellCap.
fn nbfCellCap(depth:f32,cap:f32)->u32{return select(u32(cap),min(u32(cap),6u),NB_GRADED&&depth<=-1.5);}

// One step of Heun's third-order rule over the whole dt (nbAdvect): stage 0 the field at q (k1, and G2P's grid
// velocity pic), 1 a third of the step along it (k2), 2 two thirds along k2 (k3); the end is q + dt(k1 + 3k3)/4.
// h: the cell size an axis (m); velocities in m/s, positions in cells.
fn nbfHeunK1(pic:vec3f,h:vec3f)->vec3f{return pic/h;}
fn nbfHeunStage1(q:vec3f,k1:vec3f,dt:f32)->vec3f{return q+dt*k1/3.0;}
fn nbfHeunStage2(q:vec3f,k:vec3f,dt:f32,h:vec3f)->vec3f{return q+2.0*dt*k/(3.0*h);}
fn nbfHeunEnd(q:vec3f,k1:vec3f,k:vec3f,dt:f32,h:vec3f)->vec3f{return q+0.25*dt*(k1+3.0*k/h);}
// A record whose trace stops at G2P drifts along pic (coupled: its mode is not ballistic) or its own velocity.
fn nbfDrift(q:vec3f,v:vec3f,pic:vec3f,coupled:bool,dt:f32,h:vec3f)->vec3f{return q+dt*select(v,pic,coupled)/h;}

// G2P with the FLIP blend (Zhu and Bridson 2005; FLIP ratio params.settings.x = 0.95): the last step's end
// against the field that step projected (pic) and the FLIP delta the tail recorded (d = before.xyz; an axis of
// 1e30 or more has none and takes PIC). A COLD sample takes no delta; a seed's first move takes the grid's.
fn nbfG2P(v:vec3f,d:vec3f,pic:vec3f,cold:bool,first:bool,ratio:f32)->vec3f{
 var flip=v;if(!cold){flip=select(pic,v+d,abs(d)<vec3f(1e30));}return select(mix(pic,flip,ratio),pic,first);
}
// A ballistic sample takes a step of gravity; solidAhead[a]: the cell ahead along a (nbfBallisticAhead) is solid.
fn nbfBallisticKick(v:vec3f,gravity:vec3f,dt:f32)->vec3f{return v+gravity*dt;}
fn nbfBallisticAhead(q:vec3f,v:vec3f,m:f32,axis:u32)->vec3f{var ahead=q;ahead[axis]+=select(-1.0,1.0,v[axis]>0.0)*(0.001+nbSampleRadius(m));return ahead;}
fn nbfBallisticStop(v:vec3f,axis:u32,solidAhead:bool)->bool{return (v[axis]<0.0||v[axis]>0.0)&&solidAhead;}
fn nbfFinite(v:vec3f,bound:f32)->bool{return all(abs(v)<vec3f(bound));}

// Erasure heat at G2P (nbAdvect): a sample takes its tile's target heat, capped at 1 unless it already carries
// erasure heat (over 1); a grid-supported sample cools at params.activity.x a second. heatExpiry "held": a
// shallow sample in a thin tile keeps heat 1.
fn nbfHeatAtG2P(heat:f32,tileHeat:f32,cooling:f32,dt:f32,ballistic:bool,depth:f32,thin:bool)->f32{
 var w=max(select(min(tileHeat,1.0),tileHeat,heat>1.0),max(0.0,heat-select(cooling*dt,0.0,ballistic)));
 if(depth> -1.0&&thin){w=max(w,1.0);}
 return w;
}
// A grid-supported sample's arrival heat (nbFinish, collarSpread "neighbours").
fn nbfArrivalHeat(heat:f32,arrived:f32)->f32{if(arrived>0.0){return min(heat,max(arrived,1.0));}return heat;}
// Retirement at G2P: deeper than the band or sparse-expired.
fn nbfRetireDeep(depth:f32)->bool{return depth< -NB_BAND_DEPTH;}

// Sparse isolation age (narrowBandSparseWGSL; sparseNeighbours threshold, sparseParticleLifetime lifetime):
// own: the other samples in the sample's cell; block: the samples of the 3x3x3 cells around it, itself included.
// The old code stops counting once the block passes the threshold; the result is the same.
fn nbfSparseAge(age:f32,lifetime:f32,threshold:u32,own:u32,block:u32,dt:f32)->f32{
 if(lifetime<=0.0||threshold==0u){return 0.0;}
 if(own>=threshold){return 0.0;}
 if(block>threshold){return 0.0;}
 return min(lifetime,age+max(dt,0.0));
}
fn nbfSparseExpired(age:f32,lifetime:f32,threshold:u32)->bool{return lifetime>0.0&&threshold!=0u&&age>=lifetime;}

// sampleSeparation "pairs": Sato et al. 2018 Section 3.2.2 (after Ando et al. 2012): two samples closer than the
// sum of their radii push apart, each by the other's share of their mass, d_ij = m_j/(m_i+m_j)(r_i+r_j-|p_i-p_j|)
// along p_i-p_j, solved as one Jacobi pass with constraint averaging (Macklin et al. 2014, Section 4.2).
// nbfSeparationAdd takes one neighbour (other: its position, in the same frame as q; mj its mass) into the sum
// (d, n) = acc; the caller skips the sample itself and dead records (other.w <= 0). far: nbfSeparationFar(m).
fn nbfSeparationFar(m:f32)->f32{let ri=nbSampleRadius(m);return (ri+0.5)*(ri+0.5);}
fn nbfSeparationAdd(acc:vec4f,q:vec3f,m:f32,far:f32,other:vec3f,mj:f32)->vec4f{
 let ri=nbSampleRadius(m);var d=acc.xyz;var n=acc.w;
 let x=q-other;let l2=dot(x,x);
 if(l2>=far||l2<=1e-12){return acc;}
 let reach=ri+nbSampleRadius(mj);
 if(l2<reach*reach){let l=sqrt(l2);d+=mj/(m+mj)*(reach-l)*x/l;n+=1.0;}
 return vec4f(d,n);
}
fn nbfSeparationFinish(acc:vec4f)->vec3f{return acc.xyz/max(acc.w,1.0);}
// The base vertex of the 2x2x2 cells nbSeparation reads (they hold every sample within half a cell an axis).
fn nbfSeparationBase(q:vec3f)->vec3i{return vec3i(floor(q-0.5));}

// The free surface (nbSurfaceShift): a sample within a cell of it (centre depth over -1) keeps only the tangential
// part of its correction, d - (d.n)n with n the level set's normal (Khayyer et al. 2017; Sun et al. 2019 Eq. 44;
// after Lind et al. 2012). The depth is read at q (k = 0) and the six quarter-cell offsets (nbfSurfaceShiftPoint).
fn nbfSurfaceShiftPoint(q:vec3f,k:u32)->vec3f{
 var x=q;let axis=(k-1u)>>1u;let up=((k-1u)&1u)==0u;
 if(k>0u){x[axis]+=select(-0.25,0.25,up);}
 return x;
}
// centre: the depth at q; the caller returns d unchanged when nbfSurfaceShiftNeeded(centre) is false.
fn nbfSurfaceShiftNeeded(centre:f32)->bool{return !(centre<= -1.0);}
// g accumulates the six offset depths (nbfSurfaceShiftGradient, k = 1..6, in order).
fn nbfSurfaceShiftGradient(g:vec3f,k:u32,depth:f32)->vec3f{var r=g;let axis=(k-1u)>>1u;let up=((k-1u)&1u)==0u;r[axis]+=select(-depth,depth,up);return r;}
fn nbfSurfaceShiftProject(d:vec3f,g:vec3f)->vec3f{
 let l=length(g);if(!(l>1e-6)){return d;}
 let n=g/l;return d-dot(d,n)*n;
}

// crowdedCells "merge" (nbMergeRun): a deep cell's kept rank gathers the excess ranks congruent to it, mass-weighted
// (Ferstl et al. 2016 Section 3.3 bounds the cell's mass). NbfMerge holds the running sums.
struct NbfMerge{m:f32,x:vec3f,v:vec3f,b:vec3f,none:vec3<bool>,heat:f32,age:f32}
fn nbfMergeStart(record:Particle)->NbfMerge{
 let m=nbParticleMass(record);
 return NbfMerge(m,m*record.position.xyz,m*record.velocity.xyz,m*record.before.xyz,abs(record.before.xyz)>=vec3f(1e30),record.position.w,nbParticleAge(record));
}
fn nbfMergeAdd(s:NbfMerge,o:Particle)->NbfMerge{
 if(!nbLive(o)){return s;}
 let mo=nbParticleMass(o);
 if(s.m+mo>NB_MASS_MAX){return s;}
 var r=s;
 r.m+=mo;r.x+=mo*o.position.xyz;r.v+=mo*o.velocity.xyz;r.b+=mo*o.before.xyz;r.none=r.none|(abs(o.before.xyz)>=vec3f(1e30));
 r.heat=max(r.heat,o.position.w);r.age=min(r.age,nbParticleAge(o));
 return r;
}
fn nbfMergeFinish(record:Particle,s:NbfMerge)->Particle{
 var p=record;p.position=vec4f(s.x/s.m,s.heat);p.velocity=vec4f(s.v/s.m,record.velocity.w);
 p.before=vec4f(select(s.b/s.m,vec3f(3e38),s.none),nbParticleState(s.age,nbParticleMode(record)));
 return nbWithMass(p,s.m);
}
// nbMergeAbsorbed: the gatherer of rank k takes ranks k + stride.. below i whose mass fits; i is absorbed when its
// mass still fits after them. m: the running mass (nbfAbsorbStart with the gatherer's mass, nbfAbsorbAdd per rank).
fn nbfAbsorbAdd(m:f32,o:Particle)->f32{if(!nbLive(o)){return m;}let mo=nbParticleMass(o);if(m+mo<=NB_MASS_MAX){return m+mo;}return m;}
fn nbfAbsorbed(m:f32,mi:f32)->bool{return m+mi<=NB_MASS_MAX;}
// The cell's mass bound (Ferstl 2016 3.3 as mass): the run's records past NB_CELL_MASS retire. held: the mass
// before record o; returns the mass after it, or -1 where o is the first record past the bound.
fn nbfCellMassAdd(held:f32,o:Particle)->f32{if(!nbLive(o)){return held;}let m=nbParticleMass(o);if(held>0.0&&held+m>NB_CELL_MASS){return -1.0;}return held+m;}

// A merged sample that ends shallow or in a cell its run left under the seed target splits in two a quarter cell
// apart along a direction hashed from its position (nbFinish). Returns a and b's positions; kept = ceil(m/2).
fn nbfSplitWanted(mass:f32,ballistic:bool,cap:f32,held:u32,seeded:u32)->bool{return mass>1.0&&!ballistic&&cap>0.0&&held<seeded&&mass*f32(seeded)>8.0;}
fn nbfSplitKept(mass:f32)->f32{return ceil(0.5*mass);}
fn nbfSplitAlong(q:vec3f)->vec3f{
 let bits=bitcast<vec3u>(q);let hash=(bits.x*73856093u)^(bits.y*19349663u)^(bits.z*83492791u);
 return normalize(vec3f(f32(hash&255u),f32((hash>>8u)&255u),f32((hash>>16u)&255u))-127.5);
}
fn nbfSplitA(q:vec3f,mass:f32,along:vec3f)->vec3f{let kept=ceil(0.5*mass);let moved=mass-kept;return q+0.25*(moved/mass)*along;}
fn nbfSplitB(q:vec3f,mass:f32,along:vec3f)->vec3f{let kept=ceil(0.5*mass);return q-0.25*(kept/mass)*along;}

// The fixed box (nbTraceEndpointIn with umFixedBox): the centre keeps its radius from each wall.
fn nbfBoxEndpoint(p:vec3f,lo:vec3f,hi:vec3f,inset:f32)->vec3f{return clamp(p,lo+vec3f(inset),hi-vec3f(inset));}
// An axis the endpoint clamp moved loses its velocity (nbFinish).
fn nbfClippedVelocity(v:vec3f,end:vec3f,endpoint:vec3f)->vec3f{var r=v;for(var axis=0u;axis<3u;axis++){if(endpoint[axis]!=end[axis]){r[axis]=0.0;}}return r;}

// The trilinear vertex sample (particleDepthIn, umVelocityLerp8's corner order k = x + 2y + 4z): f the fraction in
// the cell, v the eight vertex values in that order.
fn nbfVertexTrilinear(v:array<f32,8>,f:vec3f)->f32{
 var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k%2u,(k/2u)%2u,k/4u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*v[k];
 }
 return value;
}
// A point outside the band's reach (particleDepthIn): the level set's value, held at least 6 cells from zero.
fn nbfFarDepth(far:f32)->f32{return select(max(far,6.0),min(far,-6.0),far<0.0);}
// MAC face trilinear (umVelocityLerp8): v the eight face values in corner order, f the fraction.
fn nbfFaceLerp8(v:array<f32,8>,f:vec3f)->f32{
 let a=(1.0-f.x)*v[0]+f.x*v[1];let b=(1.0-f.x)*v[2]+f.x*v[3];
 let c=(1.0-f.x)*v[4]+f.x*v[5];let d=(1.0-f.x)*v[6]+f.x*v[7];
 return (1.0-f.z)*((1.0-f.y)*a+f.y*b)+f.z*((1.0-f.y)*c+f.y*d);
}
// The composite sample (umSampleVelocityWeightedIn): fine weight f of the h sample, the rest of the 4h one.
fn nbfCompositeBlend(f:f32,fine:f32,coarse4:f32)->f32{
 var value=0.0;
 if(f>0.0){value+=f*fine;}
 let coarse=max(0.0,1.0-f);
 if(coarse>0.0){value+=coarse*coarse4;}
 return value;
}
// The fine weight near h-owned tiles (umVelocitySamplingWeightsAt's loop body): q the sample in cells, origin a
// fine tile's origin (cells); the weight is the largest over the blend mask's tiles.
fn nbfFineWeight(fine:f32,q:vec3f,origin:vec3f)->f32{
 let delta=max(vec3f(0),max(origin-q,q-origin-vec3f(4.0)));
 let distance=max(delta.x,max(delta.y,delta.z));
 return max(fine,max(0.0,1.0-distance/2.0));
}
`;

/** Seeding (Zhu and Bridson 2005 jittered sub-cell sampling; Ferstl et al. 2016 inner-overlap reseed). */
export const nbfSeedFormulasWGSL = /* wgsl */`
// A seed lies anywhere in its eighth of the cell, hashed from the cell, the site and the seeding salt.
fn nbSeedJitter(c:vec3u,k:u32,salt:u32)->vec3f{
 var h=(c.x*73856093u)^(c.y*19349663u)^(c.z*83492791u)^(k*2654435761u)^(salt*0x9e3779b9u);
 var r=vec3f(0.0);
 for(var a=0u;a<3u;a++){h^=h>>16u;h*=0x7feb352du;h^=h>>15u;h*=0x846ca68bu;h^=h>>16u;r[a]=f32(h&0xffffffu)/16777216.0;}
 return r;
}
fn nbfSeedPoint(c:vec3u,k:u32,salt:u32)->vec3f{return vec3f(c)+0.5*vec3f(vec3u(k%2u,(k/2u)%2u,k/4u))+0.5*nbSeedJitter(c,k,salt);}
// The particle mass a face at depth expects when its samples cover it (narrowBandExpectedMassWGSL, graded with
// merge: the four graded seeds carry two units each, so the expected mass stays eight).
fn nbExpectedMass(depth:f32)->f32{return max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
// Seeds a band cell takes (particleDensity "graded": four below 1.5 cells), and which of the eight sites.
fn nbSeedTarget(depth:f32)->u32{return select(8u,4u,NB_GRADED&&depth<=-1.5);}
fn nbSeedSiteIndex(j:u32,depth:f32)->u32{if(nbSeedTarget(depth)==8u){return j;}return array<u32,4>(0u,3u,5u,6u)[j&3u];}
fn nbfSeedMass(depth:f32)->f32{return f32(8u/nbSeedTarget(depth));}
// seedCell's gate on the cell centre's depth.
fn nbfSeedCellDepthOk(d:f32)->bool{return !(d>0.9||d< -NB_BAND_DEPTH-0.5);}
// seedSite's verdict for a site at depth distance (blocked: a solid holds it). The reseed (initial false) skips a
// site the band already covers: shallow (distance or centre over -1 / -1.5), outside a source (sourcePhi >= 0)
// and with a sample in the 27 cells around (uncovered false). A shallow site needs an h owner (fineOwner).
fn nbfSeedSiteOk(distance:f32,blocked:bool,initial:bool,centre:f32,sourcePhi:f32,uncovered:bool,fineOwner:bool)->bool{
 if(distance>0.0||distance< -NB_BAND_DEPTH||blocked){return false;}
 if(!initial&&(distance>-1.0||centre>-1.5)&&sourcePhi>=0.0&&!uncovered){return false;}
 if(distance>-1.0&&!fineOwner){return false;}
 return true;
}
`;
