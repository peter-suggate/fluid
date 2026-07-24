import { fineLevelSetLinearWorkgroupWGSL } from "./webgpu-fine-levelset-dispatch";
import { makeOctreeFaceBandAirSamplerFunctionsWGSL } from "./webgpu-octree-face-closest-point";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";

export interface FineTransportPackedRegion { readonly offsetBytes: number; readonly sizeBytes: number }
export interface FineTransportPackedSamplerPlan {
  readonly direct: Readonly<Record<"rows" | "rowDirectory" | "velocities" | "tetraHeaders" | "tetraVertices" | "tetrahedra", FineTransportPackedRegion>>;
  readonly air: Readonly<Record<"control" | "rows" | "rowDirectory" | "velocities" | "metrics" | "transition" | "point", FineTransportPackedRegion>>;
  readonly directBytes: number;
  readonly airBytes: number;
}

/** Balanced-bracket rewrite used only while generating WGSL. It turns typed
 * runtime-array reads in the already validated air sampler into producer-arena
 * loader calls without changing the extracted interpolation functions. */
function rewriteIndex(source: string, identifier: string, loader: string): string {
  const needle = `${identifier}[`;
  let at = 0, result = "";
  while (true) {
    const start = source.indexOf(needle, at);
    if (start < 0) return result + source.slice(at);
    result += source.slice(at, start) + `${loader}(`;
    let depth = 1, cursor = start + needle.length;
    for (; cursor < source.length && depth > 0; cursor += 1) {
      const ch = source[cursor];
      if (ch === "[") depth += 1;
      else if (ch === "]") depth -= 1;
      if (depth > 0) result += ch;
    }
    if (depth !== 0) throw new Error(`Unbalanced WGSL index for ${identifier}`);
    result += ")"; at = cursor;
  }
}

export function makeFineLevelSetProductionFusedTransportWGSL(): string {
  let airFunctions = makeOctreeFaceBandAirSamplerFunctionsWGSL();
  const counts: Record<string, string> = {
    rows: "pack.bandRowCount", rowVelocities: "pack.bandVelocityCount", owners: "arrayLength(&owners)",
    metrics: "pack.bandMetricCount", tetraHeaders: "pack.tetraHeaderCount",
    tetrahedra: "pack.tetrahedronCount", tetraVertices: "pack.tetraVertexCount",
  };
  for (const [name, count] of Object.entries(counts)) {
    airFunctions = airFunctions.replaceAll(`arrayLength(&${name})`, count);
  }
  for (const [name, loader] of Object.entries({
    rows: "loadBandRow", rowVelocities: "loadBandVelocity", owners: "loadOwnerWord",
    metrics: "loadBandMetric", tetraHeaders: "loadTetraHeader",
    tetrahedra: "loadTetrahedron", tetraVertices: "loadTetraVertex",
  })) airFunctions = rewriteIndex(airFunctions, name, loader);

  return /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID_FINE:u32=1u;const VELOCITY_VALID:u32=0x80000000u;const LARGE:f32=3.402823e38;
const STATUS_EXTRAPOLATED:u32=0x10000000u;const STATUS_FACE_UNAVAILABLE:u32=0x08000000u;
const VALID:u32=0x80000000u;const ROW_SUPPORT2:u32=16u;const ROW_SUPPORT3_NODE:u32=32u;const ROW_SUPPORT3_ENDPOINT:u32=64u;
const DEPARTURE:u32=1u;const NONFINITE:u32=2u;const PROCESSED:u32=4u;const FACE_UNAVAILABLE:u32=8u;const VELOCITY_UNAVAILABLE:u32=16u;const INVALID_STATUS:u32=32u;const NONPOSITIVE:u32=64u;
struct FineParams{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct DirectP{count:u32,capacity:u32,directoryCount:u32,rowCapacity:u32,dimensions:vec3u,maximumLeafSize:u32,reserved:u32,generation:u32,cellSize:f32,pad:u32}
struct P{dims:vec3u,maximumLeaf:u32,rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,reservedDirectory:u32,powerDirectoryCapacity:u32,powerRowCapacity:u32,generation:u32,maximumRounds:u32,ownersPerBrick:u32,powerGeneration:u32,axisStride:u32,ownedFacesPerRow:u32,closedBoundaryMask:u32,pad0:u32,pad1:u32,pad2:u32}
struct SampleP{dims:vec3u,maximumLeaf:u32,powerDirectoryCapacity:u32,count:u32,rowDirectoryCapacity:u32,rowCapacity:u32,cellSize:f32,fineGeneration:u32,p1:u32,p2:u32}
struct Pack{directRowOffset:u32,directRowCount:u32,directDirectoryOffset:u32,directDirectoryCount:u32,directVelocityOffset:u32,directVelocityCount:u32,tetraHeaderOffset:u32,tetraHeaderCount:u32,tetraVertexOffset:u32,tetraVertexCount:u32,tetrahedronOffset:u32,tetrahedronCount:u32,bandControlOffset:u32,bandRowOffset:u32,bandRowCount:u32,bandDirectoryOffset:u32,bandDirectoryWordCount:u32,bandVelocityOffset:u32,bandVelocityCount:u32,bandMetricOffset:u32,bandMetricCount:u32,transitionOffset:u32,pointOffset:u32,chunkBase:u32,closedDomainBoundary:u32,openTopBoundary:u32,transportBandDistance:f32,pad0:u32,pad1:u32,pad2:u32,pad3:u32,pad4:u32}
struct Chunk{base:u32,transportBandDistance:f32,closedDomainBoundary:u32,openTopBoundary:u32}
struct DirectRow{cell:u32,size:u32,topology:u32,flags:u32}struct DirectDirectoryEntry{cellPlusOne:u32,size:u32,row:u32,morton:u32}
struct Row{cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32}
struct Metric{topology:u32,transformFlags:u32,volume:f32,reserved:u32}struct TetraHeader{first:u32,count:u32,flags:u32}struct TetraVertex{v:vec4f}
struct Owner{origin:vec3u,size:u32,valid:u32}struct ResolvedPointVectorMeasurement{value:vec4f,directSuccess:u32,boundedCatalogSearch:u32,candidateRowsTested:u32}
struct VelocitySample{value:vec4f,status:u32}
@group(0)@binding(0)var<uniform>params:FineParams;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;@group(0)@binding(4)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(5)var<storage,read>phi:array<f32>;@group(0)@binding(6)var<storage,read_write>workA:array<f32>;
@group(0)@binding(10)var<storage,read_write>positions:array<vec4f>;@group(0)@binding(12)var<storage,read_write>outcomes:array<vec2u>;@group(0)@binding(13)var<storage,read>directAuthority:array<u32>;@group(0)@binding(14)var<storage,read>airAuthority:array<u32>;
@group(0)@binding(15)var<uniform>pack:Pack;@group(0)@binding(16)var<uniform>dp:DirectP;@group(0)@binding(17)var<uniform>p:P;@group(0)@binding(18)var<uniform>sp:SampleP;@group(0)@binding(19)var<storage,read>owners:array<u32>;
fn af(i:u32)->f32{return bitcast<f32>(airAuthority[i]);}fn df(i:u32)->f32{return bitcast<f32>(directAuthority[i]);}
fn loadDirectRow(i:u32)->DirectRow{let b=pack.directRowOffset+4u*i;return DirectRow(directAuthority[b],directAuthority[b+1u],directAuthority[b+2u],directAuthority[b+3u]);}
fn loadDirectDirectory(i:u32)->DirectDirectoryEntry{let b=pack.directDirectoryOffset+4u*i;return DirectDirectoryEntry(directAuthority[b],directAuthority[b+1u],directAuthority[b+2u],directAuthority[b+3u]);}
fn loadDirectVelocity(i:u32)->vec4f{let b=pack.directVelocityOffset+4u*i;return vec4f(df(b),df(b+1u),df(b+2u),df(b+3u));}
fn loadTetraHeader(i:u32)->TetraHeader{let b=pack.tetraHeaderOffset+3u*i;return TetraHeader(directAuthority[b],directAuthority[b+1u],directAuthority[b+2u]);}
fn loadTetraVertex(i:u32)->TetraVertex{let b=pack.tetraVertexOffset+4u*i;return TetraVertex(vec4f(df(b),df(b+1u),df(b+2u),df(b+3u)));}
fn loadTetrahedron(i:u32)->u32{return directAuthority[pack.tetrahedronOffset+i];}
fn loadBandRow(i:u32)->Row{let b=pack.bandRowOffset+8u*i;return Row(airAuthority[b],airAuthority[b+1u],airAuthority[b+2u],airAuthority[b+3u],af(b+4u),af(b+5u),af(b+6u),af(b+7u));}
fn loadBandVelocity(i:u32)->vec4f{let b=pack.bandVelocityOffset+4u*i;return vec4f(af(b),af(b+1u),af(b+2u),af(b+3u));}
fn loadBandMetric(i:u32)->Metric{let b=pack.bandMetricOffset+4u*i;return Metric(airAuthority[b],airAuthority[b+1u],af(b+2u),airAuthority[b+3u]);}
fn loadOwnerWord(i:u32)->u32{return owners[i];}
fn bandControl(i:u32)->u32{return airAuthority[pack.bandControlOffset+i];}fn transitionWord(i:u32)->u32{return airAuthority[pack.transitionOffset+i];}fn pointWord(i:u32)->u32{return airAuthority[pack.pointOffset+i];}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(LARGE));}fn directVelocityValid(v:vec4f)->bool{return v.w>0.&&finite3(v.xyz);}
fn directMortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn directMorton(cell:u32)->u32{let q=vec3u(cell%dp.dimensions.x,(cell/dp.dimensions.x)%dp.dimensions.y,cell/(dp.dimensions.x*dp.dimensions.y));return directMortonPart10(q.x)|(directMortonPart10(q.y)<<1u)|(directMortonPart10(q.z)<<2u);}
fn directLevel(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn directKeyLess(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn directFind(c:u32,s:u32)->u32{let count=min(dp.directoryCount,pack.directDirectoryCount);let wantedLevel=directLevel(s);let wantedMorton=directMorton(c);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=loadDirectDirectory(middle);if(directKeyLess(directLevel(candidate.size),candidate.morton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let candidate=loadDirectDirectory(low);if(candidate.cellPlusOne==c+1u&&candidate.size==s){return candidate.row;}}return INVALID;}
fn directOwner(x:vec3f)->u32{let g=x/dp.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(dp.dimensions))){return INVALID;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let r=directFind(o.x+dp.dimensions.x*(o.y+dp.dimensions.y*o.z),s);if(r!=INVALID){return r;}if(s>=dp.maximumLeafSize){break;}s*=2u;}return INVALID;}
fn directInverse(x:vec3f,c:u32)->vec3f{let bits=c&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(c/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn directTransform(value:vec3f,code:u32)->vec3f{let k=(code/8u)%6u;var r=value;if(k==1u){r=value.xzy;}else if(k==2u){r=value.yxz;}else if(k==3u){r=value.yzx;}else if(k==4u){r=value.zxy;}else if(k==5u){r=value.zyx;}let bits=code&7u;return r*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));}
fn directWeights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let d=dot(a,cross(b,c));if(!finite3(vec3f(d))||abs(d)<=1e-10){return vec4f(-2.);}let wa=dot(point,cross(b,c))/d;let wb=dot(a,cross(point,c))/d;let wc=dot(a,cross(b,point))/d;return vec4f(1.-wa-wb-wc,wa,wb,wc);}fn directContained(w:vec4f)->bool{return all(w>=vec4f(-2e-6))&&all(w<=vec4f(1.000002));}
fn directNeighbor(center:vec3f,size:u32,transform:u32,q:vec4f)->vec4f{let point=center+f32(size)*dp.cellSize*directInverse(q.xyz,transform);let ns=u32(round(f32(size)*q.w));let no=round(point/dp.cellSize-.5*f32(ns));if(any(no<vec3f(0))){return vec4f(0.);}let n=vec3u(no);let row=directFind(n.x+dp.dimensions.x*(n.y+dp.dimensions.y*n.z),ns);if(row==INVALID||row>=pack.directVelocityCount){return vec4f(0.);}return loadDirectVelocity(row);}
fn sampleDirect(x:vec3f)->VelocitySample{let row=directOwner(x);if(row==INVALID){return VelocitySample(vec4f(0),8u);}if(row>=pack.directRowCount||row>=pack.directVelocityCount){return VelocitySample(vec4f(0),4u);}let r=loadDirectRow(row);if(r.size==0u||r.topology>=pack.tetraHeaderCount){return VelocitySample(vec4f(0),4u);}let th=loadTetraHeader(r.topology);let o=vec3u(r.cell%dp.dimensions.x,(r.cell/dp.dimensions.x)%dp.dimensions.y,r.cell/(dp.dimensions.x*dp.dimensions.y));let center=(vec3f(o)+.5*f32(r.size))*dp.cellSize;let marker=th.flags&0x10000000u;
 if((th.flags&1u)!=0u){let g=x/dp.cellSize;let lowi=select(vec3i(o)-vec3i(i32(r.size)),vec3i(o),g>=vec3f(o)+.5*f32(r.size));let low=max(lowi,vec3i(0));let t=clamp((g-(vec3f(low)+.5*f32(r.size)))/f32(r.size),vec3f(0),vec3f(1));let anchor=loadDirectVelocity(row);var value=vec3f(0.);for(var corner=0u;corner<8u;corner+=1u){let co=vec3u(low)+vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u)*r.size;var v=anchor;if(all(co<dp.dimensions)){let nr=directFind(co.x+dp.dimensions.x*(co.y+dp.dimensions.y*co.z),r.size);if(nr!=INVALID&&nr<pack.directVelocityCount){v=loadDirectVelocity(nr);}}if(!directVelocityValid(v)){return VelocitySample(vec4f(0),4u);}let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);value+=weight*v.xyz;}return VelocitySample(vec4f(value,1.),VELOCITY_VALID|0x40000000u|marker);}
 let local=directTransform((x-center)/(f32(r.size)*dp.cellSize),r.flags&63u);let anchor=loadDirectVelocity(row);if(!directVelocityValid(anchor)){return VelocitySample(vec4f(0),4u);}if(th.first>pack.tetrahedronCount||th.count>pack.tetrahedronCount-th.first){return VelocitySample(vec4f(0),2u);}for(var ti=0u;ti<th.count;ti+=1u){let packed=loadTetrahedron(th.first+ti);let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(min(75u,pack.tetraVertexCount)))){return VelocitySample(vec4f(0),2u);}let a=loadTetraVertex(selectors.x).v;let b=loadTetraVertex(selectors.y).v;let c=loadTetraVertex(selectors.z).v;let weights=directWeights(local,a.xyz,b.xyz,c.xyz);if(!directContained(weights)){continue;}let va=directNeighbor(center,r.size,r.flags&63u,a);let vb=directNeighbor(center,r.size,r.flags&63u,b);let vc=directNeighbor(center,r.size,r.flags&63u,c);if(!directVelocityValid(va)||!directVelocityValid(vb)||!directVelocityValid(vc)){return VelocitySample(vec4f(0),4u);}return VelocitySample(vec4f(weights.x*anchor.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz,1.),VELOCITY_VALID|ti|marker);}return VelocitySample(vec4f(0),8u|((r.topology&0xffffu)<<8u));}
fn bandIdentitySlot(cellKey:u32,size:u32)->u32{let domain=p.dims.x*p.dims.y*p.dims.z;let supported=size==1u||size==2u||size==4u||size==8u||size==16u||size==32u;if(!supported||cellKey>=domain){return INVALID;}let origin=vec3u(cellKey%p.dims.x,(cellKey/p.dims.x)%p.dims.y,cellKey/(p.dims.x*p.dims.y));if(any(origin%vec3u(size)!=vec3u(0u))||any(origin+vec3u(size)>p.dims)){return INVALID;}let level=31u-countLeadingZeros(size);if(p.reservedDirectory>pack.bandDirectoryWordCount||level>5u||level*domain>pack.bandDirectoryWordCount-p.reservedDirectory){return INVALID;}let relative=p.reservedDirectory+level*domain;if(relative>=pack.bandDirectoryWordCount||cellKey>=pack.bandDirectoryWordCount-relative){return INVALID;}return relative+cellKey;}
fn rowOfIdentity(cellKey:u32,size:u32)->u32{let relative=bandIdentitySlot(cellKey,size);if(relative==INVALID||pack.bandDirectoryOffset>arrayLength(&airAuthority)||relative>=arrayLength(&airAuthority)-pack.bandDirectoryOffset){return INVALID;}let encoded=airAuthority[pack.bandDirectoryOffset+relative];if(encoded==0u){return INVALID;}let row=encoded-1u;if(row>=bandControl(2u)||row>=p.rowDirectoryCapacity||row>=pack.bandRowCount){return INVALID;}let candidate=loadBandRow(row);return select(INVALID,row,candidate.cell==cellKey&&candidate.size==size);}
${airFunctions}
fn retainedBandAnchor(pointGrid:vec3f)->u32{if(any(pointGrid<vec3f(0))||any(pointGrid>=vec3f(sp.dims))){return INVALID;}let q=vec3u(floor(pointGrid));var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let band=rowOfIdentity(cell(origin),size);if(band!=INVALID&&band<sp.rowCapacity&&band<transitionWord(34u)&&band<pack.bandRowCount){let row=loadBandRow(band);if(row.cell==cell(origin)&&row.size==size&&(row.flags&64u)==0u){return band;}}if(size>=sp.maximumLeaf){break;}size<<=1u;}return INVALID;}
fn fusedBandGenerationValid()->bool{let mask=0x3fffffffu;let fine=sp.fineGeneration&mask;let paramsFine=p.generation&mask;let band=bandControl(5u)&mask;let power=p.powerGeneration&mask;let predecessor=(fine+mask)&mask;return paramsFine==fine&&(band==fine||(power==fine&&band==predecessor));}
fn sampleCompleteVelocity(world:vec3f)->VelocitySample{let direct=sampleDirect(world);if((direct.status&VELOCITY_VALID)!=0u){return direct;}let sampled=airSampleGrid(world);if(sampled.w==0.){return VelocitySample(vec4f(0),0x01000001u);}let grid=sampled.xyz;let owner=ownerAt(vec3u(floor(grid)));if(owner.valid==0u){return VelocitySample(vec4f(0),0x01000002u);}let band=retainedBandAnchor(grid);if(band==INVALID||band>=sp.rowCapacity||band>=transitionWord(34u)||band>=pack.bandRowCount){return VelocitySample(vec4f(0),0x01000003u);}if(!fusedBandGenerationValid()||pointWord(3u)!=sp.fineGeneration){return VelocitySample(vec4f(0),0x01000004u);}let row=loadBandRow(band);if((row.flags&1u)==0u){return VelocitySample(vec4f(0),0x01000003u);}let velocity=resolveFinalPointVectorMeasured(band,grid).value;if(!velocityValid(velocity)){let r=u32(round(max(1.,-velocity.w)));return VelocitySample(velocity,STATUS_FACE_UNAVAILABLE|0x01000000u|((band&0xffffu)<<8u)|((16u+min(r,11u))&255u));}if(bandControl(6u)==VELOCITY_VALID&&fusedBandGenerationValid()&&pointWord(5u)==VELOCITY_VALID&&pointWord(0u)==0u&&pointWord(3u)==sp.fineGeneration){return VelocitySample(velocity,VELOCITY_VALID|STATUS_EXTRAPOLATED);}return VelocitySample(velocity,STATUS_FACE_UNAVAILABLE|0x02000000u);}
fn outcomeFlags(v:u32)->u32{return v&255u;}fn outcomeExtrapolated(v:u32)->u32{return(v>>8u)&15u;}fn packOutcome(flags:u32,extrapolated:u32,displacement:u32)->u32{return(flags&255u)|((extrapolated&15u)<<8u)|(min(displacement,0xfffffu)<<12u);}
fn activeSample(flat:u32)->vec2u{if(arrayLength(&worklist)<5u||worklist[1]!=params.generation||worklist[3]!=1u||worklist[4]!=1u){return vec2u(INVALID);}let count=min(worklist[0],params.pageCapacity);if(5u+count>arrayLength(&worklist)||flat>=count*params.samplesPerBrick){return vec2u(INVALID);}let w=flat/params.samplesPerBrick;let local=flat-w*params.samplesPerBrick;let id=worklist[5u+w];if(id>=params.pageCapacity||metadata[id*10u]!=id||metadata[id*10u+2u]!=params.generation){return vec2u(INVALID);}return vec2u(id,local);}
fn unpackBrick(key:u32)->vec3u{let xy=params.brickDimensions.x*params.brickDimensions.y;let z=key/xy;let r=key-z*xy;let y=r/params.brickDimensions.x;return vec3u(r-y*params.brickDimensions.x,y,z);}fn localCoord(local:u32)->vec3u{let r=params.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}fn packBrick(q:vec3u)->u32{return q.x+params.brickDimensions.x*(q.y+params.brickDimensions.y*q.z);}
${makeFineLevelSetSortedWorklistLookupWGSL("params", "metadata", "worklist", "lookupFine")}
fn loadFine(q:vec3i)->vec3f{if(any(q<vec3i(0))||any(q>=vec3i(params.sampleDimensions))){return vec3f(0.,0.,1.);}let uq=vec3u(q);let brick=uq/params.brickResolution;let local=uq-brick*params.brickResolution;let id=lookupFine(packBrick(brick));if(id==INVALID){return vec3f(0.,0.,2.);}let i=id*params.samplesPerBrick+local.x+params.brickResolution*(local.y+params.brickResolution*local.z);if((sampleFlags[i]&VALID_FINE)==0u){return vec3f(0.,0.,3.);}let v=phi[i];if(v!=v||abs(v)>=LARGE){return vec3f(0.,0.,4.);}return vec3f(v,1.,0.);}
fn trilinear(x:vec3f)->vec3f{let raw=(x-params.domainOrigin)/params.fineCellWidth-vec3f(.5);let wall=vec3f(params.sampleDimensions)-vec3f(.5);let below=raw<vec3f(-.501);let above=raw>wall+vec3f(.001);if(pack.closedDomainBoundary==0u&&(any(below)||above.x||above.z||(above.y&&pack.openTopBoundary==0u))){return vec3f(0.,0.,1.);}let sampleMax=vec3f(params.sampleDimensions)-vec3f(1.);if(pack.closedDomainBoundary==0u&&(any(raw<vec3f(0.))||raw.x>sampleMax.x||raw.z>sampleMax.z||(raw.y>sampleMax.y&&pack.openTopBoundary==0u))){return vec3f(0.,0.,1.);}let extend=pack.closedDomainBoundary!=0u||pack.openTopBoundary!=0u;let lattice=select(raw,clamp(raw,vec3f(0),sampleMax),extend);let base=vec3i(floor(lattice));let f=fract(lattice);var v=0.;for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x0=0;x0<2;x0+=1){let w=select(1.-f.x,f.x,x0==1)*select(1.-f.y,f.y,y==1)*select(1.-f.z,f.z,z==1);if(w==0.){continue;}let q=loadFine(base+vec3i(x0,y,z));if(q.y==0.){return q;}v+=w*q.x;}}}return vec3f(v,1.,0.);}
@compute @workgroup_size(64)fn transportFineCharacteristic(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let local=fineLinearWorkgroup(wg,n)*64u+lid;if(local>=arrayLength(&positions)||local>=arrayLength(&outcomes)){return;}positions[local]=vec4f(0);outcomes[local]=vec2u(0u,INVALID);let flat=pack.chunkBase+local;let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*params.samplesPerBrick+a.y;if((sampleFlags[index]&VALID_FINE)==0u){return;}workA[index]=phi[index];if(abs(phi[index])>=pack.transportBandDistance){return;}let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*params.brickResolution+localCoord(a.y);let origin=params.domainOrigin+(vec3f(q)+.5)*params.fineCellWidth;var position=origin;var flags=0u;var extrapolated=0u;let segmentDt=params.timestep/f32(params.fineFactor);var segment=0u;loop{if(segment>=params.fineFactor){break;}let sampled=sampleCompleteVelocity(position);let status=sampled.status;if((status&STATUS_FACE_UNAVAILABLE)!=0u){flags|=FACE_UNAVAILABLE;}if((status&VELOCITY_VALID)==0u){if(status==0x01000003u){flags|=PROCESSED;outcomes[local]=vec2u(packOutcome(flags,extrapolated,0u),status);positions[local]=vec4f(position,0.);return;}flags|=INVALID_STATUS|VELOCITY_UNAVAILABLE;outcomes[local]=vec2u(packOutcome(flags,extrapolated,0u),status);positions[local]=vec4f(position,-1.);return;}if(sampled.value.w<=0.){flags|=NONPOSITIVE|VELOCITY_UNAVAILABLE;outcomes[local]=vec2u(packOutcome(flags,extrapolated,0u),INVALID);positions[local]=vec4f(position,-1.);return;}if(!finite3(sampled.value.xyz)){flags|=NONFINITE;outcomes[local]=vec2u(packOutcome(flags,extrapolated,0u),INVALID);positions[local]=vec4f(position,-1.);return;}if((status&STATUS_EXTRAPOLATED)!=0u){extrapolated+=1u;}position-=segmentDt*sampled.value.xyz;if(!finite3(position)){flags|=NONFINITE;outcomes[local]=vec2u(packOutcome(flags,extrapolated,0u),INVALID);positions[local]=vec4f(position,-1.);return;}segment+=1u;}
 let displacement=u32(ceil(length(position-origin)/params.fineCellWidth));let value=trilinear(position);positions[local]=vec4f(position,1.);if(value.y==0.){flags|=DEPARTURE;outcomes[local]=vec2u(packOutcome(flags,extrapolated,displacement),0x04000000u|u32(value.z));positions[local].w=-1.;return;}workA[index]=value.x;flags|=PROCESSED;outcomes[local]=vec2u(packOutcome(flags,extrapolated,displacement),INVALID);}
`;
}
