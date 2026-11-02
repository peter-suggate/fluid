/** Executable hypotheses for all ten NB-FLIP optimization questions.
 * Production surface arithmetic is retained; only data/scheduling changes. */
import {fnSpan,replaceFn,replaceOnce,probeVariants} from './probe-nbflip-variants';
export interface StudyStep {entry:string;groups?:number;indirect?:number;args?:number}
export interface StudyVariant {name:string;section:number;source:(code:string)=>string;steps:StudyStep[];claim:string}
const reset:StudyStep={entry:'studyReset',groups:1024},plan:StudyStep={entry:'surfaceSplatPlan',groups:1024},scatter:StudyStep={entry:'surfaceSplatOccupied',groups:1024};
const ordinary=[reset,plan,scatter];
const args=(mode=0):StudyStep=>({entry:'studyArgs',groups:1,args:mode});
const indirect:StudyStep={entry:'surfaceSplatOccupied',indirect:0};
const identity=(code:string)=>code;
const rewrite=(code:string,edits:readonly (readonly [string,string])[])=>edits.reduce((text,[before,after])=>replaceOnce(text,before,after),code);
function changeScatter(code:string,edit:(body:string)=>string,name='surfaceSplatOccupied'){
 const [a,b]=fnSpan(code,name);return replaceFn(code,name,edit(code.slice(a,b)));
}
const loadJob='if(lane==0u){let at=nbSurfaceCells()+2u*job;nbSurfaceJob=vec2u(links[at],links[at+1u]);}';
function stream(code:string,kind:'soa'|'tiled'|'packed'){
 const pos=kind==='tiled'?'256u*(i/64u)+(i%64u)':'i';
 const stride=kind==='tiled'?'64u':'arrayLength(&particles)';
 code=replaceFn(code,'nbPosition',`fn nbPosition(i:u32)->vec3f{let a=${pos};let n=${stride};return bitcast<vec3f>(vec3u(study.words[a],study.words[a+n],study.words[a+2u*n]));}`);
 code=replaceFn(code,'nbSurfaceParticleHot',kind==='packed'?`fn nbSurfaceParticleHot(i:u32)->bool{return (study.words[3u*arrayLength(&particles)+(i>>5u)]&(1u<<(i&31u)))!=0u;}`:
 `fn nbSurfaceParticleHot(i:u32)->bool{let a=${pos};let n=${stride};return bitcast<f32>(study.words[a+3u*n])>1.0;}`);
 return code+`\n@compute @workgroup_size(64) fn studyPack(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let n=studyMeta.x;
 for(var i=gid.x;i<n;i+=64u*groups.x){let a=${pos};let stride=${stride};let src=NB_LINKS+4u*i;
  study.words[a]=links[src];study.words[a+stride]=links[src+1u];study.words[a+2u*stride]=links[src+2u];
  ${kind==='packed'?'': 'study.words[a+3u*stride]=links[src+3u];'}
 }
 ${kind==='packed'?`for(var word=gid.x;word<(n+31u)/32u;word+=64u*groups.x){var bits=0u;for(var k=0u;k<32u&&32u*word+k<n;k++){if(bitcast<f32>(links[NB_LINKS+4u*(32u*word+k)+3u])>1.0){bits|=1u<<k;}}study.words[3u*arrayLength(&particles)+word]=bits;}`:''}
}`;
}
function fused(code:string){
 return changeScatter(code,body=>rewrite(body,[
  ['nbSurfaceJobs=atomicLoad(&bins[NB_BAND+3u]);','nbSurfaceJobs=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);'],
  [loadJob,'if(lane==0u){let tile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);nbSurfaceJob=vec2u(tile,links[nbBinTile(tile)]);}'],
  ['min(start+NB_SURFACE_CHUNK,links[bin+63u]+atomicLoad(&bins[bin+63u]))','links[bin+63u]+atomicLoad(&bins[bin+63u])'],
 ]));
}
function buckets(code:string){
 code=replaceFn(code,'surfaceSplatPlan',`fn surfaceSplatPlan(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let count=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);let cap=arrayLength(&particles);
 for(var j=gid.x;j<count;j+=64u*groups.x){let tile=atomicLoad(&bins[NB_OCCUPIED+4u+j]);if(!umTileLive(tile)){continue;}
  let bin=nbBinTile(tile);let start=links[bin];let end=links[bin+63u]+atomicLoad(&bins[bin+63u]);
  for(var first=start;first<end;first+=NB_SURFACE_CHUNK){let n=min(end-first,NB_SURFACE_CHUNK);let bucket=select(select(2u,1u,n>=128u),0u,n>=384u);
   let rank=atomicAdd(&study.counts[bucket],1u);let at=2u*(bucket*cap+rank);study.words[at]=tile;study.words[at+1u]=first;
  }
 }
}`);
 return changeScatter(code,body=>rewrite(body,[['nbSurfaceJobs=atomicLoad(&bins[NB_BAND+3u]);','nbSurfaceJobs=atomicLoad(&study.counts[0])+atomicLoad(&study.counts[1])+atomicLoad(&study.counts[2]);'],
 [loadJob,`if(lane==0u){let a=atomicLoad(&study.counts[0]);let b=atomicLoad(&study.counts[1]);var bucket=0u;var rank=job;
 if(job>=a){bucket=1u;rank=job-a;}if(job>=a+b){bucket=2u;rank=job-a-b;}
 let at=2u*(bucket*arrayLength(&particles)+rank);nbSurfaceJob=vec2u(study.words[at],study.words[at+1u]);}`]]));
}
function split(code:string){
 // Classify the original bounded jobs. The common kernel omits all hot state;
 // exceptional jobs use the unmodified two-window computation.
 const [a,b]=fnSpan(code,'surfaceSplatOccupied');let body=code.slice(a,b);
 const queued=(text:string,bucket:number)=>rewrite(text,[['nbSurfaceJobs=atomicLoad(&bins[NB_BAND+3u]);',`nbSurfaceJobs=atomicLoad(&study.counts[${bucket}]);`],
 [loadJob,`if(lane==0u){let at=2u*(${bucket}u*arrayLength(&particles)+job);nbSurfaceJob=vec2u(study.words[at],study.words[at+1u]);}`]]);
 const exceptional=replaceOnce(queued(body,1),'fn surfaceSplatOccupied','fn studyExceptional');
 body=rewrite(queued(body,0),[
  ['atomicStore(&nbSurfaceWindowHot[k],NB_SURFACE_FAR);',''],['let hot=nbSurfaceParticleHot(i);',''],
  ['if(hot){atomicMin(&nbSurfaceWindowHot[w],bitcast<u32>(d2));}',''],
  ['let hot=atomicLoad(&nbSurfaceWindowHot[k]);if(hot!=NB_SURFACE_FAR){atomicMax(&bins[NB_SURFACE_HOT_NEAREST+umVertexSlotAt(slot)],~hot);}',''],
 ]);
 code=replaceFn(code,'surfaceSplatOccupied',body);
 return code+'\n@compute @workgroup_size(128) '+exceptional+`
var<workgroup> studyHasHot:atomic<u32>;
var<workgroup> studyClassCount:u32;var<workgroup> studyClassJob:vec2u;
@compute @workgroup_size(64) fn studyClassify(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){studyClassCount=atomicLoad(&bins[NB_BAND+3u]);}let count=workgroupUniformLoad(&studyClassCount);
 for(var j=group.x;j<count;j+=groups.x){
  if(lane==0u){let at=nbSurfaceCells()+2u*j;studyClassJob=vec2u(links[at],links[at+1u]);atomicStore(&studyHasHot,0u);}
  let job=workgroupUniformLoad(&studyClassJob);let bin=nbBinTile(job.x);let end=min(job.y+NB_SURFACE_CHUNK,links[bin+63u]+atomicLoad(&bins[bin+63u]));
  for(var i=job.y+lane;i<end;i+=64u){if(nbSurfaceParticleHot(i)){atomicStore(&studyHasHot,1u);}}
  workgroupBarrier();
  if(lane==0u){let bucket=atomicLoad(&studyHasHot);let rank=atomicAdd(&study.counts[bucket],1u);let at=2u*(bucket*arrayLength(&particles)+rank);study.words[at]=job.x;study.words[at+1u]=job.y;}
  workgroupBarrier();
 }
}`;
}
export const sectionVariants:StudyVariant[]=[
 {name:'baseline',section:0,source:identity,steps:ordinary,claim:'128 lanes, 512-particle jobs, compact AoS'},
 {name:'layout-soa',section:1,source:c=>stream(c,'soa'),steps:[reset,{entry:'studyPack',groups:1024},plan,scatter],claim:'SoA, including conversion'},
 {name:'layout-tiled',section:1,source:c=>stream(c,'tiled'),steps:[reset,{entry:'studyPack',groups:1024},plan,scatter],claim:'64-particle AoSoA tiles, including conversion'},
 {name:'bandwidth-packed-hot',section:2,source:c=>stream(c,'packed'),steps:[reset,{entry:'studyPack',groups:1024},plan,scatter],claim:'Exact 1-bit hot flags: 12.125 instead of 16 storage bytes/sample; includes packing'},
 {name:'active-indirect',section:3,source:identity,steps:[reset,plan,args(),indirect],claim:'GPU count, capped indirect dispatch, includes arguments and validation'},
 {name:'cost-buckets',section:4,source:buckets,steps:ordinary,claim:'Order jobs by expected particle count; includes bucket construction'},
 {name:'fused-tile',section:5,source:fused,steps:[reset,scatter],claim:'Consume occupied tiles directly; remove planner and temporary jobs'},
 {name:'split-hot',section:6,source:split,steps:[reset,plan,{entry:'studyClassify',groups:1024},args(2),indirect,{entry:'studyExceptional',indirect:12}],claim:'No-hot common kernel plus compact exceptional queue; includes classification'},
 {name:'aggregated-atomics',section:7,source:c=>probeVariants.splatplan!(c,''),steps:ordinary,claim:'Reserve jobs once per workgroup instead of per occupied tile'},
 {name:'reuse-scalar',section:8,source:identity,steps:[reset,{entry:'surfaceCells',groups:1024},args(1),{entry:'surfaceSplatReference',indirect:0}],claim:'Per-cell scalar reference, no cooperative position staging'},
 {name:'reuse-shared',section:8,source:identity,steps:[reset,{entry:'surfaceCells',groups:1024},args(1),{entry:'surfaceSplat',indirect:0}],claim:'Same cells and output arithmetic; share positions across 64 vertex lanes'},
 {name:'lanes64',section:9,source:c=>probeVariants.splat64!(c,''),steps:ordinary,claim:'64 lanes with matching strides'},
 {name:'lanes256',section:9,source:c=>probeVariants.splat256!(c,''),steps:ordinary,claim:'256 lanes with matching strides'},
 {name:'baseline-repeat',section:0,source:identity,steps:ordinary,claim:'Unchanged A/A control'},
];
export const studyExtraWGSL=`
struct Study {counts:array<atomic<u32>,4>,words:array<u32>}
@group(0) @binding(3) var<storage,read_write> study:Study;
@group(0) @binding(4) var<uniform> studyMeta:vec4u;
@compute @workgroup_size(64) fn studyReset(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(gid.x==0u){atomicStore(&bins[NB_BAND+3u],0u);for(var k=0u;k<4u;k++){atomicStore(&study.counts[k],0u);}}
 // The fixture stores the two result planes consecutively. Clear both for
 // every timed iteration, so global minima must do real updates again.
 for(var i=gid.x;i<2u*(NB_SURFACE_HOT_NEAREST-NB_DEPTH_A);i+=64u*groups.x){atomicStore(&bins[NB_DEPTH_A+i],0u);}
}
`;
export const studyArgsWGSL=(band:number)=>`
@group(0) @binding(0) var<storage,read> bins:array<u32>;
@group(0) @binding(1) var<storage,read> study:array<u32>;
@group(0) @binding(2) var<storage,read_write> args:array<u32>;
override mode:u32=0u;
@compute @workgroup_size(1) fn studyArgs(){
 args[0]=min(bins[${band+3}u],1024u);args[1]=1u;args[2]=1u;
 if(mode==1u){args[0]=bins[${band+3}u];}
 if(mode==2u){args[0]=min(study[0],1024u);args[3]=min(study[1],1024u);args[4]=1u;args[5]=1u;}
}`;
