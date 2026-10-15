import assert from "node:assert/strict";
import test from "node:test";
import { RESIDENCY_STAGES, residencyDirectionRadii, residencyStageMeets, residencyFootprintContains } from "../lib/methods/uniform/uniform-local-residency";
import { withUniformDevice, createUniformSolver, advanceUniform } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { UNIFORM_PAGE_POOL_RECORD as R, UNIFORM_PAGE_POOL_BLOCK as B, UNIFORM_PAGE_POOL_STATUS as S, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";

test("transport air does not inherit a distant surface stencil or ballistic travel",()=>{
 const transport=residencyDirectionRadii([0,0,0],[0,0,0],[1,1,1],undefined,RESIDENCY_STAGES[1].margin);
 const particles=residencyDirectionRadii([0,0,0],[0,0,0],[1,1,1],[0,0,0,24,0,0],RESIDENCY_STAGES[4].margin);
 // Surface/particle seed at x=0, transport-only seed at x=3. The old union
 // applies the particle's +8 radius to BOTH, keeping a page starting at 9.
 assert.equal(residencyStageMeets(1,[1<<3,0],transport,[9,0,0],[16,7,7]),false);
 assert.equal(residencyStageMeets(4,[1,0],particles,[9,0,0],[16,7,7]),false);
 assert.equal(residencyStageMeets(4,[1|1<<3,0],particles,[9,0,0],[16,7,7]),true);
 // The particle's actual arrival and stencil still have backing.
 assert.equal(residencyStageMeets(4,[1,0],particles,[8,0,0],[11,3,3]),true);
});

test("stage footprints retain axial reads without filling diagonal corners",()=>{
 assert.equal(residencyFootprintContains(1,[2,2,2]),true,"eight-cell mixed sampler");
 assert.equal(residencyFootprintContains(1,[3,0,0]),false,"optional extension work is not stored input");
 assert.equal(residencyFootprintContains(3,[3,1,1]),true,"two axial reads from an incident owner");
 assert.equal(residencyFootprintContains(3,[2,2,1]),true,"axial reads can use different axes");
 assert.equal(residencyFootprintContains(3,[2,2,2]),false,"three axial steps are not a read");
 assert.equal(residencyFootprintContains(2,[1,0,0]),true);
 assert.equal(residencyFootprintContains(2,[1,1,0]),false,"projection reads adjacent faces");
});

test("swept stage queries match enumeration of individual dependency offsets",()=>{
 let random=81231;const next=()=>random=(Math.imul(random,1664525)+1013904223)>>>0;
 const offsets=RESIDENCY_STAGES.map(({core,faces})=>{
  const values:number[][]=[];
  // Independent construction: a tensor-product sampling set followed by
  // the six incident-face steps. Union deduplicates overlaps.
  const unique=new Set<string>();
  for(let z=-core;z<=core;z++)for(let y=-core;y<=core;y++)for(let x=-core;x<=core;x++){
   unique.add([x,y,z].join(","));
   let frontier=[[x,y,z]];
   for(let step=0;step<faces;step++){
    const next:number[][]=[];
    for(const q of frontier)for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){const p=[...q];p[axis]!+=sign;unique.add(p.join(","));next.push(p);}
    frontier=next;
   }
  }
  for(const p of unique)values.push(p.split(",").map(Number));return values;
 });
 for(let trial=0;trial<1200;trial++){
  const stage=next()%RESIDENCY_STAGES.length,m=RESIDENCY_STAGES[stage]!.margin;
  const mask=[next()&next(),next()&next()],travel=Array.from({length:6},()=>next()%5),radii=travel.map(v=>v+m);
  const low=Array.from({length:3},()=>4*(next()%9-4)),side=next()%2?4:8,high=low.map(v=>v+side-1);
  let expected=false;
  for(let bit=0;bit<64&&!expected;bit++)if((mask[bit>>>5]!>>>(bit&31))&1){
   const p=[bit&3,(bit>>>2)&3,bit>>>4];
   expected=offsets[stage]!.some(o=>o.every((v,k)=>p[k]!+v-travel[k]!<=high[k]!&&p[k]!+v+travel[k+3]!>=low[k]!));
  }
  assert.equal(residencyStageMeets(stage,mask,radii,low,high),expected,`trial ${trial}, stage ${stage}`);
 }
});

test("long slide stage requests and post-commit certificates agree and keep fluid support",{skip:!process.env.WEBGPU_NODE_MODULE},async t=>{
 await withUniformDevice("per-stage residency long slide",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("nbflip-water-slide-long")));
  const dt=1/60;scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  // Start with all pages resident: the test must observe actual retirement,
  // rather than pass because the empty corner was never allocated.
  scene.fluid.pagePool={...scene.fluid.pagePool,air:[]};
  const solver=await createUniformSolver(device,scene);
  const internal=solver as unknown as {mixedDynamic:{work:GPUBuffer;localOffset:number};mixedFrame:{requestNarrowBandVolumeProbe():void;pagePool:UniformPagePool;ownership:{support:GPUBuffer;capacity:{tiles:number}}}};
  const {pagePool:pool,ownership}=internal.mixedFrame,{mixedDynamic:dynamic}=internal;
  const l=pool.layout,n=8*l.blocks,certificate=9*ownership.capacity.tiles+28;
  let first=0,last=0;
  try{
   const initial=new Int32Array((await readMixedBuffer(device,pool.state)).buffer);
   assert.ok(Array.from({length:l.blocks-2},(_,i)=>l.state.records+R.words*(i+1)).some(r=>initial[r]===0&&initial[r+1]===0&&initial[r+2]===3&&initial[r+R.state]===B.resident),"corner starts allocated");
   for(let step=1;step<=169;step++){
    // Run the optional end-of-frame volume reader as well as the head gauge.
    internal.mixedFrame.requestNarrowBandVolumeProbe();
    await advanceUniform(solver,step*dt);
    assert.ok(Math.abs(solver.info.completedTime_s!-step*dt)<1e-8);
    if(step!==1&&step!==30&&step!==169)continue;
    const [rawState,rawCensus,rawSupport]=await Promise.all([readMixedBuffer(device,pool.state),readMixedBuffer(device,dynamic.work),readMixedBuffer(device,ownership.support)]);
    const state=new Uint32Array(rawState.buffer),signed=new Int32Array(rawState.buffer),census=new Uint32Array(rawCensus.buffer),support=new Uint32Array(rawSupport.buffer);
    assert.equal(state[S.fatal],0);assert.equal(support[certificate+1],0,"no missing particle or volume support");
    const resident=new Set<string>(),expected=new Map<string,number>(),legacy=new Set<string>();
    for(let b=1;b<l.blocks-1;b++){const r=l.state.records+R.words*b;if(state[r+R.state]===B.resident)resident.add(Array.from(signed.subarray(r,r+3)).join(","));}
    // Enumerate every occupied tile's swept box independently of lrMeets,
    // which inverts the query on the GPU. Compare every stage flag and every
    // requested physical page, including diagonal and page-edge support.
    for(let page=0;page<n;page++)for(let stage=0;stage<RESIDENCY_STAGES.length;stage++){
     const at=dynamic.localOffset+4*(stage*n+page),r=l.state.records+R.words*(page>>>3);
     const origin=[0,1,2].map(k=>8*signed[r+k]!+4*((page>>>k)&1));
     const neg=[0,1,2].map(k=>(census[at+2]!>>>(8*k))&255),pos=[0,1,2].map(k=>(census[at+3]!>>>(8*k))&255);
     for(let bit=0;bit<64;bit++){
      if(((census[at+(bit>>>5)]!>>>(bit&31))&1)===0)continue;
      const q=[bit&3,(bit>>>2)&3,bit>>>4].map((v,k)=>v+origin[k]!);
      const add=(margin:number,visit:(p:number[])=>void)=>{
       const lo=q.map((v,k)=>Math.floor((v-neg[k]!-margin)/4)),hi=q.map((v,k)=>Math.floor((v+pos[k]!+margin)/4));
       for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++){
        const p=[x,y,z],block=p.map(v=>Math.floor(v/2));
        if(block.some((v,k)=>v<signed[l.state.window+24+k]!||v>=signed[l.state.window+27+k]!))continue;
        visit(p);
       }
      };
      // Enumerate the dependency offsets, then test their swept tile boxes.
      // This is independent of the GPU's distance-to-box footprint query.
      add(0,p=>{
       const m=RESIDENCY_STAGES[stage]!.margin;
       let hit=false;
       for(let z=-m;z<=m&&!hit;z++)for(let y=-m;y<=m&&!hit;y++)for(let x=-m;x<=m&&!hit;x++){
        if(!residencyFootprintContains(stage,[x,y,z]))continue;
        hit=[x,y,z].every((v,k)=>q[k]!+v-(neg[k]!-m)<=4*p[k]!+3&&q[k]!+v+(pos[k]!-m)>=4*p[k]!);
       }
       if(hit){const key=p.join(",");expected.set(key,(expected.get(key)??0)|(1<<stage));}
      });
      add(5-RESIDENCY_STAGES[stage]!.margin,p=>legacy.add(p.map(v=>Math.floor(v/2)).join(",")));
     }
    }
    const requested=new Set<string>();
    for(const key of expected.keys()){const p=key.split(",").map(Number),block=p.map(v=>Math.floor(v/2)).join(",");requested.add(block);assert.ok(resident.has(block),`stage support missing physical page ${block}`);}
    for(let b=1;b<l.blocks-1;b++){
     const r=l.state.records+R.words*b;if(state[r+R.state]!==B.resident)continue;
     for(let o=0;o<8;o++){
      const key=[0,1,2].map(k=>2*signed[r+k]!+((o>>>k)&1)).join(",");
      assert.equal(support[certificate+4+8*b+o],expected.get(key)??0,`stage certificate ${key}`);
     }
    }
    assert.ok(requested.size<legacy.size,"stage union must retire pages the former shared halo required");
    if(step===1||step===169){
     assert.ok(!resident.has("0,0,3"),"empty bottom-left page must retire");
     assert.ok(!requested.has("0,0,3"),"no reader requests the empty corner");
    }
    if(step===1)first=resident.size;else last=resident.size;
    t.diagnostic(JSON.stringify({step,resident:resident.size,requested:requested.size,legacy:legacy.size}));
   }
   assert.ok(first<=8,`initial source needs only eight pages, got ${first}`);
   assert.ok(last>first,"moving fluid claims new pages");
   assert.ok((solver.narrowBandFlipInfo?.particles??0)>1000,"fluid survives and advances");
  }finally{solver.destroy();}
 },["subgroups"]);
});
