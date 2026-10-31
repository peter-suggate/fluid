import {test} from "node:test";
import assert from "node:assert/strict";
import {createBackyardPoolPartyScene} from "../lib/core/backyard-pool-party-scene";
import {createUniformTroughScene} from "../lib/core/uniform-trough-scenes";
import {sceneFluidBoxPage, sceneFluidLatticeDimensions, sceneLatticeDimensions} from "../lib/core/scene-lattice";
import {createInflowGridBoundary} from "../lib/core/inflow-boundary";
import {UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_SEED_MARGIN, uniformBoxShellTilesIn, uniformPagePoolResidency, uniformPagePoolRuleBlocks, uniformPagePoolRulePages} from "../lib/methods/uniform/uniform-page-pool";
import {uniformCensusReach, uniformPagePoolRule} from "../lib/methods/uniform/uniform-particle-census";
import {uniformLatticePagedVolume} from "../lib/methods/uniform/uniform-volume-initial";
import {uniformPageSources} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {SceneDescription} from "../lib/core/model";

const g=9.81;
const census=(scene:SceneDescription)=>{
 const cells=sceneFluidLatticeDimensions(scene,Number.MAX_SAFE_INTEGER),plug=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,sceneLatticeDimensions(scene,Number.MAX_SAFE_INTEGER)):undefined;
 const volume=uniformLatticePagedVolume(scene,UNIFORM_PAGE_POOL_EDGE).volume;
 const {air}=uniformPagePoolResidency(cells,scene.fluid.pagePool,uniformPageSources(scene,plug,scene.numerics.maxDt_s),true,()=>volume,sceneFluidBoxPage(scene));
 return {volume,air};
};

test("a reserve column reaches the head above its top resident page, capped by a closed top",()=>{
 // Two columns of three pages: one resident at the bottom only, one resident to its top.
 const air=[[0,1,0],[0,2,0]] as const;
 assert.equal(uniformPagePoolRuleBlocks([2,3,1],air,2,2),4+2+2);
 assert.equal(uniformPagePoolRuleBlocks([2,3,1],air,9,2),4+2+2);
 assert.equal(uniformPagePoolRuleBlocks([2,3,1],air,9,Number.MAX_SAFE_INTEGER),4+(9-0)+(9-2)+2);
 assert.equal(uniformPagePoolRuleBlocks([2,3,1],air,0,2),4+2);
});

test("liquid held by walls on every side makes no dam front; a dam's free face does",()=>{
 const tank=createUniformTroughScene("settled-tank"),dam=createUniformTroughScene("dam-break");
 assert.equal(uniformCensusReach(tank,census(tank).volume).v_m_s,0);
 const reach=uniformCensusReach(dam,census(dam).volume),h=dam.voxelDomain.finestCellSize_m;
 // Ritter: 2 sqrt(g H) over the dam column, H within a cell of the liquid's top above the floor.
 assert.ok(reach.v_m_s>2*Math.sqrt(g*(reach.top_m-4*h))&&reach.v_m_s<=2*Math.sqrt(g*reach.top_m)+1e-9,`${reach.v_m_s}`);
});

test("the backyard's pool follows the census, not the box",()=>{
 const scene=createBackyardPoolPartyScene(),{volume,air}=census(scene),[px,py,pz]=volume.pages,resident=px*py*pz-air.length;
 const blocks=uniformPagePoolRule(scene,volume,air)!;
 assert.ok(blocks>=resident+2&&blocks<px*py*pz,`${blocks} of ${px*py*pz} box pages, ${resident} resident`);
});

test("without gravity the head is the closure margin at rest, and unbounded for a moving body",()=>{
 const tank=createUniformTroughScene("settled-tank");tank.fluid.gravity_m_s2={x:0,y:0,z:0};
 const {volume,air}=census(tank),reach=uniformCensusReach(tank,volume);
 assert.equal(reach.head_m,UNIFORM_PAGE_POOL_SEED_MARGIN*tank.voxelDomain.finestCellSize_m);
 assert.ok(Number.isInteger(uniformPagePoolRule(tank,volume,air)));
 const yard=createBackyardPoolPartyScene();yard.fluid.gravity_m_s2={x:0,y:0,z:0};
 Object.assign(yard.rigidBodies[0]!,{linearVelocity_m_s:{x:0,y:-1,z:0}});
 const c=census(yard);
 assert.equal(uniformCensusReach(yard,c.volume).head_m,Infinity);
 assert.equal(yard.container.top,"open");
 assert.equal(uniformPagePoolRule(yard,c.volume,c.air),undefined,"an open top with no bound keeps the box default");
});

test("the rule's pages are the census and its reserve, as many as its blocks less the sentinels",()=>{
 const air=[[0,1,0],[0,2,0]] as const;
 for(const [reach,ceiling] of [[2,2],[9,2],[0,2],[1,2]] as const){
  const flags=uniformPagePoolRulePages([2,3,1],air,reach,ceiling);
  assert.equal(flags.reduce((n,f)=>n+(f?1:0),0),uniformPagePoolRuleBlocks([2,3,1],air,reach,ceiling)-2);
 }
 assert.deepEqual([...uniformPagePoolRulePages([2,3,1],air,1,2)],[1,1,2,1,0,1]);
 // An open top's reserve past the box's top page holds no box page.
 assert.deepEqual([...uniformPagePoolRulePages([2,3,1],air,9,Number.MAX_SAFE_INTEGER)],[1,1,2,1,2,1]);
});

test("V2d: the walls' solid-record tiles are those of the rule's pages, so they follow the water, not the box",()=>{
 const shell=(t:readonly number[])=>t.reduce((n,b)=>n*b,1)-t.reduce((n,b)=>n*Math.max(0,b-2),1);
 // Every page flagged: the box's whole outer layer, a partial last page included.
 for(const tiles of [[16,24,8],[13,24,8],[16,17,3]] as const)
  assert.equal(uniformBoxShellTilesIn([2,3,1],new Uint8Array(6).fill(1),tiles),shell(tiles));
 // One bottom page of a 2x3x1-page box: its floor, its low x face and both z faces.
 assert.equal(uniformBoxShellTilesIn([2,3,1],Uint8Array.of(1,0,0,0,0,0),[16,24,8]),8*8*8-7*7*6);
 // The same water in a box four times as wide: the wide box's extra pages are AIR, outside the rule.
 const narrow=uniformBoxShellTilesIn([2,3,1],uniformPagePoolRulePages([2,3,1],[[0,2,0],[1,2,0]],1,2),[16,24,8]);
 const wideAir:[number,number,number][]=[];for(let y=0;y<3;y++)for(let x=0;x<8;x++)if(x>=2||y===2)wideAir.push([x,y,0]);
 const wide=uniformBoxShellTilesIn([8,3,1],uniformPagePoolRulePages([8,3,1],wideAir,1,2),[64,24,8]);
 assert.ok(wide<=narrow&&narrow<shell([16,24,8]),`${wide} wide, ${narrow} narrow, ${shell([64,24,8])} wide box`);
});
