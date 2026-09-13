import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { SceneDescription } from "../../lib/core/model";
import { sceneDocument } from "../../lib/core/scene-definition";
import { sceneLatticeDimensions } from "../../lib/core/scene-lattice";
import { SCENE_CATALOG } from "../../lib/core/scenes";
import { fluidSolidWorldForScene } from "../../lib/core/solid-world";

const target=resolve(import.meta.dirname,"../../rust/core/testdata/solid-world-golden.json");
const definition=(id:string)=>{const d=SCENE_CATALOG.find(v=>v.id===id);if(!d)throw new Error(id);return d};
const base=()=>sceneDocument(definition("water-box-dam-break"));
const analytic=base();analytic.sceneId="solid-world-analytic-terrain";analytic.terrain={baseHeight_m:.13,features:[
  {kind:"basin",center_m:{x:-.17,z:.09},radius_m:{x:.39,z:.27},amount_m:.1,rotation_rad:.41,flat:.22},
  {kind:"basin",center_m:{x:.08,z:-.03},radius_m:{x:.22,z:.31},amount_m:.07,rotation_rad:-.27},
  {kind:"mound",center_m:{x:.31,z:.16},radius_m:{x:.28,z:.2},amount_m:.16,rotation_rad:.72,flat:.1},
]};analytic.solidVoxels=[...analytic.solidVoxels,
  {operation:"fill",minimum:[3,1,4],maximumExclusive:[10,5,9],materialId:17},
  {operation:"clear",minimum:[5,2,5],maximumExclusive:[8,4,8]},
  {operation:"fill",minimum:[6,3,6],maximumExclusive:[9,6,10],materialId:23},
];
const grid=base();grid.sceneId="solid-world-grid-terrain";grid.terrain={baseHeight_m:0,features:[],grid:{kind:"grid",origin_m:{x:-.6,z:-.4},spacing_m:.2,size:{nx:7,nz:5},heights_m:Array.from({length:35},(_,i)=>.025+.031*(i%7)+.017*Math.floor(i/7))}};
const scenery=base();scenery.sceneId="solid-world-scenery-transforms";scenery.scenery={palettes:{x:{tint:[1,1,1]}},nodes:[
  {kind:"room-shell",id:"shell",materialModel:"room",faces:[],floor:{palette:"x",value:.1},wall:{palette:"x",value:.1},ceiling:{palette:"x",value:.1}},
  {kind:"group",id:"translated",place:{position:{x:.1,y:.12,z:-.08},units:"metres",scale:1.25},children:[
    {kind:"box",id:"collider",tags:["fluid-collider"],place:{position:{x:.07,y:0,z:.03},units:"metres"},halfSize:{x:.12,y:.04,z:.09},material:{palette:"x",value:.2}},
  ]},
  {kind:"box",id:"rotated-excluded",tags:["fluid-collider"],place:{position:{x:-.2,y:.1,z:0},units:"metres",orientation:{w:.9238795325,x:0,y:.3826834324,z:0}},halfSize:{x:.2,y:.05,z:.1},material:{palette:"x",value:.2}},
]};
function freeze(scene:SceneDescription){const world=fluidSolidWorldForScene(scene);return{scene,dimensions:sceneLatticeDimensions(scene),pages:world.pages.map(p=>({coordinate:p.coordinate,solidFraction:Array.from(p.solidFraction),signedDistanceQ8:Array.from(p.signedDistanceQ8),materialId:Array.from(p.materialId)})),regions:world.regions??[]}}
const hero=sceneDocument(definition("hero-garden-hose"));hero.sceneId="solid-world-production-pond-coarse";
hero.voxelDomain={...hero.voxelDomain,finestCellSize_m:.4};hero.solidVoxels=[];
hero.scenery={palettes:{x:{tint:[1,1,1]}},nodes:[{kind:"terrain-shell",id:"shell",materialModel:"garden-terrain"}]};
const artifact=JSON.stringify({schemaVersion:1,cases:[freeze(base()),freeze(analytic),freeze(grid),freeze(scenery),freeze(hero)]},null,2)+"\n";
if(process.argv.includes("--check")){if(readFileSync(target,"utf8")!==artifact)throw new Error(`${target} is stale`)}else{mkdirSync(resolve(target,".."),{recursive:true});writeFileSync(target,artifact);console.log(target)}
