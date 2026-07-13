import { cloneScene, defaultScene, type SceneDescription } from "./model";

export type ScenarioId="dam-break"|"wave-tank"|"buoyancy"|"splash-impact"|"hydrostatic";

export interface ScenarioPreset {id:ScenarioId;name:string;description:string;stress:string}

export const SCENARIOS:readonly ScenarioPreset[]=[
  {id:"dam-break",name:"Dam break",description:"Collapsing corner reservoir",stress:"moving interface"},
  {id:"wave-tank",name:"Wave tank",description:"Fast immersed paddle launch",stress:"wake + coupling"},
  {id:"buoyancy",name:"Buoyancy",description:"Light and dense bodies submerged",stress:"solid halos"},
  {id:"splash-impact",name:"Splash impact",description:"Dense sphere strikes a shallow pool",stress:"rapid refinement"},
  {id:"hydrostatic",name:"Still water",description:"Quiescent deep-water reference",stress:"equilibrium"}
] as const;

export function createScenarioScene(id:ScenarioId):SceneDescription {
  const scene=cloneScene(defaultScene),sphere={...scene.rigidBodies[0],dimensions_m:{...scene.rigidBodies[0].dimensions_m}},box={...scene.rigidBodies[1],dimensions_m:{...scene.rigidBodies[1].dimensions_m}};
  scene.sceneId=`hierarchy-${id}`;scene.duration_s=16;scene.hierarchy.levels=3;scene.hierarchy.interfaceHaloCells=3;scene.hierarchy.solidHaloCells=3;
  if(id==="dam-break"){scene.container.fillFraction=.22;scene.fluid.initialCondition="dam-break";scene.rigidBodies=[];return scene;}
  scene.fluid.initialCondition="tank-fill";
  if(id==="wave-tank"){
    scene.container.fillFraction=.48;scene.container.fluidWallMode="free-slip";scene.hierarchy.velocityErrorTolerance=.045;
    scene.rigidBodies=[{...box,id:"wave-paddle",name:"Wave paddle",density_kg_m3:3200,dimensions_m:{x:.10,y:.34,z:.58},position_m:{x:-.42,y:.24,z:0},linearVelocity_m_s:{x:1.8,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},restitution:.05,friction:.12}];
  }else if(id==="buoyancy"){
    scene.container.fillFraction=.68;scene.hierarchy.solidHaloCells=4;
    scene.rigidBodies=[
      {...sphere,id:"light-sphere",name:"Light sphere",density_kg_m3:280,dimensions_m:{x:.095,y:.095,z:.095},position_m:{x:-.2,y:.28,z:0},linearVelocity_m_s:{x:0,y:0,z:0}},
      {...sphere,id:"dense-sphere",name:"Dense sphere",density_kg_m3:1850,dimensions_m:{x:.085,y:.085,z:.085},position_m:{x:.2,y:.52,z:0},linearVelocity_m_s:{x:0,y:0,z:0}}
    ];
  }else if(id==="splash-impact"){
    scene.container.fillFraction=.32;scene.hierarchy.interfaceHaloCells=4;scene.hierarchy.solidHaloCells=4;scene.hierarchy.regridInterval=6;
    scene.rigidBodies=[{...sphere,id:"impact-sphere",name:"Impact sphere",density_kg_m3:2400,dimensions_m:{x:.11,y:.11,z:.11},position_m:{x:0,y:1.18,z:0},linearVelocity_m_s:{x:.18,y:-2.8,z:0},angularVelocity_rad_s:{x:0,y:1.2,z:.4},restitution:.12}];
  }else{
    scene.container.fillFraction=.62;scene.rigidBodies=[];scene.hierarchy.interfaceHaloCells=2;scene.hierarchy.regridInterval=16;
  }
  return scene;
}
