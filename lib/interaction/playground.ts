import { studioStageSceneryGraph } from "../core/studio-stage-scene";
import { createEmptyScene } from '../core/empty-scene';
import type { SceneDescription } from '../core/model';
import { interactionCatalog } from './installed';
/** Small authoring/control fixture, also usable directly by the UI handoff. */
export function createInteractivePlayground(wet = false): SceneDescription {
  const scene = createEmptyScene({ sceneId: wet ? 'interactive-pool' : 'interactive-playground', extents_m: { x: 8, y: 4, z: 8 }, finestCellSize_m: .25 });
  const stage = studioStageSceneryGraph(scene);
  // The room has a physical ceiling. Its exterior stage lamp cannot light the
  // interior, so place a practical below the ceiling for authoring visibility.
  scene.scenery = {...stage,nodes:[...stage.nodes,{
    kind:'box',id:'interactive/softbox',tags:['light','emits-negative-y'],
    place:{position:{x:0,y:3.75,z:0},units:'metres'},halfSize:{x:1.5,y:.03,z:1.5},
    material:{colorLinear:[1,.96,.9],emission:12},
  }]};
  scene.numerics.fixedDt_s = 1 / 60; scene.numerics.maxDt_s = 1 / 60;
  scene.systems = { ...scene.systems, fluid: wet };
  scene.container.fillFraction = wet ? .2 : 0;
  scene.fluid.initialCondition = 'tank-fill'; scene.fluid.surfaceTension_N_m = 0;
  const add = (kind: string, id: string, x: number, y: number, z: number) => interactionCatalog.createEntity({kind, id, name: id, transform: {position_m:{x,y,z},orientation:{x:0,y:0,z:0,w:1}}});
  scene.interaction = { schemaVersion:1, entities:[add('fluid.character','player',0,1.2,1),add('fluid.crate','crate',0,.8,-1),add('fluid.switch','switch',1,1,0),add('fluid.platform','lift',-2,.2,0),add('fluid.trigger','trigger',2,.6,2)], connections:[{id:'switch-lift',source:{entity:'switch',behavior:'main',port:'active'},target:{entity:'lift',behavior:'main',port:'active'}}] };
  return scene;
}
