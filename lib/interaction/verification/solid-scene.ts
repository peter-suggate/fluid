import { createEmptyScene } from '../../core/empty-scene';
import { bakeProceduralSolid } from '../../core/procedural-solid';
import { interactionCatalog } from '../installed';
/** Browser/GPU verification fixture: the cavity straddles both eight-cell seams. */
export function createInteractiveSolidVerificationScene(wet = true) {
 const scene=createEmptyScene({sceneId:'interactive-solid-verification',extents_m:{x:4,y:4,z:4},finestCellSize_m:.25,environment:'default'});
 scene.systems={fluid:wet};scene.container.fillFraction=wet?.3:0;scene.fluid.surfaceTension_N_m=0;
 scene.numerics.fixedDt_s=1/60;scene.numerics.maxDt_s=1/60;
 const baked=bakeProceduralSolid([16,16,16],[.25,.25,.25],p=>Math.max(p[1]-1-p[0]*.2,2.5-Math.hypot(p[0]-8,p[2]-8)));
 scene.solidVoxels=[...baked.solidVoxels,...scene.solidVoxels];scene.solidBoundary=baked.solidBoundary;
 scene.scenery={palettes:{},nodes:[{kind:'room-shell',id:'shell',faces:['floor'],materialModel:'room',floor:{colorLinear:[.6,.6,.6]},wall:{colorLinear:[.6,.6,.6]},ceiling:{colorLinear:[.6,.6,.6]}},
 {kind:'box',id:'test/light',tags:['light','emits-negative-y'],place:{position:{x:0,y:3.75,z:0},units:'metres'},halfSize:{x:1,y:.03,z:1},material:{colorLinear:[1,1,1],emission:12}}]};
 const add=(kind:string,id:string,x:number,y:number,z:number)=>interactionCatalog.createEntity({kind,id,transform:{position_m:{x,y,z},orientation:{x:0,y:0,z:0,w:1}}});
 scene.interaction={schemaVersion:1,entities:[add('fluid.character','player',-1,1.8,-.8),add('fluid.crate','crate',1,1.5,1)],connections:[]};
 return scene;
}
