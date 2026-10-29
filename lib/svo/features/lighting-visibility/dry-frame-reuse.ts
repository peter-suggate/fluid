/** Exact reuse of persistent dry HDR + its unchanged auxiliary G-buffer.
 * No temporal interpolation or quantization: a hit encodes no writes at all.
 * Callers must retain the target and supply every mutation receipt below.
 */
export interface DryFrameReuseKey {
  readonly viewRevision:number;
  readonly parameterRevision:number;
  readonly controlRevision:number;
  readonly sourceRevision:number;
  readonly sceneRevision:number;
  readonly meshReceipt:string;
  readonly sceneMaintenanceActivity:number;
  readonly derivedContentRevision:number;
  readonly source:object;
  readonly scene:object;
  readonly bindings:object;
  readonly pipeline:object;
  readonly optimizedPipeline?:object;
  readonly precomputedPipeline?:object;
  readonly target:object;
  readonly gBuffer:object;
}
export interface DryFrameReuseEligibility {
  readonly enabled:boolean;
  readonly viewValid:boolean;
  readonly splitReady:boolean;
  readonly meshReady:boolean;
  readonly derivedSettled:boolean;
  readonly hasBodies:boolean;
  readonly waterShadows:boolean;
  readonly diagnostic:boolean;
  readonly evolvingVisibilityCache:boolean;
}
export function dryFrameReuseReason(state:DryFrameReuseEligibility):string {
  if(!state.enabled)return "disabled";
  if(!state.viewValid)return "view-unavailable";
  if(!state.splitReady)return "split-unavailable";
  if(!state.meshReady)return "mesh-pending";
  if(!state.derivedSettled)return "derived-lighting-settling";
  if(state.hasBodies)return "dynamic-bodies";
  if(state.waterShadows)return "water-shadows";
  if(state.diagnostic)return "diagnostic-active";
  if(state.evolvingVisibilityCache)return "persistent-lighting-cache-active";
  return "eligible";
}
export function canReuseDryFrame(state:DryFrameReuseEligibility):boolean {
  return dryFrameReuseReason(state)==="eligible";
}
export class ExactDryFrameReuse {
  private previous?:DryFrameReuseKey;
  private hits=0;
  private rendered=0;
  invalidate():void {this.previous=undefined;}
  matches(key:DryFrameReuseKey|undefined):boolean {
    if(!key){this.previous=undefined;return false;}
    const previous=this.previous;
    if(!previous)return false;
    if(!(Object.keys(key) as (keyof DryFrameReuseKey)[]).every(field=>Object.is(key[field],previous[field])))return false;
    this.hits++;return true;
  }
  publish(key:DryFrameReuseKey|undefined):void {this.previous=key;this.rendered++;}
  get statistics(){return {reusedFrames:this.hits,renderedFrames:this.rendered,retained:this.previous!==undefined};}
}
