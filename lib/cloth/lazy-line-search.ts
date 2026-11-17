import { CLOTH_LINE_SEARCH } from './oracle';
import { CLOTH_NEWTON_TRIALS } from './gpu-newton-wgsl';

/** An exact speculative prefix. The original full ladder remains the rejection path. */
export function withLazyClothLineSearch(source: string, headTrials: 1 | 2): string {
  const start=source.indexOf('@compute @workgroup_size(',source.indexOf('fn nwAdd('));
  const end=source.indexOf('// A trust-region iteration\'s model',start);
  if(start<0 || end<0 || !source.slice(start,end).includes('fn nwTrial(')) throw new Error('Missing cloth energy ladder');
  const original=source.slice(start,end);
  const loops=original.split(`k < ${CLOTH_NEWTON_TRIALS}u`).length-1;
  if(![4,6].includes(loops) || source.split('fn nwBeginStep() {').length!==2) throw new Error('Cloth energy ladder structure changed');
  const head=original.replace('fn nwTrial(', 'fn nwTrialHead(').replaceAll(`k < ${CLOTH_NEWTON_TRIALS}u`, `k < ${headTrials}u`);
  const tail=original.replace('fn nwTrial(', 'fn nwTrialTail(').replace('  let ctl = nwControlUniform(lane);', `  if(lane==0u){nwLazyDone=u32(nw[nwControl()+7u].x);}
  if(workgroupUniformLoad(&nwLazyDone)!=0u){return;}
  let ctl = nwControlUniform(lane);`);
  const gate=/* wgsl */`
// Spare control record +7 is execution-only. Existing control and trust records are untouched.
var<workgroup> nwLazyDone:u32;
@compute @workgroup_size(NW_WG) fn nwTrialHeadGate(@builtin(local_invocation_index) lane:u32){
  let ctl=nwControlUniform(lane);
  if(ctl.x!=0.0){if(lane==0u){nw[nwControl()+7u].x=0.0;}return;}
  let base=nwPartial(0u);
  let E=nwSum(lane,base);let F=nwSum(lane,base+2u*P.newton0.z);let S=nwSum(lane,base+4u*P.newton0.z);
  if(lane!=0u){return;}
  var accepted=false;
  for(var k=0u;k<${headTrials}u;k++){
    if(S[k]<0.0 && E[k]<=${CLOTH_LINE_SEARCH.armijo}*S[k]+4.0*EPS32*F[k]){accepted=true;break;}
  }
  let before=nw[nwControl()+7u];
  nw[nwControl()+7u]=vec4f(select(0.0,1.0,accepted),before.y+1.0,before.z+select(1.0,0.0,accepted),0.0);
}
`;
  return source.replace('fn nwBeginStep() {','fn nwBeginStep() {\n  nw[nwControl()+7u]=vec4f(0.0);')+'\n'+head+tail+gate;
}
