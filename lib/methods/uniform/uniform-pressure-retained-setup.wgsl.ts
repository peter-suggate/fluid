/** Exact producer evidence for the pressure hierarchy. The pressure arena's
 * geometry and coefficients have no non-pressure writers. Iterates, RHS,
 * bounds and job classification are still initialized on every solve.
 *
 * One raw and one continued-phi dirty word per block per level, plus a
 * topology word per block. A block's page identity and all 27 neighbours are
 * compared as well: equal geometry at a new address is not a cache hit.
 * Evidence lives after the pressure job list, without another GPU binding. */
export const uniformPressureRetainedSetupWGSL = /* wgsl */ `
fn mgEvidence()->u32{return 4u+16u*UM_BLOCKS;}
fn mgDirtyWord(l:u32,b:u32,extended:bool)->u32{
 return mgEvidence()+UM_BLOCKS*(1u+l+select(0u,UM_LAST+1u,extended))+b;
}
fn mgTopologyDirty(b:u32)->bool{return atomicLoad(&umList[mgEvidence()+b])!=0u;}
fn mgDirty(l:u32,b:u32,extended:bool)->bool{
 return !UM_RETAIN||mgTopologyDirty(b)||atomicLoad(&umList[mgDirtyWord(l,b,extended)])!=0u;
}
// Downsampling's air-preservation rule reaches all 26 neighbouring blocks;
// extension and baking need subsets. This conservative closure covers both,
// including changed/free neighbours and newly mapped wall slots.
fn mgInputsChanged(l:u32,b:u32,extended:bool)->bool{
 if(mgDirty(l,b,extended)){return true;}
 if(!umStored(b)){return false;}
 for(var k=0u;k<27u;k++){
  let n=umAround(b,vec3i(i32(k%3u)-1,i32((k/3u)%3u)-1,i32(k/9u)-1));
  if(n>0u&&n<UM_BLOCKS-1u&&mgDirty(l,n,extended)){return true;}
 }
 return false;
}
fn mgMarkChanged(l:u32,b:u32,extended:bool){atomicStore(&umList[mgDirtyWord(l,b,extended)],1u);}
fn mgDifferent(a:f32,b:f32)->bool{return bitcast<u32>(a)!=bitcast<u32>(b);}
// Query each block only at the first lane of its run in this workgroup.
// Small levels can put several blocks in one group, and a group can cross
// from cell rows into wall rows (whose block numbering starts over). Index
// by the run's first lane, not by block number, to handle both boundaries.
var<workgroup> mgGroupDirty:array<u32,64>;
fn mgGroupChanged(L:UMLevel,sourceLevel:u32,j:u32,lane:u32,extended:bool,cellsOnly:bool)->bool{
 let count=select(umLRows(L),L.cells,cellsOnly);let valid=j<count;
 var slot=0u;
 if(valid){
  let b=umLBlock(L,j);var first=b<<(3u*L.sh);
  if(j>=L.cells){first=L.cells+b*(6u<<(2u*L.sh));}
  first=max(first,j-lane);slot=first-(j-lane);
  if(j==first){mgGroupDirty[slot]=u32(mgInputsChanged(sourceLevel,b,false)||(extended&&mgInputsChanged(sourceLevel,b,true)));}
 }
 workgroupBarrier();
 var changed=false;if(valid){changed=mgGroupDirty[slot]!=0u;}return changed;
}
// Called by one lane per block, before any hierarchy consumer executes.
fn mgCheckTopology(b:u32){
 let records=5u*UM_KEYS+UM_PAGE_WORDS;
 let saved=mgEvidence()+(1u+2u*(UM_LAST+1u))*UM_BLOCKS+28u*b;
 var changed=!UM_RETAIN;
 let page=umTopology[records+b];
 changed=changed||atomicLoad(&umList[saved])!=page;atomicStore(&umList[saved],page);
 for(var k=0u;k<27u;k++){
  let value=umTopology[records+UM_BLOCKS+27u*b+k];
  changed=changed||atomicLoad(&umList[saved+1u+k])!=value;
  atomicStore(&umList[saved+1u+k],value);
 }
 if(changed){atomicStore(&umList[mgEvidence()+b],1u);}
}
`;
