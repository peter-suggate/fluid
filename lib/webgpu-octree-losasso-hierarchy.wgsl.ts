/** GPU construction of the compact algebraic Losasso multigrid hierarchy. */
export const octreeLosassoHierarchyWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
const ERROR_CAPACITY:u32=1u;
const ERROR_GEOMETRY:u32=2u;

struct Params {
  dimensionsTarget:vec4u,
  capacities:vec4u,
  cellSize:vec4f,
  fused:vec4u,
}
struct LeafHeader {
  cell:u32,entryStart:u32,entryCount:u32,size:u32,
  diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f,
}
struct Face {
  negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
  area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32,
}
struct AlgebraicEdge { rowA:u32,rowB:u32,coefficient:f32,reserved:u32 }

@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> leafHeaders:array<LeafHeader>;
@group(0)@binding(2)var<storage,read> fineControl:array<u32>;
@group(0)@binding(3)var<storage,read_write> fineCells:array<vec4u>;
@group(0)@binding(4)var<storage,read> fineFaces:array<u32>;
@group(0)@binding(5)var<storage,read_write> coarseControl:array<u32>;
@group(0)@binding(6)var<storage,read_write> coarseCells:array<vec4u>;
@group(0)@binding(7)var<storage,read_write> coarseFaces:array<u32>;
@group(0)@binding(8)var<storage,read_write> coarseOffsets:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write> coarseRowFaces:array<u32>;
@group(0)@binding(10)var<storage,read_write> coarseDispatch:array<u32>;
@group(0)@binding(11)var<storage,read_write> fineParents:array<u32>;
@group(0)@binding(12)var<storage,read_write> childOffsets:array<u32>;
@group(0)@binding(13)var<storage,read_write> childList:array<u32>;
@group(0)@binding(14)var<storage,read_write> directory:array<atomic<u32>>;
@group(0)@binding(16)var<storage,read_write> coarseFaceSources:array<u32>;
@group(0)@binding(17)var<storage,read_write> fusedArena:array<u32>;
@group(0)@binding(18)var<storage,read_write> edgeScratch:array<atomic<u32>>;
@group(0)@binding(19)var<storage,read_write> coarseFaceSourceOffsets:array<atomic<u32>>;
@group(0)@binding(20)var<storage,read> fineOffsets:array<u32>;
@group(0)@binding(21)var<storage,read> fineRowFaces:array<u32>;

var<workgroup> hierarchyPrefixTotals:array<u32,256>;
// The resident-solve tier never exceeds 4K rows.  Its hierarchy CSR can count
// face incidences in shared memory, scatter them once, and restore the exact
// ascending face-id order locally instead of scanning every face for every
// row twice.  Larger hierarchies retain the wide fallback below.
var<workgroup> hierarchyRowCursors:array<atomic<u32>,4096>;

fn fusedControl()->u32{return p.fused.x;}
fn fusedOffsets()->u32{return fusedControl()+8u;}
fn arenaAlign(words:u32)->u32{return ((words+63u)/64u)*64u;}
fn fusedDirectedEdges()->u32{return fusedControl()+arenaAlign(8u+p.capacities.y+1u);}
fn fusedParents()->u32{return fusedControl()+arenaAlign(
 (fusedDirectedEdges()-fusedControl())+2u*p.capacities.z);}
fn fusedChildOffsets()->u32{return fusedControl()+arenaAlign(
 (fusedParents()-fusedControl())+p.capacities.x);}
fn fusedChildList()->u32{return fusedControl()+arenaAlign(
 (fusedChildOffsets()-fusedControl())+p.capacities.y+1u);}
fn storeEdge(bufferIndex:u32,edge:AlgebraicEdge){let base=4u*bufferIndex;
 coarseFaces[base]=edge.rowA;coarseFaces[base+1u]=edge.rowB;
 coarseFaces[base+2u]=bitcast<u32>(edge.coefficient);coarseFaces[base+3u]=edge.reserved;
}
fn loadEdge(bufferIndex:u32)->AlgebraicEdge{let base=4u*bufferIndex;
 return AlgebraicEdge(coarseFaces[base],coarseFaces[base+1u],
  bitcast<f32>(coarseFaces[base+2u]),coarseFaces[base+3u]);
}
fn storeDirectedEdge(at:u32,neighbour:u32,coefficient:f32){let base=fusedDirectedEdges()+2u*at;
 fusedArena[base]=neighbour;fusedArena[base+1u]=bitcast<u32>(coefficient);
}

fn hashCell(cell:vec4u)->u32{
 var hash=(cell.w+1u)*0x9e3779b1u;
 hash=(hash^cell.x)*0x85ebca6bu;hash=(hash^cell.y)*0xc2b2ae35u;
 return (hash^cell.z)*0x27d4eb2du;
}
fn cellVolume(size:u32)->u32{
 if(size==0u||size>1625u){return 0u;}return size*size*size;
}
fn controlValid()->bool{
 return arrayLength(&fineControl)>=5u&&fineControl[3]==1u&&fineControl[4]==0u;
}
fn parentCell(fine:vec4u)->vec4u{let parentSize=max(fine.w,p.dimensionsTarget.w);
 return vec4u((fine.xyz/parentSize)*parentSize,parentSize);}

@compute @workgroup_size(64)
fn extractLosassoFinestCells(@builtin(global_invocation_id)g:vec3u){
 if(fineControl[5]==1u){return;}
 let row=g.x;if(row>=min(fineControl[1],p.capacities.x)){return;}
 let header=leafHeaders[row];let dimensions=p.dimensionsTarget.xyz;
 let origin=vec3u(header.cell%dimensions.x,
  (header.cell/dimensions.x)%dimensions.y,
  header.cell/(dimensions.x*dimensions.y));
 fineCells[row]=vec4u(origin,header.size);
}

// Small resident tier: hash parents cooperatively, choose the minimum fine-row
// representative for the exact original first-occurrence row order, then sort
// each compact child list back to ascending fine row.
@compute @workgroup_size(256)
fn buildLosassoParentRowsSmall(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 for(var word=lane;word<8u;word+=256u){coarseControl[word]=0u;}
 if(lane==0u){atomicStore(&hierarchyRowCursors[4095],0u);}
 let fineCount=min(fineControl[1],p.capacities.x);
 for(var slot=lane;slot<p.capacities.w;slot+=256u){
  atomicStore(&directory[2u*slot],0u);atomicStore(&directory[2u*slot+1u],0u);}
 storageBarrier();workgroupBarrier();
 if(!controlValid()&&lane==0u){atomicOr(&hierarchyRowCursors[4095],ERROR_GEOMETRY);}
 for(var row=lane;row<fineCount;row+=256u){let parent=parentCell(fineCells[row]);
  let volume=cellVolume(fineCells[row].w);var installedSlot=INVALID;
  if(volume==0u||any(parent.xyz+vec3u(parent.w)>p.dimensionsTarget.xyz)){
   atomicOr(&hierarchyRowCursors[4095],ERROR_GEOMETRY);
  }else{let hash=hashCell(parent);let mask=p.capacities.w-1u;
   for(var probe=0u;probe<p.capacities.w;probe+=1u){let slot=(hash+probe)&mask;let at=2u*slot;
    var claim=atomicCompareExchangeWeak(&directory[at],0u,row+1u);
    for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){
     claim=atomicCompareExchangeWeak(&directory[at],0u,row+1u);}
    if(claim.exchanged){atomicStore(&directory[at+1u],hash);installedSlot=slot;break;}
    let representative=atomicLoad(&directory[at]);
    if(representative>0u&&representative<=fineCount
      &&all(parentCell(fineCells[representative-1u])==parent)){
     atomicMin(&directory[at],row+1u);installedSlot=slot;break;}
   }
  }
  childOffsets[row]=installedSlot;
  if(installedSlot==INVALID){atomicOr(&hierarchyRowCursors[4095],ERROR_CAPACITY);}
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u){var coarseCount=0u;
  coarseControl[4]|=atomicLoad(&hierarchyRowCursors[4095]);
  for(var row=0u;row<fineCount;row+=1u){let slot=childOffsets[row];
   if(slot!=INVALID&&atomicLoad(&directory[2u*slot])==row+1u){
    if(coarseCount>=p.capacities.y){coarseControl[4]|=ERROR_CAPACITY;break;}
    fineParents[row]=coarseCount;coarseCells[coarseCount]=parentCell(fineCells[row]);coarseCount+=1u;
   }
  }
  coarseControl[0]=fineControl[0];coarseControl[1]=coarseCount;
  coarseControl[3]=select(0u,1u,coarseControl[4]==0u&&coarseCount!=0u);
 }
 storageBarrier();workgroupBarrier();
 let coarseCount=coarseControl[1];
 for(var parentRow=lane;parentRow<coarseCount;parentRow+=256u){
  atomicStore(&hierarchyRowCursors[parentRow],0u);}
 workgroupBarrier();
 for(var row=lane;row<fineCount;row+=256u){let slot=childOffsets[row];
  if(slot==INVALID){continue;}let representative=atomicLoad(&directory[2u*slot])-1u;
  let parentRow=fineParents[representative];fineParents[row]=parentRow;
  fusedArena[fusedParents()+row]=parentRow;atomicAdd(&hierarchyRowCursors[parentRow],1u);
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u){var childTotal=0u;
  for(var parentRow=0u;parentRow<coarseCount;parentRow+=1u){
   let children=atomicLoad(&hierarchyRowCursors[parentRow]);
   childOffsets[parentRow]=childTotal;fusedArena[fusedChildOffsets()+parentRow]=childTotal;
   atomicStore(&hierarchyRowCursors[parentRow],childTotal);childTotal+=children;
  }
  childOffsets[coarseCount]=childTotal;fusedArena[fusedChildOffsets()+coarseCount]=childTotal;
 }
 storageBarrier();workgroupBarrier();
 for(var row=lane;row<fineCount;row+=256u){let parentRow=fineParents[row];
  let at=atomicAdd(&hierarchyRowCursors[parentRow],1u);childList[at]=row;}
 storageBarrier();workgroupBarrier();
 for(var parentRow=lane;parentRow<coarseCount;parentRow+=256u){let begin=childOffsets[parentRow];
  let end=childOffsets[parentRow+1u];
  for(var at=begin+1u;at<end;at+=1u){let value=childList[at];var cursor=at;
   while(cursor>begin&&childList[cursor-1u]>value){childList[cursor]=childList[cursor-1u];cursor-=1u;}
   childList[cursor]=value;
  }
  for(var at=begin;at<end;at+=1u){fusedArena[fusedChildList()+at]=childList[at];}
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u){coarseDispatch[0]=select(0u,(coarseCount+63u)/64u,coarseControl[3]==1u);
  coarseDispatch[1]=1u;coarseDispatch[2]=1u;coarseDispatch[3]=0u;
  coarseDispatch[4]=1u;coarseDispatch[5]=1u;}
}

// Candidate rows are compact but mixed-size, so their geometric parents are
// not necessarily adjacent. A deterministic singleton inserts them into an
// open-addressed directory and gives every fine row exactly one parent.
@compute @workgroup_size(256)
fn buildLosassoParentRows(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 for(var word=lane;word<8u;word+=256u){coarseControl[word]=0u;}
 let fineCount=min(fineControl[1],p.capacities.x);
 for(var slot=lane;slot<p.capacities.w;slot+=256u){
  atomicStore(&directory[2u*slot],0u);atomicStore(&directory[2u*slot+1u],0u);}
 for(var row=lane;row<p.capacities.y;row+=256u){childOffsets[row]=0u;}
 storageBarrier();workgroupBarrier();
 if(lane==0u){
  if(!controlValid()){coarseControl[4]=ERROR_GEOMETRY;}
  else{var coarseCount=0u;let mask=p.capacities.w-1u;
   for(var row=0u;row<fineCount;row+=1u){
    let fine=fineCells[row];let parent=parentCell(fine);let parentSize=parent.w;
    let volume=cellVolume(fine.w);
    if(volume==0u||any(parent.xyz+vec3u(parentSize)>p.dimensionsTarget.xyz)){
     coarseControl[4]|=ERROR_GEOMETRY;continue;
    }
    let hash=hashCell(parent);var parentRow=INVALID;
    for(var probe=0u;probe<p.capacities.w;probe+=1u){
     let slot=(hash+probe)&mask;let installed=atomicLoad(&directory[2u*slot]);
     if(installed==0u){
      if(coarseCount>=p.capacities.y){coarseControl[4]|=ERROR_CAPACITY;break;}
      parentRow=coarseCount;coarseCount+=1u;coarseCells[parentRow]=parent;
      atomicStore(&directory[2u*slot],parentRow+1u);
      atomicStore(&directory[2u*slot+1u],hash);break;
     }
     let candidate=installed-1u;
     if(atomicLoad(&directory[2u*slot+1u])==hash&&all(coarseCells[candidate]==parent)){
      parentRow=candidate;break;
     }
    }
    if(parentRow==INVALID){coarseControl[4]|=ERROR_CAPACITY;continue;}
    fineParents[row]=parentRow;fusedArena[fusedParents()+row]=parentRow;
    childOffsets[parentRow]+=1u;
   }
   var childTotal=0u;
   for(var parentRow=0u;parentRow<coarseCount;parentRow+=1u){
    let children=childOffsets[parentRow];
    fusedArena[fusedChildOffsets()+parentRow]=childTotal;
    childTotal+=children;childOffsets[parentRow]=0u;
   }
   childOffsets[coarseCount]=childTotal;
   fusedArena[fusedChildOffsets()+coarseCount]=childTotal;
   for(var row=0u;row<fineCount;row+=1u){let parentRow=fineParents[row];
    let slot=fusedArena[fusedChildOffsets()+parentRow]+childOffsets[parentRow];childList[slot]=row;
    fusedArena[fusedChildList()+slot]=row;
    childOffsets[parentRow]+=1u;
   }
   for(var parentRow=0u;parentRow<coarseCount;parentRow+=1u){
    childOffsets[parentRow]=fusedArena[fusedChildOffsets()+parentRow];
   }
   coarseControl[0]=fineControl[0];coarseControl[1]=coarseCount;
   coarseControl[3]=select(0u,1u,coarseControl[4]==0u&&coarseCount!=0u);
  }
 }
 storageBarrier();workgroupBarrier();
 let coarseCount=coarseControl[1];
 if(lane==0u){coarseDispatch[0]=select(0u,(coarseCount+63u)/64u,coarseControl[3]==1u);
  coarseDispatch[1]=1u;coarseDispatch[2]=1u;
  coarseDispatch[3]=0u;coarseDispatch[4]=1u;coarseDispatch[5]=1u;}
}


fn fineFaceStride()->u32{return select(8u,4u,p.fused.y==1u);}
fn fineRows(faceId:u32)->vec2u{let base=fineFaceStride()*faceId;
 return vec2u(fineFaces[base],fineFaces[base+1u]);
}
fn fineCoefficient(faceId:u32)->f32{let base=fineFaceStride()*faceId;
 if(p.fused.y==1u){return bitcast<f32>(fineFaces[base+2u]);}
 return (bitcast<f32>(fineFaces[base+4u])*bitcast<f32>(fineFaces[base+6u]))
  *bitcast<f32>(fineFaces[base+5u]);
}
fn canonicalParents(faceId:u32)->vec2u{let rows=fineRows(faceId);
 let a=fineParents[rows.x];if(rows.y==INVALID){return vec2u(a,INVALID);}
 let b=fineParents[rows.y];return vec2u(min(a,b),max(a,b));
}
fn edgeHash(pair:vec2u)->u32{
 return ((pair.x+1u)*0x9e3779b1u)^((pair.y+1u)*0x85ebca6bu);
}
fn edgeExactAdd(limbs:ptr<function,array<i32,36>>,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;
 if(magnitude==0u){return;}let exponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,exponent!=0u);let shift=select(3u,exponent+2u,exponent!=0u);
 let first=shift>>3u;let shifted=significand<<(shift&7u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=first+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){(*limbs)[limb]+=byte;}}}
fn edgeFloorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn edgeExactValue(value:ptr<function,array<i32,36>>)->f32{
 for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=edgeFloorDiv256((*value)[limb]);
  (*value)[limb]=normalized.y;(*value)[limb+1u]+=normalized.x;}
 var result=0.;for(var limb=0u;limb<36u;limb+=1u){result+=ldexp(f32((*value)[limb]),-152+i32(limb*8u));}
 return result;
}
fn edgeSlotsBase()->u32{return 2u*p.fused.z;}
fn edgeCursorsBase()->u32{return edgeSlotsBase()+p.fused.w;}
fn coarseEdgeCapacity()->u32{return arrayLength(&coarseFaces)/4u;}
fn edgeErrorWord()->u32{return edgeCursorsBase()+max(p.capacities.y,coarseEdgeCapacity());}
fn storeCompiledEdge(edgeId:u32,pair:vec2u,coefficient:f32){
 storeEdge(edgeId,AlgebraicEdge(pair.x,pair.y,coefficient,0u));
}

// Compile one canonical algebraic edge for every unique coarse row pair. The
// minimum fine source id owns the edge, which makes edge ids independent of
// lane scheduling. A compact edge->source CSR plus exact accumulation makes
// every coefficient refresh an order-independent segmented reduction, never an
// atomic floating scatter.
@compute @workgroup_size(256)
fn buildLosassoCoarseFaces(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 let enabled=coarseControl[3]==1u;
 let fineCount=select(0u,min(fineControl[2],p.fused.w),enabled);
 let directoryCapacity=p.fused.z;let mask=directoryCapacity-1u;
 for(var word=lane;word<2u*directoryCapacity;word+=256u){
  atomicStore(&edgeScratch[word],select(0u,INVALID,(word&1u)==1u));
 }
 for(var faceId=lane;faceId<p.fused.w;faceId+=256u){
  atomicStore(&edgeScratch[edgeSlotsBase()+faceId],INVALID);
 }
 for(var faceId=lane;faceId<coarseEdgeCapacity();faceId+=256u){
  atomicStore(&edgeScratch[edgeCursorsBase()+faceId],0u);
  atomicStore(&coarseFaceSourceOffsets[faceId],0u);
 }
 if(lane==0u){atomicStore(&coarseFaceSourceOffsets[coarseEdgeCapacity()],0u);}
 if(lane==0u){atomicStore(&edgeScratch[edgeErrorWord()],0u);}
 storageBarrier();workgroupBarrier();
 for(var faceId=lane;faceId<fineCount;faceId+=256u){let pair=canonicalParents(faceId);
  if(pair.x==pair.y){continue;}let hash=edgeHash(pair);var installed=INVALID;
  for(var probe=0u;probe<directoryCapacity;probe+=1u){let slot=(hash+probe)&mask;
   let at=2u*slot;var claim=atomicCompareExchangeWeak(&edgeScratch[at],0u,faceId+1u);
   for(var retry=0u;retry<4u&&!claim.exchanged&&claim.old_value==0u;retry+=1u){
    claim=atomicCompareExchangeWeak(&edgeScratch[at],0u,faceId+1u);}
   if(claim.exchanged){installed=slot;break;}
   if(claim.old_value>0u&&all(canonicalParents(claim.old_value-1u)==pair)){
    atomicMin(&edgeScratch[at],faceId+1u);installed=slot;break;
   }
  }
  atomicStore(&edgeScratch[edgeSlotsBase()+faceId],installed);
  if(installed==INVALID){atomicOr(&edgeScratch[edgeErrorWord()],ERROR_CAPACITY);}
 }
 storageBarrier();workgroupBarrier();
 let chunk=(fineCount+255u)/256u;let begin=min(lane*chunk,fineCount);
 let end=min(begin+chunk,fineCount);var laneCount=0u;
 for(var faceId=begin;faceId<end;faceId+=1u){let slot=atomicLoad(&edgeScratch[edgeSlotsBase()+faceId]);
  if(slot!=INVALID&&atomicLoad(&edgeScratch[2u*slot])==faceId+1u){laneCount+=1u;}}
 hierarchyPrefixTotals[lane]=laneCount;workgroupBarrier();
 for(var offset=1u;offset<256u;offset*=2u){var addend=0u;
  if(lane>=offset){addend=hierarchyPrefixTotals[lane-offset];}
  workgroupBarrier();hierarchyPrefixTotals[lane]+=addend;workgroupBarrier();}
 var edgeId=hierarchyPrefixTotals[lane]-laneCount;
 for(var faceId=begin;faceId<end;faceId+=1u){let slot=atomicLoad(&edgeScratch[edgeSlotsBase()+faceId]);
  if(slot!=INVALID&&atomicLoad(&edgeScratch[2u*slot])==faceId+1u){
   atomicStore(&edgeScratch[2u*slot+1u],edgeId);edgeId+=1u;}}
 let edgeCount=hierarchyPrefixTotals[255u];
 if(edgeCount>coarseEdgeCapacity()){atomicOr(&edgeScratch[edgeErrorWord()],ERROR_CAPACITY);}
 storageBarrier();workgroupBarrier();
 for(var faceId=lane;faceId<fineCount;faceId+=256u){let slot=atomicLoad(&edgeScratch[edgeSlotsBase()+faceId]);
  if(slot==INVALID){continue;}let id=atomicLoad(&edgeScratch[2u*slot+1u]);
  atomicStore(&edgeScratch[edgeSlotsBase()+faceId],id);
  atomicAdd(&coarseFaceSourceOffsets[id+1u],1u);
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u){var total=0u;for(var id=0u;id<edgeCount;id+=1u){
   let count=atomicLoad(&coarseFaceSourceOffsets[id+1u]);
   atomicStore(&coarseFaceSourceOffsets[id],total);
   atomicStore(&edgeScratch[edgeCursorsBase()+id],total);total+=count;
  }atomicStore(&coarseFaceSourceOffsets[edgeCount],total);
 }
 storageBarrier();workgroupBarrier();
 for(var faceId=lane;faceId<fineCount;faceId+=256u){let id=atomicLoad(&edgeScratch[edgeSlotsBase()+faceId]);
  if(id==INVALID){continue;}let at=atomicAdd(&edgeScratch[edgeCursorsBase()+id],1u);
  coarseFaceSources[at]=faceId;
 }
 storageBarrier();workgroupBarrier();
 for(var id=lane;id<edgeCount;id+=256u){let first=atomicLoad(&coarseFaceSourceOffsets[id]);
  let last=atomicLoad(&coarseFaceSourceOffsets[id+1u]);
  var exact:array<i32,36>;for(var at=first;at<last;at+=1u){edgeExactAdd(&exact,fineCoefficient(coarseFaceSources[at]));}
  let coefficient=edgeExactValue(&exact);
  let pair=canonicalParents(coarseFaceSources[first]);storeCompiledEdge(id,pair,coefficient);
 }
 storageBarrier();workgroupBarrier();
 if(lane==0u){coarseControl[4]|=atomicLoad(&edgeScratch[edgeErrorWord()]);
  coarseControl[2]=select(0u,edgeCount,coarseControl[4]==0u);
  if(coarseControl[4]!=0u){coarseControl[3]=0u;coarseDispatch[0]=0u;}}
}

fn buildAlgebraicCSR(lane:u32){
 if(fineControl[5]==1u){return;}var valid=coarseControl[3]==1u;
 let count=select(0u,coarseControl[2],valid);let rows=select(0u,coarseControl[1],valid);
 for(var row=lane;row<=rows;row+=256u){atomicStore(&coarseOffsets[row],0u);
  if(row<rows){atomicStore(&edgeScratch[edgeCursorsBase()+row],0u);}}workgroupBarrier();
 for(var id=lane;id<count;id+=256u){let edge=loadEdge(id);atomicAdd(&coarseOffsets[edge.rowA+1u],1u);
  if(edge.rowB!=INVALID){atomicAdd(&coarseOffsets[edge.rowB+1u],1u);}}
 workgroupBarrier();if(lane==0u){var total=0u;for(var row=0u;row<rows;row+=1u){let n=atomicLoad(&coarseOffsets[row+1u]);
   atomicStore(&coarseOffsets[row],total);fusedArena[fusedOffsets()+row]=total;
   atomicStore(&edgeScratch[edgeCursorsBase()+row],total);total+=n;}
  atomicStore(&coarseOffsets[rows],total);fusedArena[fusedOffsets()+rows]=total;
  if(total>p.capacities.z){coarseControl[4]|=ERROR_CAPACITY;}
  coarseDispatch[3]=select(0u,(count+63u)/64u,coarseControl[4]==0u);}
 workgroupBarrier();valid=coarseControl[4]==0u;
 if(valid){for(var id=lane;id<count;id+=256u){let edge=loadEdge(id);
   let at=atomicAdd(&edgeScratch[edgeCursorsBase()+edge.rowA],1u);coarseRowFaces[at]=id;
   if(edge.rowB!=INVALID){let atB=atomicAdd(&edgeScratch[edgeCursorsBase()+edge.rowB],1u);coarseRowFaces[atB]=id;}}}
 workgroupBarrier();if(valid){for(var row=lane;row<rows;row+=256u){let first=atomicLoad(&coarseOffsets[row]);let last=atomicLoad(&coarseOffsets[row+1u]);
   for(var at=first+1u;at<last;at+=1u){let value=coarseRowFaces[at];var cursor=at;
    while(cursor>first&&coarseRowFaces[cursor-1u]>value){coarseRowFaces[cursor]=coarseRowFaces[cursor-1u];cursor-=1u;}
    coarseRowFaces[cursor]=value;}
  }}
 workgroupBarrier();if(lane==0u){if(!valid){coarseControl[3]=0u;coarseDispatch[0]=0u;}
  for(var word=0u;word<8u;word+=1u){fusedArena[fusedControl()+word]=coarseControl[word];}}
}

@compute @workgroup_size(256)
fn buildLosassoCoarseCSRSmall(@builtin(local_invocation_index)lane:u32){buildAlgebraicCSR(lane);}

// The unique edge graph has O(rows) records, so the wide publication path can
// use the same atomic-count/scatter construction as the resident tier. This
// deletes the former rows-by-faces search used to recover CSR.
@compute @workgroup_size(256)
fn buildLosassoCoarseCSR(@builtin(local_invocation_index)lane:u32){
 buildAlgebraicCSR(lane);
}

fn refreshEdge(id:u32,reusedOnly:bool){
 if(reusedOnly&&fineControl[5]!=1u){return;}
 if(id==0u&&reusedOnly){coarseControl[0]=fineControl[0];coarseControl[5]=1u;
  fusedArena[fusedControl()]=fineControl[0];fusedArena[fusedControl()+5u]=1u;}
 if(coarseControl[3]!=1u||id>=coarseControl[2]){return;}
 let first=atomicLoad(&coarseFaceSourceOffsets[id]);let last=atomicLoad(&coarseFaceSourceOffsets[id+1u]);
 if(first>last||last>fineControl[2]){coarseControl[4]|=ERROR_CAPACITY;return;}
 var exact:array<i32,36>;for(var at=first;at<last;at+=1u){edgeExactAdd(&exact,fineCoefficient(coarseFaceSources[at]));}
 let coefficient=edgeExactValue(&exact);
 var edge=loadEdge(id);edge.coefficient=coefficient;storeEdge(id,edge);
}
@compute @workgroup_size(64)
fn refreshLosassoCoarseFaces(@builtin(global_invocation_id)g:vec3u){refreshEdge(g.x,false);}
@compute @workgroup_size(64)
fn refreshLosassoReusedCoarseFaces(@builtin(global_invocation_id)g:vec3u){refreshEdge(g.x,true);}

// Materialize the immutable pressure view once per authority/coefficient
// publication. Every recurring solve then walks one sequential 8-byte record
// per incidence; no face-id or unique-edge gather remains in the hot path.
@compute @workgroup_size(64)
fn publishLosassoFinestDirectedCSR(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;let rows=min(fineControl[1],p.capacities.y);
 if(row==0u){for(var word=0u;word<8u;word+=1u){fusedArena[fusedControl()+word]=fineControl[word];}}
 if(row>rows||row>=arrayLength(&fineOffsets)){return;}
 let begin=fineOffsets[row];fusedArena[fusedOffsets()+row]=begin;
 if(row==rows){return;}let end=fineOffsets[row+1u];
 if(begin>end||end>arrayLength(&fineRowFaces)||end>p.capacities.z){return;}
 for(var at=begin;at<end;at+=1u){let id=fineRowFaces[at];let pair=fineRows(id);
  let neighbour=select(pair.x,pair.y,pair.x==row);
  storeDirectedEdge(at,neighbour,fineCoefficient(id));}
}

@compute @workgroup_size(64)
fn publishLosassoCoarseDirectedCSR(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;let rows=min(coarseControl[1],p.capacities.y);
 if(row==0u){for(var word=0u;word<8u;word+=1u){fusedArena[fusedControl()+word]=coarseControl[word];}}
 if(row>rows||row>=arrayLength(&coarseOffsets)){return;}
 let begin=atomicLoad(&coarseOffsets[row]);fusedArena[fusedOffsets()+row]=begin;
 if(row==rows){return;}let end=atomicLoad(&coarseOffsets[row+1u]);
 if(begin>end||end>arrayLength(&coarseRowFaces)||end>p.capacities.z){return;}
 for(var at=begin;at<end;at+=1u){let edge=loadEdge(coarseRowFaces[at]);
  let neighbour=select(edge.rowA,edge.rowB,edge.rowA==row);
  storeDirectedEdge(at,neighbour,edge.coefficient);}
}
`;
