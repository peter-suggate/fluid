/** GPU construction of the reduced geometric Losasso multigrid hierarchy. */
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

@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> leafHeaders:array<LeafHeader>;
@group(0)@binding(2)var<storage,read> fineControl:array<u32>;
@group(0)@binding(3)var<storage,read_write> fineCells:array<vec4u>;
@group(0)@binding(4)var<storage,read> fineFaces:array<Face>;
@group(0)@binding(5)var<storage,read_write> coarseControl:array<u32>;
@group(0)@binding(6)var<storage,read_write> coarseCells:array<vec4u>;
@group(0)@binding(7)var<storage,read_write> coarseFaces:array<Face>;
@group(0)@binding(8)var<storage,read_write> coarseOffsets:array<u32>;
@group(0)@binding(9)var<storage,read_write> coarseRowFaces:array<u32>;
@group(0)@binding(10)var<storage,read_write> coarseDispatch:array<u32>;
@group(0)@binding(11)var<storage,read_write> fineParents:array<u32>;
@group(0)@binding(12)var<storage,read_write> childOffsets:array<u32>;
@group(0)@binding(13)var<storage,read_write> childList:array<u32>;
@group(0)@binding(14)var<storage,read_write> directory:array<atomic<u32>>;
@group(0)@binding(16)var<storage,read_write> coarseFaceSources:array<u32>;
@group(0)@binding(17)var<storage,read_write> fusedArena:array<u32>;

var<workgroup> hierarchyPrefixTotals:array<u32,256>;
// The resident-solve tier never exceeds 4K rows.  Its hierarchy CSR can count
// face incidences in shared memory, scatter them once, and restore the exact
// ascending face-id order locally instead of scanning every face for every
// row twice.  Larger hierarchies retain the wide fallback below.
var<workgroup> hierarchyRowCursors:array<atomic<u32>,4096>;

fn fusedControl()->u32{return p.fused.x;}
fn fusedOffsets()->u32{return fusedControl()+8u;}
fn fusedRowFaces()->u32{return fusedOffsets()+p.capacities.y+1u;}
fn fusedFaces()->u32{return fusedRowFaces()+2u*p.capacities.z;}
fn fusedParents()->u32{return fusedFaces()+8u*p.capacities.z;}
fn fusedChildOffsets()->u32{return fusedParents()+p.capacities.y;}
fn fusedChildList()->u32{return fusedChildOffsets()+p.capacities.y+1u;}
fn storeFusedFace(faceId:u32,face:Face){let base=fusedFaces()+8u*faceId;
 fusedArena[base]=face.negativeRow;fusedArena[base+1u]=face.positiveRow;
 fusedArena[base+2u]=face.axis;fusedArena[base+3u]=face.reserved;
 fusedArena[base+4u]=bitcast<u32>(face.area);
 fusedArena[base+5u]=bitcast<u32>(face.inverseDistance);
 fusedArena[base+6u]=bitcast<u32>(face.openFraction);
 fusedArena[base+7u]=bitcast<u32>(face.normalVelocity);
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

fn centre(cell:vec4u,axis:u32)->f32{
 return f32(cell[axis])+.5*f32(cell.w);
}
fn boundaryPlane(cell:vec4u,axis:u32,positiveCell:bool)->f32{
 return f32(cell[axis])+select(f32(cell.w),0.,positiveCell);
}

// Retaining the fine face patches avoids a topology-dependent floating area
// fold. Only patches internal to one aggregate disappear; every remaining
// patch receives the closed-form parent-centre distance.
@compute @workgroup_size(256)
fn buildLosassoCoarseFaces(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 let enabled=coarseControl[3]==1u;
 let fineFaceCount=select(0u,min(fineControl[2],p.capacities.z),enabled);
 let chunk=(fineFaceCount+255u)/256u;let begin=min(lane*chunk,fineFaceCount);
 let end=min(begin+chunk,fineFaceCount);var laneCount=0u;
 for(var faceId=begin;faceId<end;faceId+=1u){let face=fineFaces[faceId];
  let negative=fineParents[face.negativeRow];var positive=INVALID;
  if(face.positiveRow!=INVALID){positive=fineParents[face.positiveRow];}
  if(negative!=positive){laneCount+=1u;}
 }
 hierarchyPrefixTotals[lane]=laneCount;workgroupBarrier();
 for(var offset=1u;offset<256u;offset*=2u){var addend=0u;
  if(lane>=offset){addend=hierarchyPrefixTotals[lane-offset];}
  workgroupBarrier();hierarchyPrefixTotals[lane]+=addend;workgroupBarrier();
 }
 var destination=hierarchyPrefixTotals[lane]-laneCount;
 for(var faceId=begin;faceId<end;faceId+=1u){
  let face=fineFaces[faceId];let negative=fineParents[face.negativeRow];
  var positive=INVALID;if(face.positiveRow!=INVALID){positive=fineParents[face.positiveRow];}
  if(negative==positive){continue;}
  let negativeCell=coarseCells[negative];var distance=0.;
  if(positive!=INVALID){distance=abs(centre(coarseCells[positive],face.axis)
    -centre(negativeCell,face.axis))*p.cellSize[face.axis];}
  else{let positiveCell=(face.reserved&0x80000000u)!=0u;
   distance=abs(boundaryPlane(fineCells[face.negativeRow],face.axis,positiveCell)
    -centre(negativeCell,face.axis))*p.cellSize[face.axis];}
  if(!(distance>0.)){coarseControl[4]|=ERROR_GEOMETRY;continue;}
  var coarseFace=face;coarseFace.negativeRow=negative;coarseFace.positiveRow=positive;
  coarseFace.inverseDistance=1./distance;coarseFaces[destination]=coarseFace;
  storeFusedFace(destination,coarseFace);
  coarseFaceSources[destination]=faceId;destination+=1u;
 }
 if(lane==255u){coarseControl[2]=select(0u,hierarchyPrefixTotals[lane],enabled);}
}

@compute @workgroup_size(256)
fn buildLosassoCoarseCSRSmall(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 let enabled=coarseControl[3]==1u;let count=select(0u,coarseControl[2],enabled);
 let rows=select(0u,coarseControl[1],enabled);
  for(var row=lane;row<rows;row+=256u){atomicStore(&hierarchyRowCursors[row],0u);}
  workgroupBarrier();
  for(var faceId=lane;faceId<count;faceId+=256u){let face=coarseFaces[faceId];
   atomicAdd(&hierarchyRowCursors[face.negativeRow],1u);
   if(face.positiveRow!=INVALID){atomicAdd(&hierarchyRowCursors[face.positiveRow],1u);}
  }
  workgroupBarrier();
  if(lane==0u){var total=0u;
   for(var row=0u;row<rows;row+=1u){let incidences=atomicLoad(&hierarchyRowCursors[row]);
    coarseOffsets[row]=total;fusedArena[fusedOffsets()+row]=total;
    atomicStore(&hierarchyRowCursors[row],total);total+=incidences;
   }
   coarseOffsets[rows]=total;fusedArena[fusedOffsets()+rows]=total;
   if(total>2u*p.capacities.z){coarseControl[4]|=ERROR_CAPACITY;}
   coarseDispatch[3]=select(0u,(count+63u)/64u,coarseControl[4]==0u);
  }
  workgroupBarrier();
  let valid=coarseControl[4]==0u;
  if(valid){
   for(var faceId=lane;faceId<count;faceId+=256u){let face=coarseFaces[faceId];
    let negativeAt=atomicAdd(&hierarchyRowCursors[face.negativeRow],1u);
    coarseRowFaces[negativeAt]=faceId;
    if(face.positiveRow!=INVALID){let positiveAt=atomicAdd(&hierarchyRowCursors[face.positiveRow],1u);
     coarseRowFaces[positiveAt]=faceId;}
   }
  }
  workgroupBarrier();
  if(valid){for(var row=lane;row<rows;row+=256u){let begin=coarseOffsets[row];let end=coarseOffsets[row+1u];
    for(var at=begin+1u;at<end;at+=1u){let value=coarseRowFaces[at];var cursor=at;
     while(cursor>begin&&coarseRowFaces[cursor-1u]>value){coarseRowFaces[cursor]=coarseRowFaces[cursor-1u];cursor-=1u;}
     coarseRowFaces[cursor]=value;
    }
    for(var at=begin;at<end;at+=1u){fusedArena[fusedRowFaces()+at]=coarseRowFaces[at];}
   }
  }
  workgroupBarrier();
  if(lane==0u){if(!valid){coarseControl[3]=0u;coarseDispatch[0]=0u;}
   for(var word=0u;word<8u;word+=1u){fusedArena[fusedControl()+word]=coarseControl[word];}
  }
}

@compute @workgroup_size(256)
fn buildLosassoCoarseCSR(@builtin(local_invocation_index)lane:u32){
 if(fineControl[5]==1u){return;}
 let enabled=coarseControl[3]==1u;let count=select(0u,coarseControl[2],enabled);
 let rows=select(0u,coarseControl[1],enabled);
 let chunk=(rows+255u)/256u;let begin=min(lane*chunk,rows);let end=min(begin+chunk,rows);
 var laneTotal=0u;for(var row=begin;row<end;row+=1u){var incidences=0u;
  for(var faceId=0u;faceId<count;faceId+=1u){let face=coarseFaces[faceId];
   incidences+=select(0u,1u,face.negativeRow==row);
   incidences+=select(0u,1u,face.positiveRow==row);
  }coarseOffsets[row+1u]=incidences;laneTotal+=incidences;
 }
 hierarchyPrefixTotals[lane]=laneTotal;workgroupBarrier();
 for(var offset=1u;offset<256u;offset*=2u){var addend=0u;
  if(lane>=offset){addend=hierarchyPrefixTotals[lane-offset];}
  workgroupBarrier();hierarchyPrefixTotals[lane]+=addend;workgroupBarrier();
 }
 var base=hierarchyPrefixTotals[lane]-laneTotal;
 for(var row=begin;row<end;row+=1u){let incidences=coarseOffsets[row+1u];
  coarseOffsets[row]=base;fusedArena[fusedOffsets()+row]=base;base+=incidences;
 }
 if(lane==255u){let total=hierarchyPrefixTotals[lane];coarseOffsets[rows]=total;
  fusedArena[fusedOffsets()+rows]=total;
  if(total>2u*p.capacities.z){coarseControl[4]|=ERROR_CAPACITY;}
  coarseDispatch[3]=select(0u,(count+63u)/64u,coarseControl[4]==0u);
 }
 storageBarrier();workgroupBarrier();
 let valid=coarseControl[4]==0u;
 if(!valid){if(lane==0u){coarseControl[3]=0u;coarseDispatch[0]=0u;}}
 if(valid){for(var row=begin;row<end;row+=1u){var cursor=coarseOffsets[row];
  for(var faceId=0u;faceId<count;faceId+=1u){let face=coarseFaces[faceId];
   if(face.negativeRow==row||face.positiveRow==row){coarseRowFaces[cursor]=faceId;
    fusedArena[fusedRowFaces()+cursor]=faceId;cursor+=1u;}
  }
 }}
 storageBarrier();workgroupBarrier();
 if(lane==0u){for(var word=0u;word<8u;word+=1u){
  fusedArena[fusedControl()+word]=coarseControl[word];
 }}
}

fn refreshCoarseFace(faceId:u32,reusedOnly:bool){
 if(reusedOnly&&fineControl[5]!=1u){return;}
 if(faceId==0u&&reusedOnly){coarseControl[0]=fineControl[0];coarseControl[5]=1u;
  fusedArena[fusedControl()]=fineControl[0];fusedArena[fusedControl()+5u]=1u;}
 if(coarseControl[3]!=1u||faceId>=coarseControl[2]){return;}
 let sourceId=coarseFaceSources[faceId];
 if(sourceId>=fineControl[2]){coarseControl[4]|=ERROR_CAPACITY;return;}
 let source=fineFaces[sourceId];var targetFace=coarseFaces[faceId];
 targetFace.openFraction=source.openFraction;
 targetFace.normalVelocity=source.normalVelocity;
 let negativeCell=coarseCells[targetFace.negativeRow];var distance=0.;
 if(targetFace.positiveRow!=INVALID){
  distance=abs(centre(coarseCells[targetFace.positiveRow],targetFace.axis)
   -centre(negativeCell,targetFace.axis))*p.cellSize[targetFace.axis];
 }else{
  let positiveCell=(source.reserved&0x80000000u)!=0u;
  distance=abs(boundaryPlane(fineCells[source.negativeRow],source.axis,positiveCell)
   -centre(negativeCell,targetFace.axis))*p.cellSize[targetFace.axis];
 }
 if(!(distance>0.)){coarseControl[4]|=ERROR_GEOMETRY;return;}
 targetFace.inverseDistance=1./distance;
 coarseFaces[faceId]=targetFace;
 storeFusedFace(faceId,targetFace);
}
@compute @workgroup_size(64)
fn refreshLosassoCoarseFaces(@builtin(global_invocation_id)g:vec3u){refreshCoarseFace(g.x,false);}
@compute @workgroup_size(64)
fn refreshLosassoReusedCoarseFaces(@builtin(global_invocation_id)g:vec3u){refreshCoarseFace(g.x,true);}
`;
