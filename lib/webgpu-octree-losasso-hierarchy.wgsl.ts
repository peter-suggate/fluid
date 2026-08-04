/** GPU construction of the reduced geometric Losasso multigrid hierarchy. */
export const octreeLosassoHierarchyWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
const ERROR_CAPACITY:u32=1u;
const ERROR_GEOMETRY:u32=2u;

struct Params {
  dimensionsTarget:vec4u,
  capacities:vec4u,
  cellSize:vec4f,
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
@group(0)@binding(12)var<storage,read_write> fineVolumes:array<f32>;
@group(0)@binding(13)var<storage,read_write> coarseInverseVolumes:array<f32>;
@group(0)@binding(14)var<storage,read_write> directory:array<vec2u>;
@group(0)@binding(15)var<storage,read_write> scratch:array<u32>;

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

@compute @workgroup_size(64)
fn extractLosassoFinestCells(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(row>=min(fineControl[1],p.capacities.x)){return;}
 let header=leafHeaders[row];let dimensions=p.dimensionsTarget.xyz;
 let origin=vec3u(header.cell%dimensions.x,
  (header.cell/dimensions.x)%dimensions.y,
  header.cell/(dimensions.x*dimensions.y));
 fineCells[row]=vec4u(origin,header.size);
}

// Candidate rows are compact but mixed-size, so their geometric parents are
// not necessarily adjacent. A deterministic singleton inserts them into an
// open-addressed directory and gives every fine row exactly one parent.
@compute @workgroup_size(1)
fn buildLosassoParentRows(){
 for(var word=0u;word<8u;word+=1u){coarseControl[word]=0u;}
 let fineCount=min(fineControl[1],p.capacities.x);
 if(!controlValid()){coarseControl[4]=ERROR_GEOMETRY;return;}
 for(var slot=0u;slot<p.capacities.w;slot+=1u){directory[slot]=vec2u(0u);}
 for(var row=0u;row<p.capacities.y;row+=1u){scratch[row]=0u;}
 var coarseCount=0u;let mask=p.capacities.w-1u;
 for(var row=0u;row<fineCount;row+=1u){
  let fine=fineCells[row];let parentSize=max(fine.w,p.dimensionsTarget.w);
  let parent=vec4u((fine.xyz/parentSize)*parentSize,parentSize);
  let volume=cellVolume(fine.w);
  if(volume==0u||any(parent.xyz+vec3u(parentSize)>p.dimensionsTarget.xyz)){
   coarseControl[4]|=ERROR_GEOMETRY;continue;
  }
  let hash=hashCell(parent);var parentRow=INVALID;
  for(var probe=0u;probe<p.capacities.w;probe+=1u){
   let slot=(hash+probe)&mask;let installed=directory[slot].x;
   if(installed==0u){
    if(coarseCount>=p.capacities.y){coarseControl[4]|=ERROR_CAPACITY;break;}
    parentRow=coarseCount;coarseCount+=1u;coarseCells[parentRow]=parent;
    directory[slot]=vec2u(parentRow+1u,hash);break;
   }
   let candidate=installed-1u;
   if(directory[slot].y==hash&&all(coarseCells[candidate]==parent)){
    parentRow=candidate;break;
   }
  }
  if(parentRow==INVALID){coarseControl[4]|=ERROR_CAPACITY;continue;}
  fineParents[row]=parentRow;fineVolumes[row]=f32(volume);
  let previous=scratch[parentRow];
  if(0xffffffffu-previous<volume){coarseControl[4]|=ERROR_CAPACITY;}
  else{scratch[parentRow]=previous+volume;}
 }
 coarseControl[0]=fineControl[0];coarseControl[1]=coarseCount;
 coarseControl[3]=select(0u,1u,coarseControl[4]==0u&&coarseCount!=0u);
}

@compute @workgroup_size(64)
fn finishLosassoParentRows(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;let coarseCount=coarseControl[1];
 if(row<coarseCount){
  coarseInverseVolumes[row]=select(0.,1./f32(scratch[row]),scratch[row]!=0u);
 }
 if(row==0u){coarseDispatch[0]=select(0u,(coarseCount+63u)/64u,coarseControl[3]==1u);
  coarseDispatch[1]=1u;coarseDispatch[2]=1u;}
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
@compute @workgroup_size(1)
fn buildLosassoCoarseFaces(){
 if(coarseControl[3]!=1u){return;}
 let fineFaceCount=min(fineControl[2],p.capacities.z);var count=0u;
 for(var faceId=0u;faceId<fineFaceCount;faceId+=1u){
  let face=fineFaces[faceId];let negative=fineParents[face.negativeRow];
  var positive=INVALID;if(face.positiveRow!=INVALID){positive=fineParents[face.positiveRow];}
  if(negative==positive){continue;}
  if(count>=p.capacities.z){coarseControl[4]|=ERROR_CAPACITY;break;}
  let negativeCell=coarseCells[negative];var distance=0.;
  if(positive!=INVALID){
   distance=abs(centre(coarseCells[positive],face.axis)-centre(negativeCell,face.axis))
    *p.cellSize[face.axis];
  }else{
   let positiveCell=(face.reserved&0x80000000u)!=0u;
   distance=abs(boundaryPlane(fineCells[face.negativeRow],face.axis,positiveCell)
    -centre(negativeCell,face.axis))*p.cellSize[face.axis];
  }
  if(!(distance>0.)){coarseControl[4]|=ERROR_GEOMETRY;continue;}
  var coarseFace=face;coarseFace.negativeRow=negative;coarseFace.positiveRow=positive;
  coarseFace.inverseDistance=1./distance;coarseFaces[count]=coarseFace;count+=1u;
 }
 coarseControl[2]=count;
}

@compute @workgroup_size(1)
fn buildLosassoCoarseCSR(){
 if(coarseControl[3]!=1u){return;}let count=coarseControl[2];
 let rows=coarseControl[1];for(var row=0u;row<=rows;row+=1u){coarseOffsets[row]=0u;}
 for(var faceId=0u;faceId<count;faceId+=1u){let face=coarseFaces[faceId];
  coarseOffsets[face.negativeRow+1u]+=1u;
  if(face.positiveRow!=INVALID){coarseOffsets[face.positiveRow+1u]+=1u;}
 }
 var total=0u;for(var row=0u;row<rows;row+=1u){
  let n=coarseOffsets[row+1u];coarseOffsets[row]=total;total+=n;scratch[row]=0u;
 }coarseOffsets[rows]=total;
 if(total>2u*p.capacities.z){coarseControl[4]|=ERROR_CAPACITY;}
 else{for(var faceId=0u;faceId<count;faceId+=1u){let face=coarseFaces[faceId];
   var slot=coarseOffsets[face.negativeRow]+scratch[face.negativeRow];
   coarseRowFaces[slot]=faceId;scratch[face.negativeRow]+=1u;
   if(face.positiveRow!=INVALID){slot=coarseOffsets[face.positiveRow]+scratch[face.positiveRow];
    coarseRowFaces[slot]=faceId;scratch[face.positiveRow]+=1u;}
  }}
 if(coarseControl[4]!=0u){coarseControl[3]=0u;coarseDispatch[0]=0u;}
}
`;
