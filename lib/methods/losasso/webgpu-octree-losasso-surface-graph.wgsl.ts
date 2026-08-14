/** Clean-room adaptive Losasso surface-graph construction and stable commit. */
export const octreeLosassoSurfaceGraphWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
const PUBLISHED:u32=0x80000000u;
const RADIX_VALID:u32=0x52535254u;
const WORKGROUP:u32=64u;
const LOCATOR_HEADER:u32=16u;

const ERROR_CAPACITY:u32=1u;
const ERROR_TOPOLOGY:u32=2u;
const ERROR_GEOMETRY:u32=4u;
const ERROR_ORDER:u32=8u;
const ERROR_LOOKUP:u32=16u;
const ERROR_CONSTRAINT:u32=32u;
const ERROR_ADJACENCY:u32=64u;
const ERROR_GENERATION:u32=128u;
const ERROR_COVERAGE:u32=256u;

const NODE_INDEPENDENT:u32=1u;
const NODE_EDGE_HANGING:u32=2u;
const NODE_FACE_HANGING:u32=4u;
const NODE_BOUNDARY:u32=8u;

struct Params{
 dimensionsMaximum:vec4u,
 capacities:vec4u,
 cellSize:vec4f,
 topologyLayout:vec4u,
 readyLayout:vec4u,
 expected:vec4u,
 ownerLayout:vec4u,
 radixLayout:vec4u,
}
struct LeafHeader{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f}
struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}
struct NodeRecord{item:u32,flags:u32,localScale:u32,constraintOffset:u32}
struct DirectoryRecord{item:u32,slot:u32}
struct LeafDirectoryRecord{cell:u32,size:u32,slot:u32,flags:u32}
struct ConstraintRecord{header:vec4u,masters:vec4u,numerators:vec4u}
struct AdjacencyRecord{negative:vec3u,positive:vec3u,negativeSpan:vec3f,positiveSpan:vec3f}

@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>topologyControl:array<u32>;
@group(0)@binding(2)var<storage,read>readyControl:array<u32>;
@group(0)@binding(3)var<storage,read>headers:array<LeafHeader>;
@group(0)@binding(4)var<storage,read_write>radixHeader:array<u32>;
@group(0)@binding(5)var<storage,read_write>radixKeys:array<u32>;
@group(0)@binding(6)var<storage,read>radixRuns:array<vec2u>;
@group(0)@binding(7)var<storage,read>radixControl:array<u32>;
@group(0)@binding(8)var<storage,read_write>candidateControl:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read_write>candidateLeaves:array<LeafRecord>;
@group(0)@binding(10)var<storage,read_write>candidateLeafDirectory:array<LeafDirectoryRecord>;
@group(0)@binding(11)var<storage,read_write>candidateNodes:array<NodeRecord>;
@group(0)@binding(12)var<storage,read_write>candidateNodeDirectory:array<DirectoryRecord>;
@group(0)@binding(13)var<storage,read_write>candidateConstraints:array<ConstraintRecord>;
@group(0)@binding(14)var<storage,read_write>candidateAdjacency:array<u32>;
@group(0)@binding(15)var<storage,read_write>candidatePhi:array<vec2f>;
@group(0)@binding(16)var<storage,read_write>candidateVelocity:array<vec4f>;
@group(0)@binding(17)var<storage,read_write>candidateValidity:array<u32>;
@group(0)@binding(18)var<storage,read_write>acceptedControl:array<atomic<u32>>;
@group(0)@binding(19)var<storage,read_write>acceptedLeaves:array<LeafRecord>;
@group(0)@binding(20)var<storage,read_write>acceptedLeafDirectory:array<LeafDirectoryRecord>;
@group(0)@binding(21)var<storage,read_write>acceptedNodes:array<NodeRecord>;
@group(0)@binding(22)var<storage,read_write>acceptedNodeDirectory:array<DirectoryRecord>;
@group(0)@binding(23)var<storage,read_write>acceptedConstraints:array<ConstraintRecord>;
@group(0)@binding(24)var<storage,read_write>acceptedAdjacency:array<u32>;
@group(0)@binding(25)var<storage,read_write>acceptedPhi:array<vec2f>;
@group(0)@binding(26)var<storage,read_write>acceptedVelocity:array<vec4f>;
@group(0)@binding(27)var<storage,read_write>acceptedValidity:array<u32>;
@group(0)@binding(28)var<storage,read_write>commitDispatch:array<u32>;
@group(0)@binding(29)var<storage,read>ownerArena:array<u32>;
@group(0)@binding(30)var<storage,read>ownerTransaction:array<u32>;
@group(0)@binding(31)var<storage,read_write>candidatePressureRowToLeaf:array<u32>;
@group(0)@binding(32)var<storage,read_write>acceptedPressureRowToLeaf:array<u32>;
@group(0)@binding(33)var<storage,read_write>candidateIncidentLeaves:array<u32>;
@group(0)@binding(34)var<storage,read_write>acceptedIncidentLeaves:array<u32>;
@group(0)@binding(35)var<storage,read_write>candidateLeafLocator:array<atomic<u32>>;
@group(0)@binding(36)var<storage,read_write>acceptedLeafLocator:array<atomic<u32>>;
@group(0)@binding(37)var<storage,read_write>candidateSurfaceMass:array<f32>;
@group(0)@binding(38)var<storage,read_write>candidateSurfaceCompression:array<f32>;
@group(0)@binding(39)var<storage,read_write>acceptedSurfaceMass:array<f32>;
@group(0)@binding(40)var<storage,read_write>acceptedSurfaceCompression:array<f32>;

fn dims()->vec3u{return p.dimensionsMaximum.xyz;}
fn nodeDims()->vec3u{return dims()+vec3u(1u);}
fn decodeCell(cell:u32)->vec3u{return vec3u(cell%dims().x,(cell/dims().x)%dims().y,cell/(dims().x*dims().y));}
fn encodeCell(q:vec3u)->u32{return q.x+dims().x*(q.y+dims().y*q.z);}
fn decodeNode(item:u32)->vec3u{let d=nodeDims();return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}
fn encodeNode(q:vec3u)->u32{let d=nodeDims();return q.x+d.x*(q.y+d.y*q.z);}
fn isPowerOfTwo(v:u32)->bool{return v!=0u&&(v&(v-1u))==0u;}
fn identityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{return cellA<cellB||(cellA==cellB&&sizeA<sizeB);}
fn flattened(wg:vec3u,lane:u32)->u32{return lane+WORKGROUP*(wg.x+wg.y*p.capacities.w);}
fn groups(count:u32)->vec3u{let blocks=(count+WORKGROUP-1u)/WORKGROUP;let x=min(p.capacities.w,blocks);let safe=max(1u,x);return vec3u(x,select(1u,(blocks+safe-1u)/safe,x>0u),1u);}
fn topologyValid(epoch:u32,count:u32)->bool{return epoch!=0u&&count<=p.capacities.z&&p.topologyLayout.x<arrayLength(&topologyControl)&&p.topologyLayout.y<arrayLength(&topologyControl)&&p.topologyLayout.z<arrayLength(&topologyControl)&&p.topologyLayout.w<arrayLength(&topologyControl)&&topologyControl[p.topologyLayout.z]==p.expected.x&&topologyControl[p.topologyLayout.w]==0u;}
fn addError(bit:u32){atomicOr(&candidateControl[4],bit);}
fn setDispatch(base:u32,d:vec3u){atomicStore(&candidateControl[base],d.x);atomicStore(&candidateControl[base+1u],d.y);atomicStore(&candidateControl[base+2u],d.z);}
fn loadAdjacency(slot:u32)->AdjacencyRecord{let b=12u*slot;return AdjacencyRecord(
 vec3u(candidateAdjacency[b],candidateAdjacency[b+1u],candidateAdjacency[b+2u]),
 vec3u(candidateAdjacency[b+3u],candidateAdjacency[b+4u],candidateAdjacency[b+5u]),
 vec3f(bitcast<f32>(candidateAdjacency[b+6u]),bitcast<f32>(candidateAdjacency[b+7u]),bitcast<f32>(candidateAdjacency[b+8u])),
 vec3f(bitcast<f32>(candidateAdjacency[b+9u]),bitcast<f32>(candidateAdjacency[b+10u]),bitcast<f32>(candidateAdjacency[b+11u])));}
fn storeAdjacency(slot:u32,a:AdjacencyRecord){let b=12u*slot;
 candidateAdjacency[b]=a.negative.x;candidateAdjacency[b+1u]=a.negative.y;candidateAdjacency[b+2u]=a.negative.z;
 candidateAdjacency[b+3u]=a.positive.x;candidateAdjacency[b+4u]=a.positive.y;candidateAdjacency[b+5u]=a.positive.z;
 candidateAdjacency[b+6u]=bitcast<u32>(a.negativeSpan.x);candidateAdjacency[b+7u]=bitcast<u32>(a.negativeSpan.y);candidateAdjacency[b+8u]=bitcast<u32>(a.negativeSpan.z);
 candidateAdjacency[b+9u]=bitcast<u32>(a.positiveSpan.x);candidateAdjacency[b+10u]=bitcast<u32>(a.positiveSpan.y);candidateAdjacency[b+11u]=bitcast<u32>(a.positiveSpan.z);}

fn inactiveOwnerTable()->u32{return 1u-(ownerArena[10u]>>31u);}
fn ownerPayloadBase()->u32{return p.ownerLayout.w+inactiveOwnerTable()*p.ownerLayout.x*512u;}
fn ownerSortCapacity()->u32{return arrayLength(&ownerTransaction)-32u-7u*p.ownerLayout.x;}
fn ownerKeyBase()->u32{return 32u+ownerSortCapacity();}
fn ownerPageBase()->u32{return ownerKeyBase()+p.ownerLayout.x;}
fn brickDims()->vec3u{return (dims()+vec3u(7u))/8u;}
fn ownerPageRow(logical:u32)->u32{let key=logical+1u;var lo=0u;var hi=atomicLoad(&candidateControl[30]);while(lo<hi){let mid=lo+(hi-lo)/2u;if(ownerTransaction[ownerKeyBase()+mid]<key){lo=mid+1u;}else{hi=mid;}}if(lo<atomicLoad(&candidateControl[30])&&ownerTransaction[ownerKeyBase()+lo]==key){return lo;}return INVALID;}
fn candidateOwner(cell:vec3u)->vec4u{let bricks=brickDims();let brick=cell/8u;let logical=brick.x+bricks.x*(brick.y+bricks.y*brick.z);let row=ownerPageRow(logical);if(row==INVALID){return vec4u(INVALID);}let page=ownerTransaction[ownerPageBase()+row];if(page==0u||page>p.ownerLayout.x){return vec4u(INVALID);}let local=cell%8u;let at=local.x+8u*local.y+64u*local.z;let word=ownerArena[ownerPayloadBase()+(page-1u)*512u+at];if((word&0x80000000u)==0u){return vec4u(INVALID);}let exponent=(word>>18u)&7u;let span=1u<<exponent;let brickOrigin=vec3i(brick*8u);let originI=brickOrigin+vec3i(i32(word&63u)-32,i32((word>>6u)&63u)-32,i32((word>>12u)&63u)-32);if(exponent>10u||any(originI<vec3i(0))){return vec4u(INVALID);}let origin=vec3u(originI);if(span>p.dimensionsMaximum.w||any(origin+vec3u(span)>dims())||any(cell<origin)||any(cell>=origin+vec3u(span))){return vec4u(INVALID);}return vec4u(origin,span);}

@compute @workgroup_size(1)fn prepareSurfaceGraph(){
 let epoch=select(0u,topologyControl[p.topologyLayout.x],p.topologyLayout.x<arrayLength(&topologyControl));
 let pressureCount=select(p.capacities.z+1u,topologyControl[p.topologyLayout.y],p.topologyLayout.y<arrayLength(&topologyControl));let pageCount=select(p.ownerLayout.x+1u,ownerTransaction[27u],arrayLength(&ownerTransaction)>28u);let supportCount=pageCount*512u;
 let valid=topologyValid(epoch,pressureCount)&&pressureCount<=arrayLength(&headers)&&pressureCount<=arrayLength(&candidatePressureRowToLeaf)&&arrayLength(&ownerTransaction)>=33u+7u*p.ownerLayout.x&&ownerTransaction[24u]==epoch&&ownerTransaction[23u]==0u&&ownerTransaction[28u]==PUBLISHED&&pageCount<=p.ownerLayout.x&&supportCount<=p.radixLayout.x;
 for(var i=0u;i<36u;i+=1u){atomicStore(&candidateControl[i],0u);}
 atomicStore(&candidateControl[0],select(0u,epoch,valid));atomicStore(&candidateControl[28],select(0u,pressureCount,valid));atomicStore(&candidateControl[30],select(0u,pageCount,valid));atomicStore(&candidateControl[31],select(0u,supportCount,valid));
 atomicStore(&candidateControl[4],select(ERROR_TOPOLOGY,0u,valid));
 atomicStore(&candidateControl[14],p.capacities.x);atomicStore(&candidateControl[15],p.capacities.y);atomicStore(&candidateControl[16],p.capacities.y);setDispatch(17u,vec3u(0u,1u,1u));setDispatch(20u,vec3u(0u,1u,1u));setDispatch(32u,vec3u(0u,1u,1u));
 let supportDispatch=groups(select(0u,supportCount,valid));let pressureDispatch=groups(select(0u,pressureCount,valid));commitDispatch[0]=supportDispatch.x;commitDispatch[1]=supportDispatch.y;commitDispatch[2]=supportDispatch.z;commitDispatch[3]=0u;commitDispatch[4]=1u;commitDispatch[5]=1u;commitDispatch[8]=pressureDispatch.x;commitDispatch[9]=pressureDispatch.y;commitDispatch[10]=pressureDispatch.z;commitDispatch[12]=supportDispatch.x;commitDispatch[13]=supportDispatch.y;commitDispatch[14]=supportDispatch.z;
 atomicStore(&candidateLeafLocator[0],p.radixLayout.y);atomicStore(&candidateLeafLocator[1],select(0u,epoch,valid));atomicStore(&candidateLeafLocator[2],select(0u,pageCount,valid));atomicStore(&candidateLeafLocator[3],select(0u,supportCount,valid));atomicStore(&candidateLeafLocator[4],0u);atomicStore(&candidateLeafLocator[5],0u);atomicStore(&candidateLeafLocator[6],0u);atomicStore(&candidateLeafLocator[7],select(ERROR_TOPOLOGY,0u,valid));atomicStore(&candidateLeafLocator[8],p.ownerLayout.x);atomicStore(&candidateLeafLocator[9],p.ownerLayout.y);atomicStore(&candidateLeafLocator[10],p.ownerLayout.z);atomicStore(&candidateLeafLocator[11],p.radixLayout.z);atomicStore(&candidateLeafLocator[12],0u);atomicStore(&candidateLeafLocator[13],0u);atomicStore(&candidateLeafLocator[14],0u);atomicStore(&candidateLeafLocator[15],select(0u,inactiveOwnerTable(),valid));
 radixHeader[0]=select(0u,RADIX_VALID,valid);radixHeader[1]=select(0u,supportCount,valid);radixHeader[2]=select(0u,epoch,valid);radixHeader[3]=0u;
}

@compute @workgroup_size(64)fn emitSurfaceGraphItems(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let item=flattened(wg,lane);if(item>=atomicLoad(&candidateControl[31])){return;}let row=item/512u;let local=item%512u;let key=ownerTransaction[ownerKeyBase()+row];let page=ownerTransaction[ownerPageBase()+row];var emitted=INVALID;if(key>0u&&page>0u&&page<=p.ownerLayout.x){let logical=key-1u;let bricks=brickDims();let brick=vec3u(logical%bricks.x,(logical/bricks.x)%bricks.y,logical/(bricks.x*bricks.y));let cell=brick*8u+vec3u(local%8u,(local/8u)%8u,local/64u);if(all(cell<dims())){let owner=candidateOwner(cell);if(owner.w!=INVALID){emitted=encodeCell(owner.xyz);}else{addError(ERROR_GEOMETRY);}}}else{addError(ERROR_GEOMETRY);}radixKeys[item]=emitted;
}

@compute @workgroup_size(1)fn prepareSurfaceGraphLeaves(){let epoch=atomicLoad(&candidateControl[0]);var count=radixControl[4];if(count>0u&&radixRuns[count-1u].x==INVALID){count-=1u;}let okay=epoch!=0u&&radixControl[2]==epoch&&radixControl[3]==epoch&&radixControl[0]==0u&&count<=p.capacities.x;atomicStore(&candidateControl[1],select(0u,count,okay));if(!okay){addError(select(ERROR_CAPACITY,ERROR_GENERATION,count<=p.capacities.x));}let leafDispatch=groups(select(0u,count,okay));setDispatch(17u,leafDispatch);commitDispatch[0]=leafDispatch.x;commitDispatch[1]=leafDispatch.y;commitDispatch[2]=leafDispatch.z;let corners=count*8u;radixHeader[0]=select(0u,RADIX_VALID,okay);radixHeader[1]=select(0u,corners,okay);radixHeader[2]=select(0u,epoch,okay);radixHeader[3]=0u;}

@compute @workgroup_size(64)fn buildSurfaceGraphLeavesAndNodeItems(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let slot=flattened(wg,lane);if(slot>=atomicLoad(&candidateControl[1])){return;}let cell=radixRuns[slot].x;let origin=decodeCell(cell);let owner=candidateOwner(origin);var valid=owner.w!=INVALID&&all(owner.xyz==origin)&&isPowerOfTwo(owner.w)&&owner.w<=p.dimensionsMaximum.w;if(slot>0u&&radixRuns[slot-1u].x>=cell){valid=false;addError(ERROR_ORDER);}let span=select(1u,owner.w,valid);candidateLeaves[slot]=LeafRecord(vec4u(origin,span),vec4u(select(0u,1u,valid),slot,0u,0u),vec4u(INVALID),vec4u(INVALID));candidateLeafDirectory[slot]=LeafDirectoryRecord(cell,span,slot,select(0u,1u,valid));if(!valid){addError(ERROR_GEOMETRY);atomicAdd(&candidateControl[26],1u);}for(var c=0u;c<8u;c+=1u){let q=min(origin+vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u)*span,dims());radixKeys[slot*8u+c]=encodeNode(q);}}

@compute @workgroup_size(1)fn prepareSurfaceGraphRecords(){
 let epoch=atomicLoad(&candidateControl[0]);let okay=epoch!=0u&&radixControl[2]==epoch&&radixControl[3]==epoch&&radixControl[0]==0u&&radixControl[4]<=p.capacities.y;
 let count=select(0u,radixControl[4],okay);atomicStore(&candidateControl[2],count);if(!okay){addError(select(ERROR_CAPACITY,ERROR_GENERATION,radixControl[4]<=p.capacities.y));}
 let nodeDispatch=groups(count);setDispatch(20u,nodeDispatch);setDispatch(32u,vec3u((count+255u)/256u,1u,1u));commitDispatch[3]=nodeDispatch.x;commitDispatch[4]=nodeDispatch.y;commitDispatch[5]=nodeDispatch.z;
}

@compute @workgroup_size(64)fn buildSurfaceGraphNodes(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let slot=flattened(wg,lane);let count=atomicLoad(&candidateControl[2]);if(slot>=count){return;}let item=radixRuns[slot].x;candidateNodes[slot]=NodeRecord(item,0u,p.dimensionsMaximum.w,INVALID);candidateNodeDirectory[slot]=DirectoryRecord(item,slot);candidateConstraints[slot]=ConstraintRecord(vec4u(0u,0u,1u,0u),vec4u(INVALID),vec4u(0u));storeAdjacency(slot,AdjacencyRecord(vec3u(INVALID),vec3u(INVALID),vec3f(0.0),vec3f(0.0)));candidateValidity[slot]=0u;
}
fn nodeSlot(item:u32,required:bool)->vec2u{var lo=0u;var hi=atomicLoad(&candidateControl[2]);var probes=0u;while(lo<hi){probes+=1u;let mid=lo+(hi-lo)/2u;if(candidateNodeDirectory[mid].item<item){lo=mid+1u;}else{hi=mid;}}atomicMax(&candidateControl[12],probes);if(lo<atomicLoad(&candidateControl[2])&&candidateNodeDirectory[lo].item==item){return vec2u(lo,probes);}if(required){atomicAdd(&candidateControl[11],1u);addError(ERROR_LOOKUP);}return vec2u(INVALID,probes);}
fn leafLowerBound(cell:u32,size:u32)->u32{var lo=0u;var hi=atomicLoad(&candidateControl[1]);while(lo<hi){let mid=lo+(hi-lo)/2u;let d=candidateLeafDirectory[mid];if(identityLess(d.cell,d.size,cell,size)){lo=mid+1u;}else{hi=mid;}}return lo;}
fn containingLeaf(cell:vec3u)->u32{var span=1u;loop{if(span>p.dimensionsMaximum.w){break;}let origin=(cell/span)*span;if(all(origin+vec3u(span)<=dims())){let key=encodeCell(origin);let at=leafLowerBound(key,span);if(at<atomicLoad(&candidateControl[1])){let d=candidateLeafDirectory[at];if(d.cell==key&&d.size==span){return at;}}}span*=2u;}return INVALID;}

@compute @workgroup_size(64)fn resolveSurfaceGraphLeaves(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let row=flattened(wg,lane);let count=atomicLoad(&candidateControl[1]);if(row>=count){return;}let prior=candidateLeaves[row];let origin=prior.originSpan.xyz;let span=prior.originSpan.w;var c0=vec4u(INVALID);var c1=vec4u(INVALID);var valid=(prior.info.x&1u)!=0u&&isPowerOfTwo(span)&&span<=p.dimensionsMaximum.w&&all(origin+vec3u(span)<=dims());for(var c=0u;c<8u;c+=1u){let q=origin+vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u)*span;let slot=nodeSlot(encodeNode(q),true).x;if(c<4u){c0[c]=slot;}else{c1[c-4u]=slot;}valid=valid&&slot!=INVALID;}candidateLeaves[row]=LeafRecord(vec4u(origin,span),vec4u(select(0u,1u,valid),row,0u,0u),c0,c1);candidateLeafDirectory[row].flags=select(0u,1u,valid);if(!valid){atomicAdd(&candidateControl[26],1u);addError(ERROR_LOOKUP);}
}

@compute @workgroup_size(64)fn resolveSurfaceGraphPressureRows(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let row=flattened(wg,lane);if(row>=atomicLoad(&candidateControl[28])){return;}let h=headers[row];let at=leafLowerBound(h.cell,h.size);var graphLeaf=INVALID;if(at<atomicLoad(&candidateControl[1])){let d=candidateLeafDirectory[at];if(d.cell==h.cell&&d.size==h.size&&(d.flags&1u)!=0u){graphLeaf=d.slot;}}candidatePressureRowToLeaf[row]=graphLeaf;if(graphLeaf==INVALID){atomicAdd(&candidateControl[29],1u);addError(ERROR_COVERAGE);}}

// Topology-publication-only compiler. Every resident owner-page cell receives
// exactly one fixed leaf slot; recurring physics reaches it through the owner
// authority's direct brick directory and performs no directory search.
@compute @workgroup_size(64)fn compileSurfaceGraphLeafLocator(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let item=flattened(wg,lane);let support=atomicLoad(&candidateControl[31]);if(item>=support){return;}let row=item/512u;let local=item%512u;let key=ownerTransaction[ownerKeyBase()+row];let page=ownerTransaction[ownerPageBase()+row];var slot=INVALID;var error=0u;if(key==0u||page==0u||page>p.ownerLayout.x){error=ERROR_GEOMETRY;}else{let logical=key-1u;let bricks=brickDims();if(logical>=p.ownerLayout.y){error=ERROR_GEOMETRY;}else{let brick=vec3u(logical%bricks.x,(logical/bricks.x)%bricks.y,logical/(bricks.x*bricks.y));let cell=brick*8u+vec3u(local%8u,(local/8u)%8u,local/64u);if(all(cell<dims())){let owner=candidateOwner(cell);if(owner.w==INVALID){error=ERROR_GEOMETRY;}else{let at=leafLowerBound(encodeCell(owner.xyz),owner.w);let leafCount=atomicLoad(&candidateControl[1]);if(at<leafCount){let d=candidateLeafDirectory[at];if(d.slot<leafCount){let leaf=candidateLeaves[d.slot];if(d.cell==encodeCell(owner.xyz)&&d.size==owner.w&&(d.flags&1u)!=0u&&(leaf.info.x&1u)!=0u&&leaf.info.y==d.slot&&all(cell>=leaf.originSpan.xyz)&&all(cell<leaf.originSpan.xyz+vec3u(leaf.originSpan.w))){slot=d.slot;}}}if(slot==INVALID){error=ERROR_LOOKUP;}}}}}let destination=LOCATOR_HEADER+(page-1u)*512u+local;if(page>0u&&page<=p.ownerLayout.x&&destination<arrayLength(&candidateLeafLocator)){atomicStore(&candidateLeafLocator[destination],slot);}else{error=ERROR_CAPACITY;}if(error!=0u){atomicOr(&candidateLeafLocator[7],error);addError(error);}atomicAdd(&candidateLeafLocator[12],1u);}

fn isBoundaryCoordinate(q:u32,o:u32,s:u32)->bool{return q==o||q==o+s;}
fn findIncident(q:vec3u,corner:u32)->u32{let delta=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);if(any(q<delta)){return INVALID;}let cell=q-delta;if(any(cell>=dims())){return INVALID;}return containingLeaf(cell);}

@compute @workgroup_size(64)fn classifySurfaceGraphNodes(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let slot=flattened(wg,lane);let count=atomicLoad(&candidateControl[2]);if(slot>=count){return;}let q=decodeNode(candidateNodes[slot].item);var chosen=INVALID;var chosenSpan=0u;var chosenInterior=0u;var minimumSpan=p.dimensionsMaximum.w;var maximumSpan=1u;var reaches=vec3u(0u);var reachPositive=vec3u(0u);
 for(var incident=0u;incident<8u;incident+=1u){let leaf=findIncident(q,incident);candidateIncidentLeaves[8u*slot+incident]=leaf;if(leaf==INVALID){continue;}let r=candidateLeaves[leaf];let o=r.originSpan.xyz;let s=r.originSpan.w;minimumSpan=min(minimumSpan,s);maximumSpan=max(maximumSpan,s);var interior=0u;for(var a=0u;a<3u;a+=1u){interior+=select(1u,0u,isBoundaryCoordinate(q[a],o[a],s));}
  if((interior==1u||interior==2u)&&(s>chosenSpan||(s==chosenSpan&&leaf<chosen))){chosen=leaf;chosenSpan=s;chosenInterior=interior;}
  for(var axis=0u;axis<3u;axis+=1u){let a=(axis+1u)%3u;let b=(axis+2u)%3u;if(isBoundaryCoordinate(q[a],o[a],s)&&isBoundaryCoordinate(q[b],o[b],s)){reaches[axis]=max(reaches[axis],q[axis]-o[axis]);reachPositive[axis]=max(reachPositive[axis],o[axis]+s-q[axis]);}}
 }
 // Strict 2:1 is a face-neighbour contract. Leaves with a 4:1 span ratio may
 // legally meet only at this node (or along an edge); they share neither a
 // flux face nor a hanging-node interpolation domain. Comparing the extrema
 // of all eight incident octants rejected those valid graded layouts and
 // froze the accepted topology at the first recurring dam transition. Check
 // only octant pairs separated across one face. Repeated slots are the same
 // coarse leaf covering both local octants and need no comparison.
 for(var incident=0u;incident<8u;incident+=1u){for(var axis=0u;axis<3u;axis+=1u){let other=incident^(1u<<axis);if(other<=incident){continue;}let a=candidateIncidentLeaves[8u*slot+incident];let b=candidateIncidentLeaves[8u*slot+other];if(a==INVALID||b==INVALID||a==b){continue;}let sa=candidateLeaves[a].originSpan.w;let sb=candidateLeaves[b].originSpan.w;if(max(sa,sb)>2u*min(sa,sb)){addError(ERROR_COVERAGE);atomicAdd(&candidateControl[24],1u);}}}
 var flags=NODE_INDEPENDENT;var constraint=ConstraintRecord(vec4u(0u,0u,1u,0u),vec4u(INVALID),vec4u(0u));
 if(chosen!=INVALID){let r=candidateLeaves[chosen];let o=r.originSpan.xyz;let s=r.originSpan.w;if(chosenInterior==1u){var axis=0u;for(var a=0u;a<3u;a+=1u){if(!isBoundaryCoordinate(q[a],o[a],s)){axis=a;}}let t=q[axis]-o[axis];var qa=q;var qb=q;qa[axis]=o[axis];qb[axis]=o[axis]+s;let ma=nodeSlot(encodeNode(qa),true).x;let mb=nodeSlot(encodeNode(qb),true).x;constraint=ConstraintRecord(vec4u(NODE_EDGE_HANGING,2u,s,0u),vec4u(ma,mb,INVALID,INVALID),vec4u(s-t,t,0u,0u));flags=NODE_EDGE_HANGING;atomicAdd(&candidateControl[9],1u);
  }else{var axes=vec2u(0u,1u);var fixed=2u;var found=0u;for(var a=0u;a<3u;a+=1u){if(!isBoundaryCoordinate(q[a],o[a],s)){axes[found]=a;found+=1u;}else{fixed=a;}}let ta=q[axes.x]-o[axes.x];let tb=q[axes.y]-o[axes.y];var masters=vec4u(INVALID);for(var c=0u;c<4u;c+=1u){var m=q;m[axes.x]=o[axes.x]+(c&1u)*s;m[axes.y]=o[axes.y]+((c>>1u)&1u)*s;m[fixed]=q[fixed];masters[c]=nodeSlot(encodeNode(m),true).x;}let numerators=vec4u((s-ta)*(s-tb),ta*(s-tb),(s-ta)*tb,ta*tb);constraint=ConstraintRecord(vec4u(NODE_FACE_HANGING,4u,s*s,0u),masters,numerators);flags=NODE_FACE_HANGING;atomicAdd(&candidateControl[10],1u);}}
 else{atomicAdd(&candidateControl[8],1u);}
 if(any(q==vec3u(0u))||any(q==dims())){flags|=NODE_BOUNDARY;}candidateNodes[slot]=NodeRecord(candidateNodes[slot].item,flags,minimumSpan,select(INVALID,slot,chosen!=INVALID));candidateConstraints[slot]=constraint;if(chosen!=INVALID){atomicAdd(&candidateControl[7],1u);}
 var neg=vec3u(INVALID);var pos=vec3u(INVALID);var negSpan=vec3f(0.0);var posSpan=vec3f(0.0);for(var axis=0u;axis<3u;axis+=1u){for(var d=1u;d<=reaches[axis];d+=1u){var n=q;n[axis]-=d;let found=nodeSlot(encodeNode(n),false).x;if(found!=INVALID){neg[axis]=found;negSpan[axis]=f32(d)*p.cellSize[axis];break;}}for(var d=1u;d<=reachPositive[axis];d+=1u){var n=q;n[axis]+=d;let found=nodeSlot(encodeNode(n),false).x;if(found!=INVALID){pos[axis]=found;posSpan[axis]=f32(d)*p.cellSize[axis];break;}}}storeAdjacency(slot,AdjacencyRecord(neg,pos,negSpan,posSpan));
}

@compute @workgroup_size(64)fn validateSurfaceGraphNodes(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){
 let slot=flattened(wg,lane);if(slot>=atomicLoad(&candidateControl[2])){return;}let node=candidateNodes[slot];if(node.constraintOffset!=INVALID){let c=candidateConstraints[node.constraintOffset];var sum=0u;for(var i=0u;i<c.header.y;i+=1u){let master=c.masters[i];sum+=c.numerators[i];if(master==INVALID||master==slot||candidateNodes[master].constraintOffset!=INVALID){addError(ERROR_CONSTRAINT);}}if(sum!=c.header.z){addError(ERROR_CONSTRAINT);}}
 let a=loadAdjacency(slot);for(var axis=0u;axis<3u;axis+=1u){if(a.negative[axis]!=INVALID){let other=loadAdjacency(a.negative[axis]);if(other.positive[axis]!=slot||a.negativeSpan[axis]<=0.0||other.positiveSpan[axis]!=a.negativeSpan[axis]){addError(ERROR_ADJACENCY);atomicAdd(&candidateControl[25],1u);}}if(a.positive[axis]!=INVALID){let other=loadAdjacency(a.positive[axis]);if(other.negative[axis]!=slot||a.positiveSpan[axis]<=0.0||other.negativeSpan[axis]!=a.positiveSpan[axis]){addError(ERROR_ADJACENCY);atomicAdd(&candidateControl[25],1u);}}}
}

@compute @workgroup_size(1)fn finishSurfaceGraph(){let epoch=atomicLoad(&candidateControl[0]);var errors=atomicLoad(&candidateControl[4]);let counts=atomicLoad(&candidateControl[8])+atomicLoad(&candidateControl[9])+atomicLoad(&candidateControl[10]);if(counts!=atomicLoad(&candidateControl[2])){errors|=ERROR_CONSTRAINT;}let locatorValid=atomicLoad(&candidateLeafLocator[0])==p.radixLayout.y&&atomicLoad(&candidateLeafLocator[1])==epoch&&atomicLoad(&candidateLeafLocator[2])==atomicLoad(&candidateControl[30])&&atomicLoad(&candidateLeafLocator[3])==atomicLoad(&candidateControl[31])&&atomicLoad(&candidateLeafLocator[12])==atomicLoad(&candidateControl[31])&&atomicLoad(&candidateLeafLocator[7])==0u;if(!locatorValid){errors|=ERROR_LOOKUP;}atomicStore(&candidateLeafLocator[4],atomicLoad(&candidateControl[1]));atomicStore(&candidateLeafLocator[5],atomicLoad(&candidateControl[2]));atomicStore(&candidateLeafLocator[6],select(0u,epoch,epoch!=0u&&errors==0u&&locatorValid));atomicStore(&candidateControl[4],errors);atomicStore(&candidateControl[3],select(0u,epoch,epoch!=0u&&errors==0u&&locatorValid));}

fn readyValid(epoch:u32,pressureRows:u32)->bool{return p.readyLayout.x<arrayLength(&readyControl)&&p.readyLayout.y<arrayLength(&readyControl)&&p.readyLayout.z<arrayLength(&readyControl)&&p.readyLayout.w<arrayLength(&readyControl)&&readyControl[p.readyLayout.x]==epoch&&readyControl[p.readyLayout.y]==pressureRows&&readyControl[p.readyLayout.z]==p.expected.y&&readyControl[p.readyLayout.w]==0u;}
@compute @workgroup_size(1)fn prepareSurfaceGraphCommit(){let epoch=atomicLoad(&candidateControl[0]);let leaves=atomicLoad(&candidateControl[1]);let nodes=atomicLoad(&candidateControl[2]);let pressureRows=atomicLoad(&candidateControl[28]);let fieldGeneration=atomicLoad(&candidateControl[5]);let locatorValid=atomicLoad(&candidateLeafLocator[0])==p.radixLayout.y&&atomicLoad(&candidateLeafLocator[1])==epoch&&atomicLoad(&candidateLeafLocator[4])==leaves&&atomicLoad(&candidateLeafLocator[5])==nodes&&atomicLoad(&candidateLeafLocator[6])==epoch&&atomicLoad(&candidateLeafLocator[7])==0u&&atomicLoad(&candidateLeafLocator[13])==fieldGeneration&&atomicLoad(&candidateLeafLocator[14])==fieldGeneration;let valid=epoch!=0u&&atomicLoad(&candidateControl[3])==epoch&&atomicLoad(&candidateControl[4])==0u&&fieldGeneration!=0u&&atomicLoad(&candidateControl[6])==fieldGeneration&&locatorValid&&readyValid(epoch,pressureRows);let ld=groups(select(0u,leaves,valid));let nd=groups(select(0u,nodes,valid));let rd=groups(select(0u,pressureRows,valid));let sd=groups(select(0u,atomicLoad(&candidateControl[31]),valid));commitDispatch[0]=ld.x;commitDispatch[1]=ld.y;commitDispatch[2]=ld.z;commitDispatch[3]=nd.x;commitDispatch[4]=nd.y;commitDispatch[5]=nd.z;commitDispatch[6]=select(0u,epoch,valid);commitDispatch[7]=0u;commitDispatch[8]=rd.x;commitDispatch[9]=rd.y;commitDispatch[10]=rd.z;commitDispatch[12]=sd.x;commitDispatch[13]=sd.y;commitDispatch[14]=sd.z;}
@compute @workgroup_size(64)fn commitSurfaceGraphLeaves(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let row=flattened(wg,lane);if(row>=atomicLoad(&candidateControl[1])){return;}acceptedLeaves[row]=candidateLeaves[row];acceptedLeafDirectory[row]=candidateLeafDirectory[row];}
@compute @workgroup_size(64)fn commitSurfaceGraphSurfaceState(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let row=flattened(wg,lane);if(row>=atomicLoad(&candidateControl[1])){return;}acceptedSurfaceMass[row]=candidateSurfaceMass[row];acceptedSurfaceCompression[row]=candidateSurfaceCompression[row];}
@compute @workgroup_size(64)fn commitSurfaceGraphNodes(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let slot=flattened(wg,lane);if(slot>=atomicLoad(&candidateControl[2])){return;}acceptedNodes[slot]=candidateNodes[slot];acceptedNodeDirectory[slot]=candidateNodeDirectory[slot];}
@compute @workgroup_size(64)fn commitSurfaceGraphRelations(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let slot=flattened(wg,lane);if(slot>=atomicLoad(&candidateControl[2])){return;}acceptedConstraints[slot]=candidateConstraints[slot];let b=12u*slot;for(var word=0u;word<12u;word+=1u){acceptedAdjacency[b+word]=candidateAdjacency[b+word];}let incidentBase=8u*slot;for(var octant=0u;octant<8u;octant+=1u){acceptedIncidentLeaves[incidentBase+octant]=candidateIncidentLeaves[incidentBase+octant];}}
@compute @workgroup_size(64)fn commitSurfaceGraphFields(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let slot=flattened(wg,lane);if(slot>=atomicLoad(&candidateControl[2])){return;}acceptedPhi[slot]=candidatePhi[slot];acceptedVelocity[2u*slot]=candidateVelocity[2u*slot];acceptedVelocity[2u*slot+1u]=candidateVelocity[2u*slot+1u];acceptedValidity[slot]=candidateValidity[slot];}
@compute @workgroup_size(64)fn commitSurfaceGraphPressureRows(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let row=flattened(wg,lane);if(row>=atomicLoad(&candidateControl[28])){return;}acceptedPressureRowToLeaf[row]=candidatePressureRowToLeaf[row];}
@compute @workgroup_size(64)fn commitSurfaceGraphLeafLocator(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let item=flattened(wg,lane);if(item>=atomicLoad(&candidateControl[31])){return;}let row=item/512u;let local=item%512u;let page=ownerTransaction[ownerPageBase()+row];if(page==0u||page>p.ownerLayout.x){return;}let at=LOCATOR_HEADER+(page-1u)*512u+local;if(at<arrayLength(&candidateLeafLocator)&&at<arrayLength(&acceptedLeafLocator)){atomicStore(&acceptedLeafLocator[at],atomicLoad(&candidateLeafLocator[at]));}}
@compute @workgroup_size(1)fn finishSurfaceGraphCommit(){let epoch=commitDispatch[6];if(epoch==0u){return;}for(var i=0u;i<16u;i+=1u){if(i!=6u){atomicStore(&acceptedLeafLocator[i],atomicLoad(&candidateLeafLocator[i]));}}for(var i=0u;i<36u;i+=1u){if(i!=3u){atomicStore(&acceptedControl[i],atomicLoad(&candidateControl[i]));}}atomicStore(&acceptedLeafLocator[6],epoch);atomicStore(&acceptedControl[3],epoch);}
`;
