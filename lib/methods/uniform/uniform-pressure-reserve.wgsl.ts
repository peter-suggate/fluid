/** One workgroup executes a complete reserve traversal. Storage barriers
 * provide the global ordering because this dispatch has exactly one group.
 * The ordinary parallel traversal remains preferable for expected work;
 * this path avoids dozens of empty dispatches in closed safety slots. */
export const uniformPressureReserveWGSL=/* wgsl */`
fn reserveRows(l:u32)->u32{
 let L=umLevel(l);let perBlock=(1u<<(3u*L.sh))+6u*(1u<<(2u*L.sh));
 return select(perBlock*atomicLoad(&umList[UM_COMPACT]),160u*umJobs0(),l==0u);
}
fn reserveRow(l:u32,j:u32)->u32{
 if(l!=0u){
  let L=umLevel(l);let cells=1u<<(3u*L.sh);let walls=6u*(1u<<(2u*L.sh));let perBlock=cells+walls;
  let block=atomicLoad(&umList[UM_COMPACT+1u+j/perBlock]);let local=j%perBlock;
  return select(L.cells+block*walls+local-cells,block*cells+local,local<cells);
 }
 let id=umJob0(j/160u);if(!UM_LISTED&&!umStored(id>>3u)){return UM_NONE;}
 let local=j%160u;if(local<64u){return umJobKey(id,local);}
 let face=(local-64u)/16u;let plane=(local-64u)%16u;let axis=face>>1u;
 var p=vec3u(0);p[axis]=select(0u,3u,(face&1u)!=0u);p[(axis+1u)%3u]=plane%4u;p[(axis+2u)%3u]=plane/4u;
 let cell=umJobKey(id,p.x+4u*(p.y+4u*p.z));let near=umLNear(umLevel(0u),cell,face);
 return select(UM_NONE,near,near>=umLevel(0u).cells);
}
fn reserveEnd(row:u32,value:f32,to:u32,end:u32){
 if(end==0u){native[to+row]=value;}
 else if(end==2u){mixed[UM_P+row]=value;rounding[row+1u].x=0.0;}
 else{
  umAddPressure(row,value);
  if(end==3u){umAddPressure(row,rounding[row+1u].y);umAddPressure(row,mixed[UM_BACKUP+row]);}
 }
}
// Compact 16h visits usually fit two rows per lane. Hold their stencils in
// registers and both iterates in the coarse kernel's existing shared array.
// A larger current resident set uses the complete storage traversal below.
fn reserveCompactIndex(L:UMLevel,row:u32)->u32{
 let cells=1u<<(3u*L.sh);let walls=6u*(1u<<(2u*L.sh));let block=umLBlock(L,row);
 let rank=atomicLoad(&umList[UM_COMPACT+1u+UM_BLOCKS+block]);
 return rank*(cells+walls)+select(cells+row-L.cells-block*walls,row-block*cells,row<L.cells);
}
fn reserveCachedSmooth(l:u32,start:u32,rhs:u32,low:u32,lane:u32)->u32{
 if(lane==0u){umGo=reserveRows(l);}
 let count=workgroupUniformLoad(&umGo);let L=umLevel(l);
 if(count>UM_CACHED_ROWS){return reserveSmooth(l,select(0u,1u,rhs==L.b.y),start,0u,lane);}
 var masks:array<u32,2>;var rows:array<u32,2>;var old:array<f32,2>;var minimum:array<f32,2>;var b:array<f32,2>;var diagonal:array<f32,2>;
 var weights:array<f32,12>;var neighbours:array<u32,12>;
 for(var k=0u;k<2u;k++){
  let i=lane+k*UM_LANES;if(i>=count){continue;}
  let row=reserveRow(l,i);rows[k]=row;let mask=umLMask(L,row);masks[k]=mask;
  if((mask&UM_ROW)==0u){continue;}
  old[k]=native[L.p[start]+row];minimum[k]=native[low+row];b[k]=native[rhs+row];umIterate[i]=old[k];
  var a:array<f32,6>;
  for(var f=0u;f<6u;f++){
   a[f]=native[L.c+7u*row+f];weights[6u*k+f]=a[f];neighbours[6u*k+f]=i;
   if(((mask>>(f+1u))&1u)!=0u){neighbours[6u*k+f]=reserveCompactIndex(L,umLNear(L,row,f));}
  }
  diagonal[k]=umSum6(a);
 }
 workgroupBarrier();var parity=0u;
 for(var step=0u;step<UM_WN;step++){
  for(var k=0u;k<2u;k++){
   let mask=masks[k];if((mask&UM_ROW)==0u){continue;}
   var next=max(old[k],minimum[k]);
   if((mask&1u)!=0u){
    var terms:array<f32,6>;
    for(var f=0u;f<6u;f++){terms[f]=select(0.0,weights[6u*k+f]*umIterate[UM_CACHED_ROWS*parity+neighbours[6u*k+f]],((mask>>(f+1u))&1u)!=0u);}
    let value=select(0.0,(umSum6(terms)+b[k])/diagonal[k],diagonal[k]>0.0);
    next=max(mix(old[k],value,umWeight(step)),minimum[k]);
   }
   old[k]=next;umIterate[UM_CACHED_ROWS*(parity^1u)+lane+k*UM_LANES]=next;
  }
  workgroupBarrier();parity^=1u;
 }
 let end=start^(UM_WN&1u);
 for(var k=0u;k<2u;k++){if((masks[k]&UM_ROW)!=0u){native[L.p[end]+rows[k]]=old[k];}}
 storageBarrier();return end;
}
fn reserveSmooth(l:u32,rhsSet:u32,start:u32,end:u32,lane:u32)->u32{
 let L=umLevel(l);let rows=reserveRows(l);var parity=start;
 for(var step=0u;step<UM_WN;step++){
  for(var j=lane;j<rows;j+=UM_LANES){
   let row=reserveRow(l,j);if(row==UM_NONE){continue;}
   let mask=umLMask(L,row);if((mask&UM_ROW)==0u){continue;}
   let old=native[L.p[parity]+row];let minimum=native[L.m[rhsSet]+row];var next=max(old,minimum);
   if((mask&1u)!=0u){
    var nb:array<f32,6>;
    for(var f=0u;f<6u;f++){if(((mask>>(f+1u))&1u)!=0u){nb[f]=native[L.p[parity]+umLNear(L,row,f)];}}
    next=umLRelax(L,row,mask,old,nb,L.b[rhsSet],minimum,umWeight(step));
   }
   reserveEnd(row,next,L.p[parity^1u],select(0u,end,step+1u==UM_WN));
  }
  storageBarrier();parity^=1u;
 }
 return parity;
}
fn reserveProlong(l:u32,src:u32,to:u32,coarse:u32,add:bool,lane:u32){
 let F=umLevel(l);let C=umLevel(l+1u);let rows=reserveRows(l);
 for(var j=lane;j<rows;j+=UM_LANES){
  let row=reserveRow(l,j);if(row!=UM_NONE&&(umLMask(F,row)&UM_ROW)!=0u){umLProlongAt(F,C,src,to,coarse,add,row);}
 }
 storageBarrier();
}
fn reserveCycle(a:u32,rhsSet:u32,start:u32,end:u32,lane:u32)->u32{
 var par:array<u32,4>;par[a]=start;
 for(var l=a;l<UM_SINGLE;l++){
  let L=umLevel(l);let rhs=select(0u,rhsSet,l==a);
  par[l]=reserveSmooth(l,rhs,par[l],0u,lane);
  umLTransition(l,par[l],L.b[rhs],L.m[rhs],lane);par[l+1u]=0u;
 }
 let C=umLevel(UM_SINGLE);let rhs=select(0u,rhsSet,a==UM_SINGLE);
 par[UM_SINGLE]=umLV(UM_SINGLE,C.b[rhs],C.m[rhs],par[UM_SINGLE],lane);
 for(var i=UM_SINGLE;i>a;i--){
  let l=i-1u;
  reserveProlong(l,par[l],par[l]^1u,par[i],true,lane);par[l]^=1u;
  par[l]=reserveSmooth(l,select(0u,rhsSet,l==a),par[l],select(0u,end,l==0u),lane);
 }
 return par[a];
}
fn reserveRestrict(full:bool,inner:bool,lane:u32){
 let L=umLevel(0u);let rows=reserveRows(0u);let rhs=select(UM_B0,UM_B1,inner);let low=select(UM_M0,UM_M1,inner);
 for(var j=lane;j<rows;j+=UM_LANES){
  let row=reserveRow(0u,j);if(row==UM_NONE){continue;}
  var b=0.0;
  if(row<L.cells){if(umLiquid(row)){b=mixed[rhs+row]-umRow(row).y;}}
  else{let k=umLOwner(L,row);b=mixed[rhs+row]-umWall(k,row,umLFace(L,row)>>1u)*umDifference(row,k);}
  let own=umP(row);let lower=mixed[low+row]-own;
  if(full){mixed[UM_BACKUP+row]=own;rounding[row+1u].y=rounding[row+1u].x;mixed[UM_M1+row]=lower;mixed[UM_B1+row]=b;}
  native[L.p.x+row]=0.0;native[L.b.x+row]=b;native[L.m.x+row]=lower;
 }
 storageBarrier();
}
fn reserveOpening(lane:u32){
 for(var l=0u;l<UM_LAST;l++){let L=umLevel(l);let rhs=select(1u,0u,l==0u);umLRestrict(l,L.b[rhs],L.m[rhs],lane);}
 let B=umLevel(UM_LAST);for(var j=lane;j<umLRows(B);j+=UM_LANES){native[B.p.x+j]=0.0;}storageBarrier();
 var top=umLSolve(UM_LAST,0u,B.b.y,B.m.y,lane);
 for(var i=UM_LAST;i>0u;i--){
  let l=i-1u;reserveProlong(l,0u,0u,top,false,lane);
  if(l>0u){if(l>=UM_SINGLE){let L=umLevel(l);top=umLV(l,L.b.y,L.m.y,0u,lane);}else{top=reserveCycle(l,1u,0u,0u,lane);}}
 }
}
@compute @workgroup_size(UM_LANES) fn reserveTraversal(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){umGo=select(1u,0u,umSlotClosed());}
 if(workgroupUniformLoad(&umGo)==0u){return;}
 for(var traversal=0u;traversal<select(1u,2u,UM_FULL);traversal++){
  reserveRestrict(UM_FULL&&traversal==0u,traversal==1u,lane);
  if(UM_FULL&&traversal==0u){reserveOpening(lane);}
  let result=reserveCycle(0u,0u,0u,select(1u,2u+traversal,UM_FULL),lane);
 }
}
`;
