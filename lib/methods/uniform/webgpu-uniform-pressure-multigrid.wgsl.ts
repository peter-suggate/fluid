/** CM11a geometry and coefficient setup of the pressure root's correction
 * levels: kernels of the mixed pressure cycles' module (it supplies the rows,
 * the level table, the block neighbour table and the launch record). */
export { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "./pressure-policy";
export const uniformPressureHierarchyWGSL = /* wgsl */ `
// ---- The correction's geometry, once a solve. Level 0's phi and fractions
// are setup's; each level below is coarsened from the one above; then, a
// level at a time, phi is continued a cell into the closed cells and the
// coefficients are baked. UM_SL: the level a launch writes. A launch is a
// lane to a row (or a cell) of that level over the whole pool; a block that
// holds no page has no geometry, and its rows take only the mask that says so.

// The cell at q, in cells from block b's page corner (at most a page out):
// kind 0 a stored cell and its row, 1 air (an AIR page: far air, open, every
// fraction 1), 2 a wall (a SOLID page).
struct MGCell{kind:u32,row:u32}
fn mgCellAt(L:UMLevel,b:u32,q:vec3i)->MGCell{
 let m=i32(L.side);let d=vec3i(q>=vec3i(m))-vec3i(q<vec3i(0));
 var word=b;if(any(d!=vec3i(0))){word=umAround(b,d);}
 if(word==0u){return MGCell(1u,0u);}
 if(word>=UM_BLOCKS-1u){return MGCell(2u,0u);}
 return MGCell(0u,umLCell(L,word,vec3u(q-m*d)));
}
fn mgV(L:UMLevel,row:u32)->vec4f{let at=L.v+4u*row;return vec4f(native[at],native[at+1u],native[at+2u],native[at+3u]);}
fn mgSum8Vec4(v:array<vec4f,8>)->vec4f{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// A level's liquid: phi below the surface by a thousandth of the level's spacing.
fn mgLiquidPhi(l:u32,phi:f32)->bool{return phi< -0.001*(f32(4u<<l)*UM_MIN_H);}
fn mgSurfaceTheta(liquidPhi:f32,airPhi:f32)->f32{return clamp(-liquidPhi/max(airPhi-liquidPhi,1e-30),0.001,1000.0);}
// The coefficient of a face of open fraction v between rows of phi a and b:
// over the spacing squared, and over the ghost-fluid fraction where one is air.
fn mgFace(l:u32,axis:u32,v:f32,a:f32,b:f32)->f32{
 if(v<=1e-6){return 0.0;}
 let h=f32(4u<<l)*UM_H[axis];let la=mgLiquidPhi(l,a);let lb=mgLiquidPhi(l,b);var theta=1.0;
 if(la&&!lb){theta=mgSurfaceTheta(a,b);}
 if(!la&&lb){theta=mgSurfaceTheta(b,a);}
 return v/(h*h*theta);
}
// The phi a wall slot carries: its cell's where that is open and below the
// surface (the continuation of a closed cell, which a slot is), air otherwise.
fn mgSlotPhi(L:UMLevel,cell:u32)->f32{
 let phi=native[L.phi.x+cell];return select(2.0*UM_MIN_H,phi,native[L.v+4u*cell]>1e-5&&phi<0.0);
}
// Whether the open children of the level-F cells under coarse cell q (in
// coarse cells from block b's page corner) average to liquid. A wall has no
// open child, and counts as submerged; far air does not.
fn mgAverageSubmerged(lf:u32,F:UMLevel,b:u32,q:vec3i)->bool{
 let m=i32(F.side)/2;let d=vec3i(q>=vec3i(m))-vec3i(q<vec3i(0));
 var word=b;if(any(d!=vec3i(0))){word=umAround(b,d);}
 if(word>=UM_BLOCKS-1u){return true;}
 if(word==0u){return false;}
 let c=vec3u(q-m*d);var terms:array<f32,8>;var count=0.0;
 for(var corner=0u;corner<8u;corner++){
  let r=umLCell(F,word,2u*c+vec3u(umCorner(corner)));let open=native[F.v+4u*r]>1e-5;
  terms[corner]=select(0.0,native[F.phi.x+r],open);count+=select(0.0,1.0,open);
 }
 return count==0.0||mgLiquidPhi(lf,umSum8(terms)/count);
}
// Averaging would erase this coarse cell's air: every cell around it averages to liquid.
fn mgAirVanishes(lf:u32,F:UMLevel,b:u32,c:vec3i)->bool{
 for(var k=0u;k<27u;k++){
  if(k!=13u&&!mgAverageSubmerged(lf,F,b,c+vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1))){return false;}
 }
 return true;
}
// Level UM_SL's phi and fractions from the level above. A cell's phi is the
// average over its open children; where that is liquid though some child is
// air and the air would vanish from the level (an air layer thinner than a
// coarse cell, under a lid or beside the solid that fills the rest: its
// liquid rows would be closed on every side, a singular problem the coarsest
// solve answers with a pressure that grows with its sweeps), the average of
// its air children instead. Its open fraction is the children's average, a
// face's the average of the four child faces on it; a wall slot's fraction is
// the average of the four slots it covers.
@compute @workgroup_size(64) fn mgDownsample(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let lf=UM_SL-1u;let C=umLevel(UM_SL);let F=umLevel(lf);let j=gid.x+64u*launched.x*gid.y;
 if(j>=umLRows(C)){return;}let b=umLBlock(C,j);if(!umStored(b)){return;}
 if(j>=C.cells){
  let s=j-C.cells;let at=F.sv+(s/(C.side*C.side))*F.side*F.side+2u*(s%C.side)+F.side*2u*((s/C.side)%C.side);
  native[C.sv+s]=((native[at]+native[at+F.side+1u])+(native[at+1u]+native[at+F.side]))/4.0;return;
 }
 let c=umLLocal(C,j);
 var topologyTerms:array<vec4f,8>;var phiTerms:array<f32,8>;var openTerms:array<f32,8>;var openFlags:array<f32,8>;
 var airTerms:array<f32,8>;var airFlags:array<f32,8>;var liquidFlags:array<f32,8>;var faceTerms:array<vec4f,8>;
 for(var corner=0u;corner<8u;corner++){
  let o=umCorner(corner);let r=umLCell(F,b,2u*c+vec3u(o));
  let v=mgV(F,r);let phi=native[F.phi.x+r];topologyTerms[corner]=v;phiTerms[corner]=phi;
  let open=v.x>1e-5;openTerms[corner]=select(0.0,phi,open);openFlags[corner]=select(0.0,1.0,open);
  // A child is air unless it is liquid by the test the masks are baked with: one whose centre the
  // surface crosses holds a phi of either sign within rounding, and it has no pressure row.
  let air=open&&!mgLiquidPhi(lf,phi);
  airTerms[corner]=select(0.0,phi,air);airFlags[corner]=select(0.0,1.0,air);liquidFlags[corner]=select(0.0,1.0,open&&!air);
  faceTerms[corner]=vec4f(0.0,select(0.0,2.0*v.y,o.x==1),select(0.0,2.0*v.z,o.y==1),select(0.0,2.0*v.w,o.z==1));
 }
 let openCount=umSum8(openFlags);let airCount=umSum8(airFlags);
 let phiSum=select(umSum8(phiTerms),umSum8(openTerms)*8.0/max(openCount,1.0),openCount>0.0);
 var phi=phiSum/8.0;
 if(airCount>0.0&&umSum8(liquidFlags)>0.0&&mgLiquidPhi(lf,phi)&&mgAirVanishes(lf,F,b,vec3i(c))){phi=umSum8(airTerms)/max(airCount,1.0);}
 let fv=mgSum8Vec4(faceTerms);let topology=vec4f(mgSum8Vec4(topologyTerms).x,fv.y,fv.z,fv.w)/8.0;
 native[C.phi.x+j]=phi;for(var i=0u;i<4u;i++){native[C.v+4u*j+i]=topology[i];}
}
// Level UM_SL's phi continued one cell into its closed cells: the average of
// the open neighbours below the surface, by their open fraction. It never
// feeds a coarsening.
@compute @workgroup_size(64) fn mgExtend(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let L=umLevel(UM_SL);let j=gid.x+64u*launched.x*gid.y;
 if(j>=L.cells){return;}let b=umLBlock(L,j);if(!umStored(b)){return;}
 var phi=native[L.phi.x+j];
 if(native[L.v+4u*j]<=1e-5){
  let c=vec3i(umLLocal(L,j));var terms:array<f32,6>;var weights:array<f32,6>;
  for(var n=0u;n<6u;n++){
   terms[n]=0.0;weights[n]=0.0;let near=mgCellAt(L,b,c+umStep(n));if(near.kind!=0u){continue;}
   let v=native[L.v+4u*near.row];let other=native[L.phi.x+near.row];
   if(v>1e-5&&other<0.0){terms[n]=v*other;weights[n]=v;}
  }
  let weight=umSum6(weights);phi=select(phi,umSum6(terms)/max(weight,1e-9),weight>0.0);
 }
 native[L.phi.y+j]=phi;
}
// Level UM_SL's rows: the six face coefficients and the mask (UM_ROW: a row
// of the solve; bit 0 the row is liquid, bits 1..6 its -x,+x,-y,+y,-z,+z
// neighbours are; on the coarsest level UM_ODD, its colour). A row that is
// not liquid has no coefficients. A wall slot couples to its cell alone,
// through the wall face's fraction; a slot with no SOLID page across its
// face is no row.
@compute @workgroup_size(64) fn mgBake(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let l=UM_SL;let L=umLevel(l);let j=gid.x+64u*launched.x*gid.y;
 if(j>=umLRows(L)){return;}let b=umLBlock(L,j);let at=L.c+7u*j;
 if(!umStored(b)){native[at+6u]=0.0;return;}
 var a:array<f32,6>;var mask=UM_ROW;
 // A page's colour is its coordinates' parity, a wall slot's the other one: the two rows of a face differ.
 var odd=0u;if(l==UM_LAST){let page=umBlockPage(b);odd=u32(page.x+page.y+page.z+1)&1u;}
 if(j>=L.cells){
  let face=umLFace(L,j);let axis=face>>1u;
  if(umAcross(b,axis,face&1u)<UM_BLOCKS-1u){native[at+6u]=0.0;return;}
  let cell=umLOwner(L,j);let phi=mgSlotPhi(L,cell);odd^=1u;
  if(mgLiquidPhi(l,phi)){
   let other=native[L.phi.y+cell];var v=native[L.sv+j-L.cells];if((face&1u)==1u){v=native[L.v+4u*cell+1u+axis];}
   a[face^1u]=mgFace(l,axis,v,phi,other);mask|=1u;if(mgLiquidPhi(l,other)){mask|=2u<<(face^1u);}
  }
 }else{
  let phi=native[L.phi.y+j];
  if(mgLiquidPhi(l,phi)){
   let c=vec3i(umLLocal(L,j));mask|=1u;
   for(var n=0u;n<6u;n++){
    let axis=n>>1u;let near=mgCellAt(L,b,c+umStep(n));
    // Far air, open on every face.
    var other=UM_FAR;var v=1.0;
    if(near.kind==0u){other=native[L.phi.y+near.row];v=native[L.v+4u*near.row+1u+axis];}
    else if(near.kind==2u){other=mgSlotPhi(L,j);v=native[L.sv+umLSlot(L,b,n,vec3u(c))-L.cells];}
    if((n&1u)==1u){v=native[L.v+4u*j+1u+axis];}
    a[n]=mgFace(l,axis,v,phi,other);if(mgLiquidPhi(l,other)){mask|=2u<<n;}
   }
  }
 }
 for(var n=0u;n<6u;n++){native[at+n]=a[n];}native[at+6u]=f32(mask|(odd*UM_ODD));
}
// The root lattice's jobs, classified: bit 0 a row of the job is liquid, bit
// 1 one is bounded (the seeded bound: a wall slot's, or a closed tile's).
// A workgroup to a job, a lane to its tiles.
var<workgroup> mgLive:atomic<u32>;var<workgroup> mgBound:atomic<u32>;
@compute @workgroup_size(64) fn mgClassify(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) launched:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=g.x+launched.x*g.y;let stored=job<8u*UM_BLOCKS&&umStored(job>>3u);
 if(stored){
  let L=umLevel(0u);let k=umJobKey(job,lane);
  if((umLMask(L,k)&1u)!=0u){atomicStore(&mgLive,1u);}
  if(native[L.m.x+k]> -3.0e38){atomicStore(&mgBound,1u);}
  for(var n=0u;n<6u;n++){
   let s=umLNear(L,k,n);
   if(s>=L.cells&&native[L.m.x+s]> -3.0e38){atomicStore(&mgBound,1u);}
  }
 }
 let live=workgroupUniformLoad(&mgLive);let bound=workgroupUniformLoad(&mgBound);
 if(lane==0u&&stored){atomicStore(&umList[4u+8u*UM_BLOCKS+job],live|(bound<<1u));}
}
// The cycle list: a job that is bounded, or liquid, or beside a liquid one
// (the 26 around it, through the block neighbour table). A lane to a job.
@compute @workgroup_size(64) fn mgList(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let job=gid.x+64u*launched.x*gid.y;if(job>=8u*UM_BLOCKS){return;}
 let b=job>>3u;if(!umStored(b)){return;}
 let classes=4u+8u*UM_BLOCKS;let o=vec3i(vec3u(job&1u,(job>>1u)&1u,(job>>2u)&1u));
 var listed=(atomicLoad(&umList[classes+job])&2u)!=0u;
 for(var k=0u;k<27u&&!listed;k++){
  let q=o+vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1);let d=vec3i(q>vec3i(1))-vec3i(q<vec3i(0));
  var word=b;if(any(d!=vec3i(0))){word=umAround(b,d);}
  if(word-1u>=UM_BLOCKS-2u){continue;}
  let c=vec3u(q-2*d);listed=(atomicLoad(&umList[classes+8u*word+(c.x|(c.y<<1u)|(c.z<<2u))])&1u)!=0u;
 }
 if(!listed){return;}
 let slot=atomicAdd(&umList[0],1u);atomicStore(&umList[4u+slot],job);
}
`;
