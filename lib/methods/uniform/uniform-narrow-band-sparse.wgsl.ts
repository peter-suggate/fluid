/** A full h cell seeds eight samples. Requiring eight *other* samples in a
 * two-cell radius is deliberately much less dense than a one-layer sheet.
 * Measure in h coordinates so the criterion follows anisotropic resolution.
 * The continuous isolation age travels with the complete particle record.
 * Never use reconstructed phi as a veto: an orphan can create its own zero. */
export const NARROW_BAND_SPARSE_NEIGHBOURS=8;
export const NARROW_BAND_SPARSE_RADIUS=2;
export const narrowBandSparseWGSL=/* wgsl */`
const NB_SPARSE_NEIGHBOURS=${NARROW_BAND_SPARSE_NEIGHBOURS}u;
const NB_SPARSE_RADIUS=${NARROW_BAND_SPARSE_RADIUS}.0;
fn nbSparseAge(i:u32,p:vec3f,age:f32)->f32{
 let lifetime=params.activity.y;
 if(lifetime<=0.0){return 0.0;}
 let cell=vec3i(floor(p));let own=nbCellRun(cell);
 // Every point in the same h cell is less than two cells away. Exclude self.
 var neighbours=own.y-own.x-1u;
 if(neighbours>=NB_SPARSE_NEIGHBOURS){return 0.0;}
 // The cell across the closest face is wholly within radius two. If d is
 // that face distance, its farthest corner is at squared distance at most
 // (1+d)^2+2*(1-d)^2 <= 3, with 0 <= d <= 1/2. Its population counts without
 // fetching positions; seeded liquid (eight per cell) usually ends here.
 let fraction=p-vec3f(cell);let distance=min(fraction,1.0-fraction);
 var axis=0u;if(distance.y<distance.x){axis=1u;}if(distance.z<distance[axis]){axis=2u;}
 var near=vec3i(0);near[axis]=select(-1,1,fraction[axis]>=0.5);
 let adjacent=nbCellRun(cell+near);neighbours+=adjacent.y-adjacent.x;
 if(neighbours>=NB_SPARSE_NEIGHBOURS){return 0.0;}
 // A face cell whose farthest point (its far face on the axis, the far side
 // of this cell on the others) is within the radius counts whole, without
 // fetching positions: a lower bound, so eight or more is dense. Samples at
 // a sheet or a surface usually end here.
 let r2=NB_SPARSE_RADIUS*NB_SPARSE_RADIUS;
 let spread=max(fraction,1.0-fraction);let spread2=spread*spread;let across=spread2.x+spread2.y+spread2.z;
 var bound=neighbours;
 for(var f=0u;f<6u;f++){
  let a=f>>1u;var o=vec3i(0);o[a]=select(-1,1,(f&1u)!=0u);if(all(o==near)){continue;}
  let far=select(1.0+fraction[a],2.0-fraction[a],(f&1u)!=0u);
  if(across-spread2[a]+far*far>r2){continue;}
  let run=nbCellRun(cell+o);bound+=run.y-run.x;if(bound>=NB_SPARSE_NEIGHBOURS){return 0.0;}
 }
 // The exact count, from this cell and the adjacent one: x rows of the 5x5
 // (y,z) neighbourhood, the 3x3 core first, each clipped on x to the cells
 // the radius reaches, with those two cells cut out. A row piece within one
 // tile is one contiguous run of samples (nbRun). Reads use the immutable
 // position bank; the following compaction changes bins.
 for(var k=0;k<25;k++){
  var row=vec2i(k%3-1,k/3-1);
  if(k>=9){let r=k-9;if(r<5){row=vec2i(r-2,-2);}else if(r<10){row=vec2i(r-7,2);}else if(r<13){row=vec2i(-2,r-11);}else{row=vec2i(2,r-14);}}
  let low=vec2f(cell.yz+row);let gap=max(max(low-p.yz,p.yz-(low+1.0)),vec2f(0));let left=r2-dot(gap,gap);
  if(left<0.0){continue;}
  // A margin keeps every cell the radius touches; a farther one only adds rejected samples.
  let reach=sqrt(left)+1e-3;
  var x=max(-2,i32(ceil(fraction.x-1.0-reach)));let last=min(2,i32(floor(fraction.x+reach)));
  var cut0=1;var cut1=0;
  if(all(row==vec2i(0))){cut0=min(0,near.x);cut1=max(0,near.x);}else if(all(row==near.yz)&&near.x==0){cut0=0;cut1=0;}
  while(x<=last){
   if(cut0<=cut1&&x>=cut0&&x<=cut1){x=cut1+1;continue;}
   var stop=last;if(cut0<=cut1&&x<cut0){stop=min(stop,cut0-1);}
   let c=vec3i(cell.x+x,cell.y+row.x,cell.z+row.y);stop=min(stop,x+3-(c.x&3));
   let run=nbRun(c,vec3i(cell.x+stop,c.y,c.z));
   for(var j=run.x;j<run.y;j++){
    let d=nbPosition(j)-p;
    if(j!=i&&dot(d,d)<=r2){neighbours++;if(neighbours>=NB_SPARSE_NEIGHBOURS){return 0.0;}}
   }
   x=stop+1;
  }
 }
 return min(lifetime,age+max(params.hDt.w,0.0));
}
fn nbSparseExpired(age:f32)->bool{return params.activity.y>0.0&&age>=params.activity.y;}
`;
