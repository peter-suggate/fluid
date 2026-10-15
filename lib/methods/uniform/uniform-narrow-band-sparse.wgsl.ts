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
 // Near cells first: dense liquid normally exits in the first shell. Reads
 // use the immutable position bank; the following compaction changes bins.
 for(var shell=1;shell<=2;shell++){
  for(var z=-shell;z<=shell;z++){for(var y=-shell;y<=shell;y++){for(var x=-shell;x<=shell;x++){
   let offset=vec3i(x,y,z);if(max(abs(x),max(abs(y),abs(z)))!=shell){continue;}
   if(all(offset==near)){continue;}let c=cell+offset;
   let nearest=clamp(p,vec3f(c),vec3f(c+1));let delta=nearest-p;
   if(dot(delta,delta)>NB_SPARSE_RADIUS*NB_SPARSE_RADIUS){continue;}
   let run=nbCellRun(c);
   for(var j=run.x;j<run.y;j++){
    let d=nbPosition(j)-p;
    if(j!=i&&dot(d,d)<=NB_SPARSE_RADIUS*NB_SPARSE_RADIUS){
     neighbours++;if(neighbours>=NB_SPARSE_NEIGHBOURS){return 0.0;}
    }
   }
  }}}
 }
 return min(lifetime,age+max(params.hDt.w,0.0));
}
fn nbSparseExpired(age:f32)->bool{return params.activity.y>0.0&&age>=params.activity.y;}
`;
