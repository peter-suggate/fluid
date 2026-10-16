/** A full h cell seeds eight samples. Requiring eight *other* samples in
 * the 3x3x3 h cells around a sample's cell (about a two-cell radius's
 * volume) is deliberately much less dense than a one-layer sheet. Cells
 * are h cells, so the criterion follows anisotropic resolution.
 * The continuous isolation age travels with the complete particle record.
 * Never use reconstructed phi as a veto: an orphan can create its own zero. */
export const NARROW_BAND_SPARSE_NEIGHBOURS=8;
export const narrowBandSparseWGSL=/* wgsl */`
const NB_SPARSE_NEIGHBOURS=${NARROW_BAND_SPARSE_NEIGHBOURS}u;
fn nbSparseAge(i:u32,p:vec3f,age:f32)->f32{
 let lifetime=params.activity.y;
 if(lifetime<=0.0){return 0.0;}
 let cell=vec3i(floor(p));let own=nbCellRun(cell);
 // Every point in the same h cell is less than two cells away. Exclude self.
 var neighbours=own.y-own.x-1u;
 if(neighbours>=NB_SPARSE_NEIGHBOURS){return 0.0;}
 // The other samples of the 3x3x3 cells around the sample's cell, by
 // population alone (no positions): x rows of the 3x3 (y,z) neighbourhood,
 // its own row first, each split where it crosses a tile (a piece within
 // one tile is one run). The block's 27 cells are about the two-cell
 // sphere's volume (33.5); counting positions within the sphere took half
 // of update for the samples near the threshold. Reads use the immutable
 // position bank's bins; the following compaction changes them.
 var block=0u;
 for(var k=0u;k<9u;k++){
  let r=(k+4u)%9u;let c=vec3i(cell.x-1,cell.y+i32(r%3u)-1,cell.z+i32(r/3u)-1);
  let stop=min(cell.x+1,c.x|3);
  var run=nbRun(c,vec3i(stop,c.y,c.z));block+=run.y-run.x;
  if(stop<cell.x+1){run=nbRun(vec3i(stop+1,c.y,c.z),vec3i(cell.x+1,c.y,c.z));block+=run.y-run.x;}
  // The block holds the sample itself.
  if(block>NB_SPARSE_NEIGHBOURS){return 0.0;}
 }
 return min(lifetime,age+max(params.hDt.w,0.0));
}
fn nbSparseExpired(age:f32)->bool{return params.activity.y>0.0&&age>=params.activity.y;}
`;
