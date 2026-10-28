/** Fit v(x)=v0+J(x-x0) to the actual move-field samples in each cell.
 * Coordinates are lattice cells, velocities m/s. Centre the moments to
 * avoid cancellation from large window coordinates. Degenerate particle
 * clouds have no measurable 3D divergence and are reported as null. */
export function sampledCellDivergence(samples: Iterable<{
  x:number;y:number;z:number;vx:number;vy:number;vz:number;mass:number;depth:number;bx?:number;by?:number;bz?:number;
}>, h:number) {
  const cells=new Map<string,{cell:number[];n:number;mass:number;depth:number;faceDivergence:number|null;s:number[];xx:number[][];xv:number[][]}>();
  for(const p of samples){
    const cell=[Math.floor(p.x),Math.floor(p.y),Math.floor(p.z)],key=cell.join(",");
    const c=cells.get(key)??{cell,n:0,mass:0,depth:p.depth,faceDivergence:p.bx===undefined?null:p.bx+p.by!+p.bz!,s:Array<number>(6).fill(0),xx:Array.from({length:3},()=>[0,0,0]),xv:Array.from({length:3},()=>[0,0,0])};
    const a=[p.x-cell[0]!,p.y-cell[1]!,p.z-cell[2]!,p.vx,p.vy,p.vz];
    c.n++;c.mass+=p.mass;
    for(let i=0;i<6;i++)c.s[i]!+=a[i]!;
    for(let i=0;i<3;i++)for(let j=0;j<3;j++){c.xx[i]![j]!+=a[i]!*a[j]!;c.xv[i]![j]!+=a[i]!*a[j+3]!;}
    cells.set(key,c);
  }
  return [...cells.values()].map(c=>{
    const a=c.xx.map((row,i)=>row.map((v,j)=>v-c.s[i]!*c.s[j]!/c.n));
    const b=c.xv.map((row,i)=>row.map((v,j)=>v-c.s[i]!*c.s[j+3]!/c.n));
    let divergence:number|null=0;
    const scale=Math.max(...a.flat().map(Math.abs));
    for(let i=0;i<3;i++){
      let pivot=i;for(let j=i+1;j<3;j++)if(Math.abs(a[j]![i]!)>Math.abs(a[pivot]![i]!))pivot=j;
      if(c.n<4||Math.abs(a[pivot]![i]!)<1e-6*Math.max(scale,1e-12)){divergence=null;break;}
      [a[i],a[pivot]]=[a[pivot]!,a[i]!];[b[i],b[pivot]]=[b[pivot]!,b[i]!];
      const d=a[i]![i]!;for(let k=0;k<3;k++){a[i]![k]!/=d;b[i]![k]!/=d;}
      for(let j=0;j<3;j++)if(j!==i){const t=a[j]![i]!;for(let k=0;k<3;k++){a[j]![k]!-=t*a[i]![k]!;b[j]![k]!-=t*b[i]![k]!;}}
    }
    if(divergence!==null)divergence=(b[0]![0]!+b[1]![1]!+b[2]![2]!)/h;
    return {cell:c.cell,records:c.n,mass:c.mass,depth:c.depth,faceDivergence_per_s:c.faceDivergence,divergence_per_s:divergence};
  });
}
