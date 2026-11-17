/** Off-live executable specification for a transactional temporal broadphase.
 * Keeps all padded overlaps seen during a physical timestep. It deliberately
 * uses exhaustive traversal: GPU traversal optimizations must match this model.
 */
import {affineAxisBounds, type SweptCandidate} from './cloth-swept-candidates';
import type {AffineVertex} from './cloth-affine-ccd';
export type CandidateBox={lo:number[];hi:number[]};
const overlaps=(a:CandidateBox,b:CandidateBox)=>a.lo.every((v,k)=>v<=b.hi[k]!&&b.lo[k]!<=a.hi[k]!);
const contains=(a:CandidateBox,b:CandidateBox)=>a.lo.every((v,k)=>v<=b.lo[k]!&&a.hi[k]!>=b.hi[k]!);
const key=(c:SweptCandidate)=>c.kind+':'+c.ids.join(',');
export class CoherentCandidateReference {
  private boxes:CandidateBox[]=[];
  private pairs=new Map<string,SweptCandidate>();
  private readonly triangles:number[];
  private readonly edges:number[];
  constructor(triangles:readonly number[],edges:readonly number[],readonly skin:number,readonly thickness:number,readonly capacity=Infinity){
    if(triangles.length%3||edges.length%2||![skin,thickness].every(v=>Number.isFinite(v)&&v>=0)||!(capacity>=0)||capacity!==Infinity&&!Number.isInteger(capacity))throw new Error('Invalid coherent cache configuration');
    this.triangles=[...triangles];this.edges=[...edges];
  }
  reset(){this.boxes=[];this.pairs.clear();}
  snapshot(){return {boxes:this.boxes.map(b=>({lo:[...b.lo],hi:[...b.hi]})),pairs:[...this.pairs.values()].map(c=>({...c,ids:[...c.ids] as [number,number,number,number]}))};}
  update(vertices:readonly AffineVertex[]){
    for(const i of [...this.triangles,...this.edges])if(!Number.isInteger(i)||i<0||i>=vertices.length)throw new Error('Invalid coherent topology');
    if(this.boxes.length&&vertices.length!==this.boxes.length)throw new Error('Reset required after topology change');
    const old=this.boxes;
    const needed=vertices.map(p=>{const bounds=[0,1,2].map(k=>affineAxisBounds(p.x[k]!,p.d[k]!,this.thickness));return {lo:bounds.map(b=>b[0]),hi:bounds.map(b=>b[1])};});
    const changed=needed.map((b,i)=>!old[i]||!contains(old[i]!,b));
    // Outward round padding as well as the actual sweep. Never mutate old boxes.
    const next=needed.map((b,i)=>changed[i]?{lo:b.lo.map(v=>affineAxisBounds(v,0,this.skin)[0]),hi:b.hi.map(v=>affineAxisBounds(v,0,this.skin)[1])}:old[i]!);
    const primitive=(boxes:CandidateBox[],ids:readonly number[]):CandidateBox=>({lo:[0,1,2].map(k=>Math.min(...ids.map(i=>boxes[i]!.lo[k]!))),hi:[0,1,2].map(k=>Math.max(...ids.map(i=>boxes[i]!.hi[k]!)))});
    const all:SweptCandidate[]=[],append:SweptCandidate[]=[];let visited=0;
    const consider=(c:SweptCandidate,a:number[],b:number[])=>{
      // Exhaustive full overlap set doubles as a complete overflow fallback.
      if(!overlaps(primitive(next,a),primitive(next,b)))return;
      all.push(c);
      if(!a.some(i=>changed[i])&&!b.some(i=>changed[i]))return;
      visited++;
      if(old.length&&overlaps(primitive(old,a),primitive(old,b)))return;
      append.push(c);
    };
    for(let v=0;v<vertices.length;v++)for(let t=0;t<this.triangles.length;t+=3){const face=this.triangles.slice(t,t+3);if(!face.includes(v))consider({kind:'vt',ids:[v,...face] as [number,number,number,number]},[v],face);}
    for(let i=0;i<this.edges.length;i+=2)for(let j=i+2;j<this.edges.length;j+=2){const a=this.edges.slice(i,i+2),b=this.edges.slice(j,j+2);if(!a.some(v=>b.includes(v)))consider({kind:'ee',ids:[...a,...b] as [number,number,number,number]},a,b);}
    const merged=new Map(this.pairs);for(const c of append)merged.set(key(c),c);
    const overflow=merged.size>this.capacity;
    const committed=overflow?new Map(all.map(c=>[key(c),c])):merged;
    // Never publish boxes paired with an incomplete queue. Failed fallback
    // leaves the old transaction intact; caller must not consume it for new x/d.
    if(committed.size>this.capacity)throw new Error('Complete coherent cache rebuild exceeds capacity');
    this.boxes=next;this.pairs=committed;
    return {pairs:[...committed.values()],changedVertices:changed.filter(Boolean).length,appended:append.length,visitedOverlaps:visited,rebuilt:overflow};
  }
}
