/** Same six tetrahedra as the production NB-FLIP surface-volume authority. */
export function tetrahedralWaterFill(v:readonly number[]):number{
 if(v.length!==8||v.some(x=>!Number.isFinite(x)))throw new Error('Invalid local level set');
 const tetra=(ids:readonly number[])=>{
  const a=ids.map(i=>v[i]!).filter(x=>x<0),b=ids.map(i=>v[i]!).filter(x=>x>=0);
  if(!a.length)return 0;if(!b.length)return 1;
  if(a.length===1)return b.reduce((p,x)=>p*(-a[0]!/(x-a[0]!)),1);
  if(a.length===3)return 1-a.reduce((p,x)=>p*(b[0]!/(b[0]!-x)),1);
  const x=-a[0]!/(b[0]!-a[0]!),y=-a[0]!/(b[1]!-a[0]!),z=-a[1]!/(b[0]!-a[1]!),w=-a[1]!/(b[1]!-a[1]!);
  return Math.max(0,Math.min(1,x*y+y*z*(1-x)+z*w*(1-y)));
 };
 return [[0,1,3,7],[0,1,5,7],[0,2,3,7],[0,2,6,7],[0,4,5,7],[0,4,6,7]].reduce((s,t)=>s+tetra(t),0)/6;
}

export interface LocalWaterCell {readonly vertices:readonly number[];readonly capacity_m3:number;readonly previousCapacity_m3?:number}
export interface LocalLevelSetExchange {readonly phi_m:Float32Array;readonly cellDelta_m3:Float64Array;readonly freeDelta_m3:number;readonly requestedFreeDelta_m3:number}
/** Solve a monotone, compactly supported surface displacement. Unweighted
 * vertices remain bit-identical; measure ALL incident cells, including those
 * beyond the support, so shared corners cannot silently create/destroy water.
 * The accepted amount is measured from f32 phi, never assumed from a request. */
export function planLocalLevelSetExchange(phi_m:Float32Array,influence:ArrayLike<number>,cells:readonly LocalWaterCell[],requestedFreeDelta_m3:number,h_m:number):LocalLevelSetExchange{
 if(influence.length!==phi_m.length||!Number.isFinite(requestedFreeDelta_m3)||!(h_m>0))throw new Error('Invalid local water exchange');
 for(let i=0;i<influence.length;i++)if(!Number.isFinite(influence[i])||influence[i]!<0)throw new Error('Invalid local exchange support');
 const initial=cells.map(c=>{if(c.vertices.length!==8||c.vertices.some(i=>!Number.isInteger(i)||i<0||i>=phi_m.length)||!(c.capacity_m3>=0))throw new Error('Invalid exchange cell');return tetrahedralWaterFill(c.vertices.map(i=>phi_m[i]!))*(c.previousCapacity_m3??c.capacity_m3);});
 const atRest=cells.reduce((sum,c,i)=>sum+tetrahedralWaterFill(c.vertices.map(k=>phi_m[k]!))*c.capacity_m3-initial[i]!,0);
 const residual=requestedFreeDelta_m3-atRest,direction=residual>=0?-1:1,requested=Math.abs(residual);
 const evaluate=(shift:number)=>{
  const phi=Float32Array.from(phi_m,(v,i)=>Math.fround(v+direction*shift*influence[i]!));
  const delta=Float64Array.from(cells,(c,i)=>tetrahedralWaterFill(c.vertices.map(k=>phi[k]!))*c.capacity_m3-initial[i]!);
  return {phi_m:phi,cellDelta_m3:delta,freeDelta_m3:delta.reduce((a,b)=>a+b,0),requestedFreeDelta_m3};
 };
 if(requested<=1e-14*Math.max(1e-30,cells.reduce((sum,c)=>sum+c.capacity_m3,0)))return evaluate(0);
 const progress=(p:LocalLevelSetExchange)=>direction*(atRest-p.freeDelta_m3);
 let low=0,high=h_m,best=evaluate(0),upper=evaluate(high);
 for(let i=0;i<64&&progress(upper)<requested;i++){low=high;best=upper;high*=2;upper=evaluate(high);if(!Number.isFinite(high))break;}
 if(progress(upper)<=requested)return upper;
 for(let i=0;i<64;i++){
  const mid=(low+high)/2,result=evaluate(mid);
  if(progress(result)<=requested){low=mid;best=result;}else high=mid;
 }
 return best;
}

/** Reconstruct each connected wet reservoir independently. Dry material
 * corridors are not liquid connections. A shared derivative cell may not
 * give two reservoirs overlapping control of its level-set vertices. */
export function planLocalCapacityRemap(phi_m:Float32Array,influence:ArrayLike<number>,cells:readonly LocalWaterCell[],h_m:number):LocalLevelSetExchange{
 const parent=Int32Array.from(cells,(_,i)=>i),faces=new Map<string,number>();
 const root=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
 const wet=cells.map(c=>Math.max(c.capacity_m3,c.previousCapacity_m3??c.capacity_m3)>0&&tetrahedralWaterFill(c.vertices.map(v=>phi_m[v]!))>0);
 cells.forEach((c,i)=>{if(!wet[i])return;for(let a=0;a<3;a++)for(let side=0;side<2;side++){
  const vertices=c.vertices.filter((_,corner)=>((corner>>a)&1)===side);if(!vertices.some(v=>phi_m[v]!<0))continue;
  const key=vertices.slice().sort((a,b)=>a-b).join(','),previous=faces.get(key);if(previous===undefined)faces.set(key,i);else parent[root(i)]=root(previous);
 }});
 const vertexReservoir=new Map<number,number>(),ambiguous=new Set<number>();
 cells.forEach((c,i)=>{if(!wet[i])return;const r=root(i);for(const v of c.vertices){const owner=vertexReservoir.get(v);if(owner===undefined)vertexReservoir.set(v,r);else if(owner!==r)ambiguous.add(v);}});
 const active=Float64Array.from({length:phi_m.length},(_,v)=>vertexReservoir.has(v)&&!ambiguous.has(v)?influence[v]!:0);
 // One cell's tetrahedral volume is nonlinear in its corner values. Freeze
 // corners whose derivative cell would otherwise be edited by two solves.
 for(const cell of cells){const owners=new Set(cell.vertices.filter(v=>active[v]!>0).map(v=>vertexReservoir.get(v)!));if(owners.size>1)for(const v of cell.vertices)active[v]=0;}
 const components=new Map<number,number[]>();cells.forEach((c,i)=>{
  const owners=new Set(c.vertices.filter(v=>active[v]!>0).map(v=>vertexReservoir.get(v)!));if(wet[i])owners.add(root(i));
  for(const r of owners){const group=components.get(r)??[];group.push(i);components.set(r,group);}
 });
 const output=phi_m.slice();
 for(const [reservoir,group] of components){
  const localIndices=[...new Set(group.flatMap(i=>[...cells[i]!.vertices]))],localMap=new Map(localIndices.map((v,i)=>[v,i]));
  const result=planLocalLevelSetExchange(Float32Array.from(localIndices,v=>phi_m[v]!),Float64Array.from(localIndices,v=>vertexReservoir.get(v)===reservoir?active[v]!:0),
   group.map(i=>({...cells[i]!,vertices:cells[i]!.vertices.map(v=>localMap.get(v)!)})),0,h_m);
  const volume=group.reduce((sum,i)=>sum+(cells[i]!.previousCapacity_m3??cells[i]!.capacity_m3),0);
  if(Math.abs(result.freeDelta_m3)>Math.max(1e-18,volume*2e-6))throw new Error(`Material motion left its conservative local free-water support: delta=${result.freeDelta_m3} m3, capacity=${volume} m3, cells=${group.length}, phi=[${localIndices.reduce((m,i)=>Math.min(m,phi_m[i]!),Infinity)},${localIndices.reduce((m,i)=>Math.max(m,phi_m[i]!),-Infinity)}]`);
  localIndices.forEach((v,i)=>{if(vertexReservoir.get(v)===reservoir&&active[v]!>0)output[v]=result.phi_m[i]!;});
 }
 const delta=Float64Array.from(cells,c=>tetrahedralWaterFill(c.vertices.map(v=>output[v]!))*c.capacity_m3-tetrahedralWaterFill(c.vertices.map(v=>phi_m[v]!))*(c.previousCapacity_m3??c.capacity_m3));
 const total=cells.reduce((sum,c)=>sum+(c.previousCapacity_m3??c.capacity_m3),0);
 if(Math.abs(delta.reduce((a,b)=>a+b,0))>Math.max(1e-18,total*2e-6))throw new Error('Separated capacity remaps did not preserve their combined physical water volume');
 return {phi_m:output,cellDelta_m3:delta,freeDelta_m3:delta.reduce((a,b)=>a+b,0),requestedFreeDelta_m3:0};
}

export interface CapacityMomentumSite {readonly p:readonly number[];readonly oldVolume_m3:number;readonly deltaVolume_m3:number;readonly velocity_m_s:number}

export interface LocalPoreReceiver {
 readonly requestedBound_m3:number;
 /** Physical partition weights, summing to one for each material control. */
 readonly capacityWeights:readonly {cell:number;weight:number}[];
 readonly phiWeights:readonly {vertex:number;weight:number}[];
}
export interface LocalPoreExchange {
 readonly plan:LocalLevelSetExchange;
 readonly boundByReceiver_m3:Float64Array;
 readonly receiverShares:readonly (readonly {v:number;share:number}[])[];
}

/** Exact free/bound closure after quantizing phi. At FIXED phi, physical free
 * volume is affine in the bound partition. Solve that equation once per
 * connected local support, instead of iterating receiver credits against a
 * discontinuous f32 level-set map. A sparse transport solve gives each donor
 * cell and receiver precisely matching volume/momentum margins. */
export function planLocalPoreExchange(phi:Float32Array,cells:readonly LocalWaterCell[],receivers:readonly LocalPoreReceiver[],h:number):LocalPoreExchange{
 if(receivers.some(r=>!Number.isFinite(r.requestedBound_m3))||receivers.some(r=>r.requestedBound_m3>0)&&receivers.some(r=>r.requestedBound_m3<0))throw new Error('Local pore transactions require one finite transfer sign');
 for(const receiver of receivers)if(receiver.requestedBound_m3!==0){
  if(receiver.capacityWeights.some(w=>!Number.isInteger(w.cell)||w.cell<0||w.cell>=cells.length||!Number.isFinite(w.weight)||w.weight<0)||Math.abs(receiver.capacityWeights.reduce((sum,w)=>sum+w.weight,0)-1)>1e-10)throw new Error('Pore partition weights must conserve one control volume');
  if(receiver.phiWeights.some(w=>!Number.isInteger(w.vertex)||w.vertex<0||w.vertex>=phi.length||!Number.isFinite(w.weight)||w.weight<0))throw new Error('Invalid local pore surface support');
 }
 const n=receivers.length,parent=Int32Array.from(receivers,(_,i)=>i),support=Array.from({length:cells.length},()=>new Set<number>()),incident=new Map<number,number[]>();
 const root=(i:number):number=>{while(parent[i]!==i){parent[i]=parent[parent[i]!]!;i=parent[i]!;}return i;};
 cells.forEach((c,i)=>c.vertices.forEach(v=>{const list=incident.get(v)??[];list.push(i);incident.set(v,list);}));
 const initial=cells.map(c=>tetrahedralWaterFill(c.vertices.map(v=>phi[v]!))),capacity=cells.map(c=>c.previousCapacity_m3??c.capacity_m3);
 receivers.forEach((r,v)=>{if(!r.requestedBound_m3)return;for(const w of r.capacityWeights)if(w.weight>0)support[w.cell]!.add(v);for(const w of r.phiWeights)if(w.weight>0)for(const cell of incident.get(w.vertex)??[])support[cell]!.add(v);});
 for(const set of support){const members=[...set];for(let i=1;i<members.length;i++)parent[root(members[i]!)]=root(members[0]!);}
 const groups=new Map<number,{receivers:number[];cells:number[]}>();receivers.forEach((r,v)=>{if(r.requestedBound_m3){const at=root(v),g=groups.get(at)??{receivers:[],cells:[]};g.receivers.push(v);groups.set(at,g);}});
 support.forEach((set,c)=>{if(set.size)groups.get(root(set.values().next().value!))!.cells.push(c);});
 const resultPhi=phi.slice(),delta=new Float64Array(cells.length),accepted=new Float64Array(n),shares=Array.from({length:cells.length},()=>[] as {v:number;share:number}[]);
 for(const group of groups.values()){
  const points=[...new Set(group.cells.flatMap(c=>[...cells[c]!.vertices]))],pointIndex=new Map(points.map((v,i)=>[v,i])),cellIndex=new Map(group.cells.map((c,i)=>[c,i]));
  const localPhi=Float32Array.from(points,v=>phi[v]!),weights=new Float64Array(points.length),proposed=Float64Array.from(group.cells,c=>capacity[c]!);
  let request=0;for(const v of group.receivers){const r=receivers[v]!;request+=r.requestedBound_m3;for(const w of r.capacityWeights)proposed[cellIndex.get(w.cell)!]!-=w.weight*r.requestedBound_m3;for(const w of r.phiWeights)weights[pointIndex.get(w.vertex)!]!+=Math.abs(r.requestedBound_m3)*w.weight;}
  const maximum=weights.reduce((a,b)=>Math.max(a,b),0);if(!maximum)throw new Error('Local pore receiver has no level-set support');for(let i=0;i<weights.length;i++)weights[i]!/=maximum;
  const model=group.cells.map((c,i)=>({vertices:cells[c]!.vertices.map(v=>pointIndex.get(v)!),previousCapacity_m3:capacity[c]!,capacity_m3:proposed[i]!}));
  if(model.some(c=>c.capacity_m3<0))throw new Error('Pore capture exceeds local material capacity');
  const candidate=planLocalLevelSetExchange(localPhi,weights,model,-request,h),fill=Float64Array.from(model,c=>tetrahedralWaterFill(c.vertices.map(v=>candidate.phi_m[v]!)));
  let removedByPhi=0,unfilledCapture=0;group.cells.forEach((c,i)=>removedByPhi+=(initial[c]!-fill[i]!)*capacity[c]!);
  for(const v of group.receivers){const r=receivers[v]!,covered=r.capacityWeights.reduce((sum,w)=>sum+w.weight*fill[cellIndex.get(w.cell)!]!,0);unfilledCapture+=r.requestedBound_m3*(1-covered);}
  const tolerance=Math.max(1e-30,Math.abs(request))*1e-12;
  let scale=Math.abs(unfilledCapture)<=tolerance?1:removedByPhi/unfilledCapture;
  if(!Number.isFinite(scale)||scale< -1e-8||scale>1+1e-6)throw new Error(`Local pore affine closure is outside its requested bounds (${scale})`);
  scale=Math.max(0,Math.min(1,scale));for(const v of group.receivers)accepted[v]=scale*receivers[v]!.requestedBound_m3;
  const finalCapacity=Float64Array.from(group.cells,c=>capacity[c]!);for(const v of group.receivers)for(const w of receivers[v]!.capacityWeights)finalCapacity[cellIndex.get(w.cell)!]!-=w.weight*accepted[v]!;
  points.forEach((v,i)=>{if(weights[i]!>0)resultPhi[v]=candidate.phi_m[i]!;});
  group.cells.forEach((c,i)=>delta[c]=fill[i]!*finalCapacity[i]!-initial[c]!*capacity[c]!);
  const defect=group.cells.reduce((sum,c)=>sum+delta[c]!,0)+group.receivers.reduce((sum,v)=>sum+accepted[v]!,0);
  if(Math.abs(defect)>Math.max(1e-24,Math.abs(request)*1e-10))throw new Error(`Local pore affine closure lost physical water (${defect})`);
  // Dinic transportation on the actual compact receiver/cell adjacency.
  // No cell can credit a receiver in a disconnected support component.
  const total=group.receivers.reduce((sum,v)=>sum+Math.abs(accepted[v]!),0);if(total===0)continue;
  const source=0,cellStart=1,receiverStart=cellStart+group.cells.length,sink=receiverStart+group.receivers.length;
  type Edge={to:number;reverse:number;capacity:number;initial:number};const graph:Edge[][]=Array.from({length:sink+1},()=>[]),receiverIndex=new Map(group.receivers.map((v,i)=>[v,i]));
  const add=(a:number,b:number,amount:number):Edge=>{const edge={to:b,reverse:graph[b]!.length,capacity:amount,initial:amount};graph[a]!.push(edge);graph[b]!.push({to:a,reverse:graph[a]!.length-1,capacity:0,initial:0});return edge;};
  const donorEdges:Edge[]=[];
  group.cells.forEach((c,i)=>{const amount=Math.abs(delta[c]!);donorEdges.push(add(source,cellStart+i,amount));for(const v of support[c]!)add(cellStart+i,receiverStart+receiverIndex.get(v)!,total);});
  // A surface change can withdraw water from another cell of the same
  // compact liquid reservoir. Route through actual wet shared faces; dry
  // textile support is not a liquid connection. Keep original donor
  // provenance when decomposing the flow for paired momentum below.
  const wetFaces=new Map<string,number>();
  group.cells.forEach((c,i)=>{if(!(initial[c]!>0||fill[i]!>0))return;
   for(let axis=0;axis<3;axis++)for(let side=0;side<2;side++){
    const face=cells[c]!.vertices.filter((_,corner)=>((corner>>axis)&1)===side);
    if(!face.some(v=>phi[v]!<0||resultPhi[v]!<0))continue;
    const key=face.slice().sort((a,b)=>a-b).join(','),other=wetFaces.get(key);
    if(other===undefined)wetFaces.set(key,i);else{add(cellStart+i,cellStart+other,total);add(cellStart+other,cellStart+i,total);}
   }
  });
  group.receivers.forEach((v,i)=>add(receiverStart+i,sink,Math.abs(accepted[v]!)));
  const epsilon=total*1e-14,level=new Int32Array(graph.length),next=new Uint32Array(graph.length);let flow=0;
  for(;;){level.fill(-1);level[source]=0;const queue=[source];for(let at=0;at<queue.length;at++)for(const e of graph[queue[at]!]!)if(e.capacity>epsilon&&level[e.to]<0){level[e.to]=level[queue[at]!]!+1;queue.push(e.to);}if(level[sink]<0)break;next.fill(0);
   const push=(at:number,amount:number):number=>{if(at===sink)return amount;for(;next[at]!<graph[at]!.length;next[at]!++){const e=graph[at]![next[at]!]!;if(e.capacity<=epsilon||level[e.to]!==level[at]!+1)continue;const sent=push(e.to,Math.min(amount,e.capacity));if(sent>epsilon){e.capacity-=sent;graph[e.to]![e.reverse]!.capacity+=sent;return sent;}}return 0;};
   for(;;){const sent=push(source,total);if(sent<=epsilon)break;flow+=sent;}
  }
  if(total-flow>Math.max(1e-24,total*1e-9)){
   const error=new Error(`Local water donors cannot reach the requested receivers (${total-flow} m3)`);
   Object.defineProperty(error,'localPoreDiagnostic',{value:{phi_m:Array.from(phi),cells,receivers,h_m:h,accepted:Array.from(accepted),delta:Array.from(delta),group,flow,total}});throw error;
  }
  // Decompose accepted transport from each ORIGINAL donor, not the last
  // transit cell. Cyclic circulation has no sink and carries no exchange.
  const used=graph.map(row=>row.map(e=>e.initial>0?Math.max(0,e.initial-e.capacity):0));
  group.cells.forEach((cell,i)=>{
   let remaining=donorEdges[i]!.initial-donorEdges[i]!.capacity;const cellAmount=Math.abs(delta[cell]!);
   const path=(at:number,seen:Set<number>,edges:{from:number;index:number}[]):{v:number;edges:{from:number;index:number}[]}|undefined=>{
    if(at>=receiverStart&&at<sink)return {v:group.receivers[at-receiverStart]!,edges};
    seen.add(at);for(let j=0;j<graph[at]!.length;j++){const edge=graph[at]![j]!;if(used[at]![j]!<=epsilon||seen.has(edge.to)||edge.to===source||edge.to===sink)continue;const found=path(edge.to,seen,[...edges,{from:at,index:j}]);if(found)return found;}return undefined;
   };
   while(remaining>epsilon){const route=path(cellStart+i,new Set(),[]);if(!route)throw new Error('Accepted local water flow lost its donor provenance');const amount=Math.min(remaining,...route.edges.map(e=>used[e.from]![e.index]!));for(const e of route.edges)used[e.from]![e.index]!-=amount;remaining-=amount;
    const found=shares[cell]!.find(s=>s.v===route.v);if(found)found.share+=amount/cellAmount;else shares[cell]!.push({v:route.v,share:amount/cellAmount});
   }
  });
  // Close f64 max-flow roundoff on each donor row, so every removed fluid
  // momentum has an equal/opposite material impulse, including tiny rows.
  for(const cell of group.cells)if(delta[cell]!==0){const row=shares[cell]!,sum=row.reduce((a,b)=>a+b.share,0);if(sum>0)for(const item of row)item.share/=sum;else row.push({v:support[cell]!.values().next().value!,share:1});}
 }
 return {plan:{phi_m:resultPhi,cellDelta_m3:delta,freeDelta_m3:delta.reduce((a,b)=>a+b,0),requestedFreeDelta_m3:-receivers.reduce((sum,r)=>sum+r.requestedBound_m3,0)},boundByReceiver_m3:accepted,receiverShares:shares};
}
/** Conservative finite-volume remap on a connected wet MAC support. Removed
 * fluid carries its old velocity to newly available neighbouring capacity.
 * No solid impulse is added: pressure already supplies the geometric force. */
export function remapCapacityMomentum(sites:readonly CapacityMomentumSite[]):Float64Array{
 const result=Float64Array.from(sites,s=>s.velocity_m_s),index=new Map(sites.map((s,i)=>[s.p.join(','),i])),visited=new Set<number>();
 for(let seed=0;seed<sites.length;seed++){
  if(visited.has(seed)||Math.max(sites[seed]!.oldVolume_m3,sites[seed]!.oldVolume_m3+sites[seed]!.deltaVolume_m3)<=0)continue;
  const component=[seed];visited.add(seed);
  for(let head=0;head<component.length;head++){const at=sites[component[head]!]!;for(let a=0;a<3;a++)for(const sign of [-1,1]){
   const p=[...at.p];p[a]!+=sign;const i=index.get(p.join(','));if(i===undefined||visited.has(i))continue;
   const s=sites[i]!;if(Math.max(s.oldVolume_m3,s.oldVolume_m3+s.deltaVolume_m3)<=0)continue;visited.add(i);component.push(i);
  }}
  const donors=component.filter(i=>sites[i]!.deltaVolume_m3<0).map(i=>({i,remaining:-sites[i]!.deltaVolume_m3}));
  const total=component.reduce((s,i)=>s+sites[i]!.oldVolume_m3,0),imbalance=component.reduce((s,i)=>s+sites[i]!.deltaVolume_m3,0);
  if(Math.abs(imbalance)>Math.max(1e-18,total*2e-6))throw new Error('Capacity momentum remap cannot cross disconnected water supports');
  for(const i of component.filter(i=>sites[i]!.deltaVolume_m3>0)){
   const s=sites[i]!,mass=s.oldVolume_m3+s.deltaVolume_m3;let remaining=s.deltaVolume_m3,momentum=s.oldVolume_m3*s.velocity_m_s;
   const nearest=[...donors].sort((a,b)=>sites[a.i]!.p.reduce((d,x,k)=>d+(x-s.p[k]!)**2,0)-sites[b.i]!.p.reduce((d,x,k)=>d+(x-s.p[k]!)**2,0));
   for(const donor of nearest){const amount=Math.min(remaining,donor.remaining);momentum+=amount*sites[donor.i]!.velocity_m_s;remaining-=amount;donor.remaining-=amount;if(remaining<=0)break;}
   if(remaining>Math.max(1e-18,total*2e-6))throw new Error('Capacity remap exhausted its local fluid donors');
   momentum+=remaining*s.velocity_m_s;result[i]=momentum/mass;
  }
 }
 return result;
}
