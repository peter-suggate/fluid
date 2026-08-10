/** Clean-room adaptive nodal level-set kernels.  The graph ABI is documented by
 * `WebGPUOctreeLosassoAdaptivePhiGraphBankSource`; no dense field is bound by
 * any recurring entry point. */
export const octreeLosassoAdaptivePhiWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu; const MAGIC:u32=0x41504849u;
const ERR_GRAPH:u32=1u; const ERR_SAMPLE:u32=2u; const ERR_FINITE:u32=4u;
const ERR_CONSTRAINT:u32=8u; const ERR_REDISTANCE:u32=16u; const ERR_VOLUME:u32=32u;
const VALID:u32=1u; const FINITE:u32=2u; const INTERFACE:u32=4u;
struct Params { dims:vec4u, originCell:vec4f, limits:vec4u, numerics:vec4f,
 policy:vec4u, volume:vec4f, inflowPositionRadius:vec4f, inflowVelocityStrength:vec4f }
struct Constraint { header:vec4u, masters:vec4u, numerators:vec4u }
struct LeafRecord { originSpan:vec4u, info:vec4u, corners0:vec4u, corners1:vec4u }
struct LeafLookup { cell:u32, span:u32, slot:u32, flags:u32 }
struct Sample { value:f32, valid:u32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> graph:array<u32>;
@group(0)@binding(2)var<storage,read> leaves:array<LeafRecord>;
@group(0)@binding(3)var<storage,read> corners:array<u32>;
@group(0)@binding(4)var<storage,read> nodes:array<vec4u>;
@group(0)@binding(6)var<storage,read> leafDirectory:array<LeafLookup>;
@group(0)@binding(7)var<storage,read> constraints:array<Constraint>;
@group(0)@binding(8)var<storage,read> adjacency:array<u32>;
@group(0)@binding(9)var<storage,read_write> control:array<atomic<u32>>;
@group(0)@binding(10)var<storage,read_write> receipts:array<atomic<u32>>;
@group(0)@binding(11)var<storage,read_write> phi:array<vec2f>;
@group(0)@binding(12)var<storage,read_write> distanceA:array<f32>;
@group(0)@binding(13)var<storage,read_write> distanceB:array<f32>;
@group(0)@binding(14)var<storage,read> velocityControl:array<u32>;
@group(0)@binding(15)var<storage,read> velocity:array<vec4u>;
@group(0)@binding(17)var<storage,read_write> rowPhi:array<vec4u>;
@group(0)@binding(18)var<storage,read_write> rowGradient:array<vec4f>;
@group(0)@binding(19)var<storage,read_write> physicalVolumes:array<f32>;
@group(0)@binding(20)var<storage,read_write> referenceDelta:array<atomic<u32>>;
@group(0)@binding(21)var<storage,read> pressureRowToGraphLeaf:array<u32>;
@group(0)@binding(23)var<storage,read_write> preRedistanceVolumes:array<f32>;
@group(0)@binding(24)var<storage,read> velocityReceipt:array<u32>;
fn finite1(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn quantizePhi(v:f32)->f32{return round(v*65536.)/65536.;}
@compute @workgroup_size(1)fn applyReferenceVolumeDelta(){let delta=bitcast<f32>(atomicLoad(&referenceDelta[0]));if(delta==0.){return;}if(!finite1(delta)){atomicOr(&control[12],ERR_VOLUME);return;}var prior=atomicLoad(&control[16]);loop{let value=max(0.,bitcast<f32>(prior)+delta);let exchanged=atomicCompareExchangeWeak(&control[16],prior,bitcast<u32>(value));if(exchanged.exchanged){break;}prior=exchanged.old_value;}atomicStore(&referenceDelta[0],0u);atomicStore(&receipts[12],bitcast<u32>(delta));atomicStore(&control[17],1u);}
fn nodeCount()->u32{return min(graph[2],p.limits.x);}
fn leafCount()->u32{return min(min(graph[1],p.limits.y),arrayLength(&leaves));}
fn pressureRowCount()->u32{return min(min(graph[28],arrayLength(&pressureRowToGraphLeaf)),arrayLength(&rowPhi));}
fn currentBank()->u32{return atomicLoad(&control[6])&1u;}
fn redistanceEnabled()->bool{return atomicLoad(&control[17])!=0u;}
fn candidateDiagnostics()->bool{return p.volume.z>.5;}
fn diagnosticWord(accepted:u32,candidate:u32)->u32{return select(accepted,candidate,candidateDiagnostics());}
fn readPhi(slot:u32,bank:u32)->f32{return phi[slot][bank];}
fn nodePosition(slot:u32)->vec3u{let item=nodes[slot].x;let d=p.dims.xyz+vec3u(1u);return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}
fn cellItem(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}
fn logSpan(v:u32)->u32{var n=0u;var x=v;while(x>1u){x>>=1u;n+=1u;}return n;}
fn mortonPart(v:u32)->u32{var x=v&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(q:vec3u)->u32{return mortonPart(q.x)|(mortonPart(q.y)<<1u)|(mortonPart(q.z)<<2u);}
fn less(cellA:u32,spanA:u32,cellB:u32,spanB:u32)->bool{return cellA<cellB||(cellA==cellB&&spanA<spanB);}
fn findLeafHalfOpen(q:vec3f)->u32{if(any(q<vec3f(0))||any(q>vec3f(p.dims.xyz))){return INVALID;}
 var span=1u;loop{let origin=vec3u(floor(q/vec3f(f32(span))))*span;let cell=cellItem(origin);var lo=0u;var hi=leafCount();while(lo<hi){let mid=lo+(hi-lo)/2u;let e=leafDirectory[mid];if(less(e.cell,e.span,cell,span)){lo=mid+1u;}else{hi=mid;}}
  if(lo<leafCount()){let e=leafDirectory[lo];if(e.cell==cell&&e.span==span&&e.flags!=0u){return e.slot;}}
  if(span>=p.dims.w){break;}span*=2u;}return INVALID;}
fn findLeaf(q:vec3f)->u32{if(any(q<vec3f(0))||any(q>vec3f(p.dims.xyz))){return INVALID;}let onPlane=abs(q-round(q))<=vec3f(1e-4);for(var mask=0u;mask<8u;mask+=1u){var probe=q;var eligible=true;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1u<<axis))!=0u){if(!onPlane[axis]||q[axis]<=0.){eligible=false;}else{probe[axis]-=1e-5;}}}if(eligible){let leaf=findLeafHalfOpen(min(probe,vec3f(p.dims.xyz)-vec3f(1e-5)));if(leaf!=INVALID){return leaf;}}}return INVALID;}
fn constrainedValue(slot:u32,bank:u32)->Sample{if(slot>=nodeCount()){return Sample(0.,0u);}let c=constraints[slot];let count=c.header.y;
 if(count==0u){let v=readPhi(slot,bank);return Sample(v,select(0u,1u,finite1(v)));}
 if((count!=2u&&count!=4u)||c.header.z==0u){return Sample(0.,0u);}var terms:array<f32,4>;var numerator=0u;
 for(var j=0u;j<count;j+=1u){if(c.masters[j]>=nodeCount()){return Sample(0.,0u);}let v=readPhi(c.masters[j],bank);if(!finite1(v)){return Sample(0.,0u);}terms[j]=f32(c.numerators[j])*v;numerator+=c.numerators[j];}var value=orderedConstraintSum(terms,count);
 value/=f32(c.header.z);return Sample(value,select(0u,1u,finite1(value)&&numerator==c.header.z));}
fn cornerSlot(leaf:u32,c:u32)->u32{if(c<4u){return leaves[leaf].corners0[c];}return leaves[leaf].corners1[c-4u];}
fn neighbor(slot:u32,axis:u32,positive:bool)->u32{return adjacency[12u*slot+axis+select(0u,3u,positive)];}
fn neighborSpan(slot:u32,axis:u32,positive:bool)->f32{return bitcast<f32>(adjacency[12u*slot+6u+axis+select(0u,3u,positive)]);}
fn samplePhiAt(q:vec3f,bank:u32)->Sample{var bounded=q;var exterior=0.;
 if(q.y>f32(p.dims.y)&&p.policy.x!=0u){return Sample(p.numerics.w+(q.y-f32(p.dims.y))*p.originCell.w,1u);}
 let clamped=clamp(q,vec3f(0),vec3f(p.dims.xyz));exterior=length((q-clamped)*p.originCell.w);bounded=clamped;
 // At the upper closed wall, select the leaf immediately below it.
 let ownerPoint=min(bounded,vec3f(p.dims.xyz)-vec3f(0.00001));let leaf=findLeaf(ownerPoint);
 if(leaf==INVALID||leaf>=leafCount()){return Sample(0.,0u);}let record=leaves[leaf].originSpan;let local=clamp((bounded-vec3f(record.xyz))/f32(record.w),vec3f(0),vec3f(1));
 var terms:array<f32,8>;for(var c=0u;c<8u;c+=1u){let s=constrainedValue(cornerSlot(leaf,c),bank);
  if(s.valid==0u){return Sample(0.,0u);}let w=select(1.-local.x,local.x,(c&1u)!=0u)*select(1.-local.y,local.y,(c&2u)!=0u)*select(1.-local.z,local.z,(c&4u)!=0u);terms[c]=w*s.value;}var value=orderedSum8(terms);
 value+=exterior;return Sample(value,select(0u,1u,finite1(value)));}
fn sampleVelocityAt(q:vec3f)->vec4f{let leaf=findLeaf(min(clamp(q,vec3f(0),vec3f(p.dims.xyz)),vec3f(p.dims.xyz)-vec3f(.00001)));
 if(leaf==INVALID||leaf>=leafCount()){return vec4f(0.,0.,0.,0.);}let r=leaves[leaf].originSpan;let t=clamp((q-vec3f(r.xyz))/f32(r.w),vec3f(0),vec3f(1));var tx:array<f32,8>;var ty:array<f32,8>;var tz:array<f32,8>;
 for(var c=0u;c<8u;c+=1u){let slot=cornerSlot(leaf,c);if(slot>=nodeCount()||2u*slot>=arrayLength(&velocity)){return vec4f(0);}
  let packed=velocity[2u*slot];let s=bitcast<vec3f>(packed.xyz);if((packed.w&7u)!=7u||any(s!=s)){return vec4f(0);}let w=select(1.-t.x,t.x,(c&1u)!=0u)*select(1.-t.y,t.y,(c&2u)!=0u)*select(1.-t.z,t.z,(c&4u)!=0u);tx[c]=w*s.x;ty[c]=w*s.y;tz[c]=w*s.z;}
 return vec4f(orderedSum8(tx),orderedSum8(ty),orderedSum8(tz),1.);}
fn nodeVelocity(slot:u32)->vec4f{if(slot>=nodeCount()||2u*slot>=arrayLength(&velocity)){return vec4f(0.);}let packed=velocity[2u*slot];let value=bitcast<vec3f>(packed.xyz);if((packed.w&7u)!=7u||any(value!=value)){return vec4f(0.);}return vec4f(value,1.);}
fn inflowPhi(world:vec3f)->f32{let direction=p.inflowVelocityStrength.xyz;if(dot(direction,direction)<.5||p.inflowPositionRadius.w<=0.){return 3.402823e38;}let relative=world-p.inflowPositionRadius.xyz;let axial=dot(relative,direction);let radial=relative-axial*direction;return max(sqrt(max(0.,dot(radial,radial)))-p.inflowPositionRadius.w,max(-axial,axial-2.*p.originCell.w));}
fn orderedBefore(a:f32,b:f32)->bool{return a<b||(a==b&&bitcast<u32>(a)<bitcast<u32>(b));}
fn orderedSum4(input:array<f32,4>)->f32{var terms=input;for(var x=0u;x<4u;x+=1u){for(var y=x+1u;y<4u;y+=1u){if(orderedBefore(terms[y],terms[x])){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}return(terms[0]+terms[1])+(terms[2]+terms[3]);}
fn orderedConstraintSum(terms:array<f32,4>,count:u32)->f32{if(count==2u){return terms[0]+terms[1];}return orderedSum4(terms);}
fn orderedSum8(input:array<f32,8>)->f32{var terms=input;for(var x=0u;x<8u;x+=1u){for(var y=x+1u;y<8u;y+=1u){if(orderedBefore(terms[y],terms[x])){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}let a0=terms[0]+terms[1];let a1=terms[2]+terms[3];let a2=terms[4]+terms[5];let a3=terms[6]+terms[7];return(a0+a1)+(a2+a3);}
@compute @workgroup_size(1)fn prepareBootstrap(){for(var i=0u;i<32u;i+=1u){atomicStore(&receipts[i],0u);}for(var i=0u;i<20u;i+=1u){atomicStore(&control[i],0u);}
 atomicStore(&control[0],MAGIC);atomicStore(&control[1],graph[0]);atomicStore(&control[4],nodeCount());atomicStore(&control[5],leafCount());atomicStore(&control[6],0u);
 if(graph[0]==0u||graph[3]!=graph[0]||graph[4]!=0u||graph[2]>p.limits.x||graph[1]>p.limits.y){atomicOr(&control[12],ERR_GRAPH);}}
@compute @workgroup_size(64)fn bootstrapNodal(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}
 let v=distanceA[i];if(!finite1(v)){atomicOr(&control[12],ERR_FINITE);return;}phi[i]=vec2f(v);}
@compute @workgroup_size(64)fn bootstrapNodalLattice(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let q=nodePosition(i);let d=p.dims.xyz+vec3u(1u);let at=q.x+d.x*(q.y+d.y*q.z);if(at>=arrayLength(&distanceA)){atomicOr(&control[12],ERR_SAMPLE);return;}let v=distanceA[at];if(!finite1(v)){atomicOr(&control[12],ERR_FINITE);return;}phi[i]=vec2f(v);}
fn denseCellIndex(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}
@compute @workgroup_size(64)fn bootstrapCellCentred(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let q=vec3f(nodePosition(i));
 let bounded=clamp(q,vec3f(.5),vec3f(p.dims.xyz)-vec3f(.5));let low=vec3i(floor(bounded-vec3f(.5)));let t=bounded-(vec3f(low)+vec3f(.5));var terms:array<f32,8>;
 for(var c=0u;c<8u;c+=1u){let o=vec3i(i32(c&1u),i32((c>>1u)&1u),i32((c>>2u)&1u));let cell=vec3u(clamp(low+o,vec3i(0),vec3i(p.dims.xyz)-vec3i(1)));
  let at=denseCellIndex(cell);if(at>=arrayLength(&distanceA)||!finite1(distanceA[at])){atomicOr(&control[12],ERR_FINITE);return;}let w=select(1.-t.x,t.x,(c&1u)!=0u)*select(1.-t.y,t.y,(c&2u)!=0u)*select(1.-t.z,t.z,(c&4u)!=0u);terms[c]=w*distanceA[at];}
 phi[i]=vec2f(orderedSum8(terms));}
@compute @workgroup_size(64)fn projectAccepted(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let c=constraints[i];let count=c.header.y;if(count==0u){return;}
 var terms:array<f32,4>;var weights=0u;if((count!=2u&&count!=4u)||c.header.z==0u){atomicOr(&control[12],ERR_CONSTRAINT);return;}
 for(var j=0u;j<count;j+=1u){if(c.masters[j]>=nodeCount()){atomicOr(&control[12],ERR_CONSTRAINT);return;}weights+=c.numerators[j];}if(weights!=c.header.z){atomicOr(&control[12],ERR_CONSTRAINT);return;}
 var projected=vec2f(0.);for(var bank=0u;bank<2u;bank+=1u){for(var j=0u;j<count;j+=1u){terms[j]=f32(c.numerators[j])*readPhi(c.masters[j],bank);}var value=orderedConstraintSum(terms,count)/f32(c.header.z);if(!finite1(value)){atomicOr(&control[12],ERR_CONSTRAINT);return;}projected[bank]=value;}phi[i]=projected;}
@compute @workgroup_size(1)fn finalizeBootstrap(){let valid=atomicLoad(&control[12])==0u;atomicStore(&control[7],select(0u,1u,valid));atomicStore(&control[2],select(0u,1u,valid));
 atomicStore(&receipts[0],select(0u,1u,valid));atomicStore(&receipts[1],nodeCount());}
@compute @workgroup_size(1)fn captureReferenceVolume(){var volume=0.;for(var leaf=0u;leaf<leafCount();leaf+=1u){volume+=physicalVolumes[leaf];}
 atomicStore(&control[16],bitcast<u32>(volume));atomicStore(&receipts[11],bitcast<u32>(volume));}
@compute @workgroup_size(1)fn measureDerivations(){var volume=0.;for(var leaf=0u;leaf<leafCount();leaf+=1u){volume+=physicalVolumes[leaf];}atomicStore(&receipts[11],bitcast<u32>(volume));}
@compute @workgroup_size(1)fn prepareAdvance(){for(var i=0u;i<=12u;i+=1u){atomicStore(&receipts[i],0u);}for(var i=13u;i<=15u;i+=1u){atomicStore(&receipts[i],0u);}for(var i=32u;i<36u;i+=1u){atomicStore(&receipts[i],0u);}atomicStore(&control[12],0u);atomicStore(&control[13],0u);atomicStore(&control[14],0u);atomicStore(&control[15],0u);atomicStore(&control[17],1u);atomicStore(&control[18],0u);atomicStore(&control[19],0u);
 let reach=bitcast<f32>(velocityReceipt[1]);if(atomicLoad(&control[7])!=1u||graph[0]==0u||graph[3]!=graph[0]||graph[0]!=atomicLoad(&control[1])||graph[5]!=atomicLoad(&control[2])||velocityControl[0]!=graph[0]||velocityControl[2]<nodeCount()||velocityControl[3]!=velocityControl[0]||velocityControl[4]!=0u||velocityControl[5]!=graph[5]||velocityControl[6]!=graph[5]||velocityReceipt[0]!=graph[5]||velocityReceipt[7]!=1u||!finite1(reach)||reach<0.){atomicOr(&control[12],ERR_GRAPH);}}
@compute @workgroup_size(1)fn prepareCandidateRepair(){atomicStore(&receipts[24],0u);atomicStore(&receipts[25],0u);atomicStore(&receipts[26],0u);atomicStore(&receipts[27],0u);atomicStore(&receipts[28],0u);atomicStore(&receipts[29],0u);atomicStore(&receipts[30],0u);atomicStore(&receipts[31],0u);atomicStore(&control[13],0u);atomicStore(&control[14],0u);atomicStore(&control[15],0u);if(atomicLoad(&control[11])!=1u||graph[0]==0u||graph[3]!=graph[0]||graph[4]!=0u){atomicOr(&control[12],ERR_GRAPH);}else{atomicStore(&control[12],0u);}}
@compute @workgroup_size(64)fn projectTransported(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let c=constraints[i];let count=c.header.y;if(count==0u){return;}let bank=1u-currentBank();
 var terms:array<f32,4>;var weights=0u;if((count!=2u&&count!=4u)||c.header.z==0u){atomicOr(&control[12],ERR_CONSTRAINT);return;}for(var j=0u;j<count;j+=1u){let m=c.masters[j];if(m>=nodeCount()){atomicOr(&control[12],ERR_CONSTRAINT);return;}terms[j]=f32(c.numerators[j])*readPhi(m,bank);weights+=c.numerators[j];}var value=orderedConstraintSum(terms,count);
 value/=f32(c.header.z);if(!finite1(value)||weights!=c.header.z){atomicOr(&control[12],ERR_CONSTRAINT);return;}phi[i][bank]=value;if(!candidateDiagnostics()){atomicAdd(&control[18],1u);}}
@compute @workgroup_size(1)fn captureTransportReceipt(){let maximumDelta=atomicLoad(&control[13]);atomicStore(&receipts[0],bitcast<u32>(p.numerics.x));atomicStore(&receipts[1],maximumDelta);atomicStore(&receipts[2],atomicLoad(&control[14]));atomicStore(&receipts[3],atomicLoad(&control[15]));atomicStore(&receipts[4],atomicLoad(&control[18]));atomicStore(&receipts[5],atomicLoad(&control[19]));let retain=maximumDelta==0u&&atomicLoad(&receipts[12])==0u&&p.inflowVelocityStrength.w==0.;atomicStore(&control[17],select(1u,0u,retain));atomicStore(&control[13],0u);atomicStore(&control[14],0u);atomicStore(&control[15],0u);}
@compute @workgroup_size(64)fn canonicalizeCandidatePhi(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()||constraints[i].header.y!=0u){return;}let value=readPhi(i,1u-currentBank());if(!finite1(value)){atomicOr(&control[12],ERR_FINITE);return;}phi[i]=vec2f(quantizePhi(value));}
fn trilinear(values:array<f32,8>,t:vec3f)->f32{var terms:array<f32,8>;for(var c=0u;c<8u;c+=1u){terms[c]=values[c]*select(1.-t.x,t.x,(c&1u)!=0u)*select(1.-t.y,t.y,(c&2u)!=0u)*select(1.-t.z,t.z,(c&4u)!=0u);}return orderedSum8(terms);}
fn leafFractionAtBank(leaf:u32,bank:u32)->Sample{if(leaf>=leafCount()){return Sample(0.,0u);}var values:array<f32,8>;for(var c=0u;c<8u;c+=1u){let value=constrainedValue(cornerSlot(leaf,c),bank);if(value.valid==0u){return Sample(0.,0u);}values[c]=value.value;}
 var liquid=0.;for(var z=0u;z<3u;z+=1u){for(var y=0u;y<3u;y+=1u){for(var x=0u;x<3u;x+=1u){let value=trilinear(values,(vec3f(f32(x),f32(y),f32(z))+vec3f(.5))/3.);
  liquid+=select(0.,1.,value<0.);}}}let fraction=liquid/27.;return Sample(fraction,select(0u,1u,finite1(fraction)));}
fn leafVolumeAtBank(leaf:u32,bank:u32)->Sample{let fraction=leafFractionAtBank(leaf,bank);let width=f32(leaves[leaf].originSpan.w)*p.originCell.w;let volume=fraction.value*width*width*width;return Sample(volume,select(0u,1u,fraction.valid!=0u&&finite1(volume)));}
@compute @workgroup_size(64)fn capturePreRedistanceVolumes(@builtin(global_invocation_id)gid:vec3u){let leaf=gid.x;if(leaf>=leafCount()){return;}let bank=select(currentBank(),1u-currentBank(),redistanceEnabled());let measured=leafVolumeAtBank(leaf,bank);if(measured.valid==0u){preRedistanceVolumes[leaf]=0.;atomicOr(&control[12],ERR_VOLUME|ERR_FINITE);return;}preRedistanceVolumes[leaf]=measured.value;}
@compute @workgroup_size(64)fn deriveRows(@builtin(global_invocation_id)gid:vec3u){let row=gid.x;if(row>=pressureRowCount()){return;}let leaf=pressureRowToGraphLeaf[row];if(leaf>=leafCount()){atomicOr(&control[12],ERR_GRAPH);return;}let r=leaves[leaf].originSpan;var v:array<f32,8>;let bank=select(1u-currentBank(),currentBank(),p.volume.w>.5);var mn=3.402823e38;var mx=-3.402823e38;
 for(var c=0u;c<8u;c+=1u){let s=constrainedValue(cornerSlot(leaf,c),bank);if(s.valid==0u){atomicOr(&control[12],ERR_SAMPLE);return;}v[c]=s.value;mn=min(mn,s.value);mx=max(mx,s.value);}
 let centre=clamp(trilinear(v,vec3f(.5)),mn,mx);let flags=VALID|FINITE|select(0u,INTERFACE,mn<=0.&&mx>=0.);rowPhi[row]=vec4u(bitcast<u32>(centre),bitcast<u32>(mn),bitcast<u32>(mx),flags);
 let scale=max(f32(r.w)*p.originCell.w,1e-20);
 let gx=.25*orderedSum4(array<f32,4>(v[1]-v[0],v[3]-v[2],v[5]-v[4],v[7]-v[6]))/scale;
 let gy=.25*orderedSum4(array<f32,4>(v[2]-v[0],v[3]-v[1],v[6]-v[4],v[7]-v[5]))/scale;
 let gz=.25*orderedSum4(array<f32,4>(v[4]-v[0],v[5]-v[1],v[6]-v[2],v[7]-v[3]))/scale;
 rowGradient[row]=vec4f(gx,gy,gz,1.);}
@compute @workgroup_size(64)fn deriveLeafVolumes(@builtin(global_invocation_id)gid:vec3u){let leaf=gid.x;if(leaf>=leafCount()){return;}let scale=f32(leaves[leaf].originSpan.w)*p.originCell.w;let bank=select(1u-currentBank(),currentBank(),p.volume.w>.5);let fraction=leafFractionAtBank(leaf,bank);if(fraction.valid==0u){atomicOr(&control[12],ERR_VOLUME|ERR_FINITE);physicalVolumes[leaf]=0.;return;}physicalVolumes[leaf]=fraction.value*scale*scale*scale;}
@compute @workgroup_size(64)fn derivePostRedistanceVolumes(@builtin(global_invocation_id)gid:vec3u){let leaf=gid.x;if(leaf>=leafCount()){return;}let bank=select(currentBank(),1u-currentBank(),redistanceEnabled());let measured=leafVolumeAtBank(leaf,bank);if(measured.valid==0u){atomicOr(&control[12],ERR_VOLUME|ERR_FINITE);physicalVolumes[leaf]=0.;return;}physicalVolumes[leaf]=measured.value;atomicAdd(&receipts[select(40u,48u,candidateDiagnostics())],1u);}
@compute @workgroup_size(1)fn finalizeAccepted(){atomicStore(&receipts[8],atomicLoad(&control[14]));atomicStore(&receipts[9],atomicLoad(&control[15]));atomicStore(&receipts[13],atomicLoad(&control[16]));
 let residual=bitcast<f32>(atomicLoad(&receipts[6]));if(residual>p.numerics.z){atomicOr(&control[12],ERR_REDISTANCE);}if(atomicLoad(&receipts[43])==0u){atomicOr(&control[12],ERR_VOLUME);}let valid=atomicLoad(&control[12])==0u;
 if(valid){if(redistanceEnabled()){atomicStore(&control[6],1u-currentBank());}atomicAdd(&control[2],1u);}atomicStore(&control[7],select(0u,1u,valid));atomicStore(&control[13],select(0u,1u,valid));atomicStore(&receipts[14],select(0u,1u,valid));}
@compute @workgroup_size(1)fn finalizeCandidateRepair(){atomicStore(&receipts[27],atomicLoad(&control[14]));atomicStore(&receipts[28],atomicLoad(&control[15]));atomicStore(&receipts[29],atomicLoad(&control[16]));let residual=bitcast<f32>(atomicLoad(&receipts[24]));if(residual>p.numerics.z){atomicOr(&control[12],ERR_REDISTANCE);}if(atomicLoad(&receipts[51])==0u){atomicOr(&control[12],ERR_VOLUME);}let valid=atomicLoad(&control[12])==0u;atomicStore(&control[13],select(0u,1u,valid));atomicStore(&receipts[31],select(0u,1u,valid));if(candidateDiagnostics()&&atomicLoad(&control[7])==1u){atomicStore(&control[12],0u);}}
`;

/** Transactional evidence gate for exact pre/post-redistance volume measurement.
 * Physical drift is telemetry only; graph, coverage, finiteness, and hanging
 * constraint closure remain fail-closed publication requirements. */
export const octreeLosassoAdaptivePhiVolumeEvidenceWGSL = /* wgsl */ `
const MAGIC:u32=0x41504849u;const ERR_GRAPH:u32=1u;const ERR_FINITE:u32=4u;const ERR_CONSTRAINT:u32=8u;const ERR_VOLUME:u32=32u;
struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f}struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}struct Sample{value:f32,valid:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read_write>state:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>receipts:array<atomic<u32>>;@group(0)@binding(5)var<storage,read>phi:array<vec2f>;@group(0)@binding(6)var<storage,read>physicalVolumes:array<f32>;@group(0)@binding(7)var<storage,read>preRedistanceVolumes:array<f32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn candidate()->bool{return p.volume.z>.5;}fn diagnosticWord(accepted:u32,candidateWord:u32)->u32{return select(accepted,candidateWord,candidate());}fn volumeReceiptBase()->u32{return select(36u,44u,candidate());}fn nodeCount()->u32{return min(graph[2],min(p.limits.x,arrayLength(&constraints)));}fn leafCount()->u32{return min(graph[1],min(p.limits.y,arrayLength(&physicalVolumes)));}fn bank()->u32{let current=atomicLoad(&state[6])&1u;return select(current,1u-current,atomicLoad(&state[17])!=0u);}
fn master(slot:u32,b:u32)->Sample{if(slot>=nodeCount()||constraints[slot].header.y!=0u){return Sample(0.,0u);}let value=phi[slot][b];return Sample(value,select(0u,1u,finite(value)));}fn projected(slot:u32,b:u32)->Sample{if(slot>=nodeCount()){return Sample(0.,0u);}let c=constraints[slot];let count=c.header.y;if(count==0u){return master(slot,b);}if((count!=2u&&count!=4u)||c.header.z==0u){return Sample(0.,0u);}let a=master(c.masters.x,b);let d=master(c.masters.y,b);if(a.valid*d.valid==0u){return Sample(0.,0u);}let t0=f32(c.numerators.x)*a.value;let t1=f32(c.numerators.y)*d.value;var numerator=c.numerators.x+c.numerators.y;var value=t0+t1;if(count==4u){let e=master(c.masters.z,b);let f=master(c.masters.w,b);if(e.valid*f.valid==0u){return Sample(0.,0u);}var terms=array<f32,4>(t0,t1,f32(c.numerators.z)*e.value,f32(c.numerators.w)*f.value);for(var x=0u;x<4u;x+=1u){for(var y=x+1u;y<4u;y+=1u){if(terms[y]<terms[x]||(terms[y]==terms[x]&&bitcast<u32>(terms[y])<bitcast<u32>(terms[x]))){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}value=(terms[0]+terms[1])+(terms[2]+terms[3]);numerator+=c.numerators.z+c.numerators.w;}value/=f32(c.header.z);return Sample(value,select(0u,1u,finite(value)&&numerator==c.header.z));}
@compute @workgroup_size(1)fn prepareVolumeEvidence(){let base=volumeReceiptBase();for(var i=0u;i<8u;i+=1u){atomicStore(&receipts[base+i],0u);}atomicStore(&receipts[base],graph[0]);atomicStore(&receipts[base+1u],select(graph[5],atomicLoad(&state[2]),candidate()));let epoch=graph[0];var matches=epoch!=0u&&graph[3]==epoch&&graph[4]==0u&&graph[1]<=p.limits.y&&graph[2]<=p.limits.x&&graph[2]<=arrayLength(&constraints)&&graph[2]<=arrayLength(&phi)&&graph[1]<=arrayLength(&physicalVolumes)&&graph[1]<=arrayLength(&preRedistanceVolumes);if(candidate()){matches=matches&&atomicLoad(&state[0])==MAGIC&&atomicLoad(&state[8])==epoch&&atomicLoad(&state[10])==graph[2]&&atomicLoad(&state[11])==1u&&atomicLoad(&state[2])!=0u;}else{matches=matches&&atomicLoad(&state[0])==MAGIC&&atomicLoad(&state[1])==epoch&&atomicLoad(&state[2])==graph[5]&&atomicLoad(&state[3])==graph[6]&&atomicLoad(&state[4])==graph[2]&&atomicLoad(&state[5])==graph[1]&&atomicLoad(&state[7])==1u;}if(!matches){atomicOr(&state[12],ERR_GRAPH);}}
@compute @workgroup_size(1)fn validateVolumeEvidence(){let base=volumeReceiptBase();let epoch=graph[0];let generation=select(graph[5],atomicLoad(&state[2]),candidate());var matches=atomicLoad(&receipts[base])==epoch&&atomicLoad(&receipts[base+1u])==generation&&epoch!=0u&&graph[3]==epoch&&graph[4]==0u&&graph[1]==leafCount()&&graph[2]==nodeCount();if(candidate()){matches=matches&&atomicLoad(&state[8])==epoch&&atomicLoad(&state[10])==graph[2]&&atomicLoad(&state[11])==1u;}else{matches=matches&&atomicLoad(&state[1])==epoch&&atomicLoad(&state[2])==generation&&atomicLoad(&state[3])==graph[6]&&atomicLoad(&state[4])==graph[2]&&atomicLoad(&state[5])==graph[1]&&atomicLoad(&state[7])==1u;}if(!matches){atomicOr(&state[12],ERR_GRAPH);}var constrainedCount=0u;for(var i=0u;i<nodeCount();i+=1u){let value=phi[i][bank()];if(!finite(value)){atomicOr(&state[12],ERR_VOLUME|ERR_FINITE);}if(constraints[i].header.y!=0u){constrainedCount+=1u;let expected=projected(i,bank());if(expected.valid==0u||abs(value-expected.value)>p.volume.y){atomicOr(&state[12],ERR_VOLUME|ERR_CONSTRAINT);}}}atomicStore(&receipts[base+2u],nodeCount());atomicStore(&receipts[base+3u],constrainedCount);if(atomicLoad(&receipts[base+4u])!=leafCount()){atomicOr(&state[12],ERR_VOLUME);}}
@compute @workgroup_size(1)fn finalizeVolumeEvidence(){let base=volumeReceiptBase();let epoch=graph[0];let generation=select(graph[5],atomicLoad(&state[2]),candidate());let matches=atomicLoad(&receipts[base])==epoch&&atomicLoad(&receipts[base+1u])==generation&&epoch!=0u&&graph[3]==epoch&&graph[4]==0u&&graph[1]==leafCount()&&atomicLoad(&receipts[base+4u])==leafCount();if(!matches){atomicOr(&state[12],ERR_GRAPH|ERR_VOLUME);}var measured=0.;var signedDrift=0.;var totalAbsoluteDrift=0.;var maximumLeafDrift=0.;for(var leaf=0u;leaf<leafCount();leaf+=1u){let post=physicalVolumes[leaf];let pre=preRedistanceVolumes[leaf];let drift=pre-post;if(!finite(post)||post<0.||!finite(pre)||pre<0.||!finite(drift)){atomicOr(&state[12],ERR_VOLUME|ERR_FINITE);}measured+=post;signedDrift+=drift;totalAbsoluteDrift+=abs(drift);maximumLeafDrift=max(maximumLeafDrift,abs(drift));}if(!finite(measured)||!finite(signedDrift)||!finite(totalAbsoluteDrift)||!finite(maximumLeafDrift)){atomicOr(&state[12],ERR_VOLUME|ERR_FINITE);}let valid=atomicLoad(&state[12])==0u;atomicStore(&receipts[diagnosticWord(10u,30u)],bitcast<u32>(signedDrift));atomicStore(&receipts[base+5u],bitcast<u32>(maximumLeafDrift));atomicStore(&receipts[base+6u],bitcast<u32>(totalAbsoluteDrift));atomicStore(&receipts[base+7u],select(0u,1u,valid));atomicStore(&receipts[select(11u,25u,candidate())],bitcast<u32>(measured));}
`;

/** Compact scheduled characteristic backtrace; no scalar interpolation is reachable here. */
/** Arena reset and GPU-authored reach publication; deliberately two bindings. */
export const octreeLosassoAdaptivePhiWorklistPrepareWGSL = /* wgsl */ `
struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}@group(0)@binding(0)var<storage,read_write>arena:array<atomic<u32>>;@group(0)@binding(1)var<storage,read>velocityReceipt:array<u32>;@group(0)@binding(2)var<uniform>p:Params;
fn resetArena(){for(var i=0u;i<13u;i+=1u){atomicStore(&arena[i],0u);}}@compute @workgroup_size(1)fn prepareTransportBand(){resetArena();atomicStore(&arena[11],velocityReceipt[1]);}@compute @workgroup_size(1)fn prepareRedistanceBand(){resetArena();atomicStore(&arena[11],bitcast<u32>(2.*p.originCell.w));atomicStore(&arena[12],1u);}
`;

/** Initialize every live mask from current phi and the GPU-authored reach. */
export const octreeLosassoAdaptivePhiWorklistReachWGSL = /* wgsl */ `
struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}
const INVALID:u32=0xffffffffu;const ERR_GRAPH:u32=1u;struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(3)var<storage,read>phi:array<vec2f>;@group(0)@binding(4)var<storage,read_write>arena:array<atomic<u32>>;@group(0)@binding(5)var<storage,read_write>bandMask:array<u32>;@group(0)@binding(6)var<storage,read>leaves:array<LeafRecord>;@group(0)@binding(7)var<storage,read>incidentLeaves:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn nodeCount()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&phi)),arrayLength(&bandMask));}fn leafCount()->u32{return min(graph[1],arrayLength(&leaves));}fn cornerSlot(r:LeafRecord,c:u32)->u32{if(c<4u){return r.corners0[c];}return r.corners1[c-4u];}fn changesSign(a:f32,b:f32)->bool{return(a<=0.&&b>=0.)||(a>=0.&&b<=0.);}
@compute @workgroup_size(64)fn markTransportReach(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let reach=bitcast<f32>(atomicLoad(&arena[11]));let current=atomicLoad(&control[6])&1u;let bank=select(current,1u-current,atomicLoad(&arena[12])==1u);let centre=phi[i][bank];if(!finite(centre)){atomicOr(&control[12],ERR_GRAPH);bandMask[i]=0u;return;}var scheduled=abs(centre)<reach;var mixed=false;var frozen=false;if(8u*i+7u>=arrayLength(&incidentLeaves)){atomicOr(&control[12],ERR_GRAPH);}else{for(var incident=0u;incident<8u;incident+=1u){let leaf=incidentLeaves[8u*i+incident];if(leaf==INVALID){continue;}if(leaf>=leafCount()){atomicOr(&control[12],ERR_GRAPH);continue;}let record=leaves[leaf];var minimumPhi=3.402823e38;var maximumPhi=-3.402823e38;var localCorner=INVALID;var valid=true;for(var corner=0u;corner<8u;corner+=1u){let slot=cornerSlot(record,corner);if(slot>=nodeCount()){atomicOr(&control[12],ERR_GRAPH);valid=false;continue;}let value=phi[slot][bank];if(!finite(value)){atomicOr(&control[12],ERR_GRAPH);valid=false;continue;}scheduled=scheduled||abs(value)<reach;minimumPhi=min(minimumPhi,value);maximumPhi=max(maximumPhi,value);if(slot==i){localCorner=corner;}}if(!valid||localCorner==INVALID){atomicOr(&control[12],ERR_GRAPH);continue;}mixed=mixed||(minimumPhi<=0.&&maximumPhi>=0.);for(var axis=0u;axis<3u;axis+=1u){let other=cornerSlot(record,localCorner^(1u<<axis));if(other>=nodeCount()){atomicOr(&control[12],ERR_GRAPH);continue;}let otherPhi=phi[other][bank];if(!finite(otherPhi)){atomicOr(&control[12],ERR_GRAPH);continue;}frozen=frozen||changesSign(centre,otherPhi);}}}scheduled=scheduled||mixed||frozen;bandMask[i]=select(0u,1u,scheduled)|select(0u,2u,frozen);}
`;

/** Conservatively union authored inflow-cylinder support into the live mask. */
export const octreeLosassoAdaptivePhiWorklistInflowWGSL = /* wgsl */ `
struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>nodes:array<vec4u>;@group(0)@binding(3)var<storage,read_write>bandMask:array<u32>;
fn nodeCount()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&nodes)),arrayLength(&bandMask));}fn nodePosition(i:u32)->vec3u{let item=nodes[i].x;let d=p.dims.xyz+vec3u(1);return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}
@compute @workgroup_size(64)fn markTransportInflow(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;let direction=p.inflowVelocityStrength.xyz;if(i>=nodeCount()||(bandMask[i]&1u)!=0u||p.inflowVelocityStrength.w<=0.||dot(direction,direction)<.5){return;}let relative=p.originCell.xyz+vec3f(nodePosition(i))*p.originCell.w-p.inflowPositionRadius.xyz;let axial=dot(relative,direction);let radial=relative-axial*direction;let reach=p.numerics.y;let radius=p.inflowPositionRadius.w+reach;if(axial>=-reach&&axial<=2.*p.originCell.w+reach&&dot(radial,radial)<=radius*radius){bandMask[i]|=1u;}}
`;

/** Publish independent scheduled/retained nodes from the finalized mark mask. */
export const octreeLosassoAdaptivePhiWorklistIndependentWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>phi:array<vec2f>;@group(0)@binding(5)var<storage,read_write>transportControl:array<atomic<u32>>;@group(0)@binding(6)var<storage,read_write>worklist:array<u32>;@group(0)@binding(7)var<storage,read>bandMask:array<u32>;
fn nodeCount()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&constraints)),arrayLength(&bandMask));}fn bank()->u32{return atomicLoad(&control[6])&1u;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}@compute @workgroup_size(64)fn publishTransportIndependent(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()||constraints[i].header.y!=0u){return;}let marked=bandMask[i];if(marked>3u){atomicOr(&control[12],ERR_GRAPH);return;}if((marked&1u)==0u){if(atomicLoad(&transportControl[12])==0u){let current=bank();phi[i][1u-current]=phi[i][current];}atomicAdd(&transportControl[1],1u);return;}let rank=atomicAdd(&transportControl[0],1u);if(rank>=capacity()){atomicOr(&control[12],ERR_GRAPH);atomicStore(&transportControl[4],1u);return;}worklist[rank]=i|select(0u,0x80000000u,(marked&2u)!=0u&&atomicLoad(&transportControl[12])==1u);}
`;

/** Validate constrained masters and derive final constrained masks without list atomics. */
export const octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL = /* wgsl */ `
const ERR_CONSTRAINT:u32=8u;struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
@group(0)@binding(0)var<storage,read>graph:array<u32>;@group(0)@binding(1)var<storage,read>constraints:array<Constraint>;@group(0)@binding(2)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(3)var<storage,read_write>bandMask:array<atomic<u32>>;
fn nodeCount()->u32{return min(min(graph[2],arrayLength(&constraints)),arrayLength(&bandMask));}fn validConstraint(c:Constraint)->bool{return(c.header.y==2u||c.header.y==4u)&&c.header.z!=0u;}
@compute @workgroup_size(64)fn markTransportConstraintMasters(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let c=constraints[i];let marked=atomicLoad(&bandMask[i]);if(c.header.y==0u||marked==0u){return;}if(marked>3u||!validConstraint(c)){atomicOr(&control[12],ERR_CONSTRAINT);return;}for(var j=0u;j<c.header.y;j+=1u){let master=c.masters[j];if(master>=nodeCount()||constraints[master].header.y!=0u){atomicOr(&control[12],ERR_CONSTRAINT);return;}atomicOr(&bandMask[master],marked&1u);}}
@compute @workgroup_size(64)fn markTransportConstrained(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let c=constraints[i];if(c.header.y==0u){return;}let direct=atomicLoad(&bandMask[i]);atomicStore(&bandMask[i],0u);if(direct>3u||!validConstraint(c)){atomicOr(&control[12],ERR_CONSTRAINT);return;}var scheduled=false;for(var j=0u;j<c.header.y;j+=1u){let master=c.masters[j];if(master>=nodeCount()||constraints[master].header.y!=0u){atomicOr(&control[12],ERR_CONSTRAINT);return;}let marked=atomicLoad(&bandMask[master]);if(marked>3u){atomicOr(&control[12],ERR_CONSTRAINT);return;}scheduled=scheduled||(marked&1u)!=0u;}atomicStore(&bandMask[i],select(0u,1u,scheduled)|(direct&2u));}
`;

/** Publish constrained scheduled/retained nodes from finalized masks. */
export const octreeLosassoAdaptivePhiWorklistConstrainedWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>phi:array<vec2f>;@group(0)@binding(5)var<storage,read_write>transportControl:array<atomic<u32>>;@group(0)@binding(6)var<storage,read_write>worklist:array<u32>;@group(0)@binding(7)var<storage,read_write>bandMask:array<u32>;
fn nodeCount()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&constraints)),arrayLength(&bandMask));}fn bank()->u32{return atomicLoad(&control[6])&1u;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}@compute @workgroup_size(64)fn publishTransportConstrained(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()||constraints[i].header.y==0u){return;}let marked=bandMask[i];if(marked>3u){atomicOr(&control[12],ERR_GRAPH);return;}if((marked&1u)==0u){if(atomicLoad(&transportControl[12])==0u){let current=bank();phi[i][1u-current]=phi[i][current];}atomicAdd(&transportControl[3],1u);return;}let rank=atomicAdd(&transportControl[2],1u);if(rank>=capacity()){atomicOr(&control[12],ERR_GRAPH);atomicStore(&transportControl[4],1u);return;}worklist[capacity()+rank]=i|select(0u,0x80000000u,(marked&2u)!=0u&&atomicLoad(&transportControl[12])==1u);}
`;

/** Validate list capacity and publish the two indirect dispatch records. */
export const octreeLosassoAdaptivePhiWorklistFinalizeWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(2)var<storage,read_write>arena:array<u32>;
fn capacity()->u32{return p.limits.x;}@compute @workgroup_size(1)fn finalizeTransportDispatch(){let rawIndependent=arena[0];let rawConstrained=arena[2];let cap=capacity();let independent=min(rawIndependent,cap);let constrained=min(rawConstrained,cap);if(rawIndependent>cap||rawConstrained>cap||arena[4]!=0u){atomicOr(&control[12],ERR_GRAPH);}arena[5]=(independent+63u)/64u;arena[6]=1u;arena[7]=1u;arena[8]=(constrained+63u)/64u;arena[9]=1u;arena[10]=1u;}
`;

/** Publish the raw scheduled/retained partition without modifying dispatch state. */
export const octreeLosassoAdaptivePhiWorklistReceiptWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>arena:array<atomic<u32>>;@group(0)@binding(1)var<storage,read_write>receipts:array<atomic<u32>>;
@compute @workgroup_size(1)fn publishTransportPartition(){atomicStore(&receipts[32],atomicLoad(&arena[0]));atomicStore(&receipts[33],atomicLoad(&arena[1]));atomicStore(&receipts[34],atomicLoad(&arena[2]));atomicStore(&receipts[35],atomicLoad(&arena[3]));}
`;

/** One-entry canonical projection of active hanging-node constraints. */
export const octreeLosassoAdaptivePhiWorklistProjectWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;const ERR_CONSTRAINT:u32=8u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>phi:array<vec2f>;@group(0)@binding(5)var<storage,read_write>transportControl:array<atomic<u32>>;@group(0)@binding(6)var<storage,read>worklist:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn quantizePhi(v:f32)->f32{return round(v*65536.)/65536.;}fn nodeCount()->u32{return min(graph[2],p.limits.x);}fn bank()->u32{return atomicLoad(&control[6])&1u;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn before(a:f32,b:f32)->bool{return a<b||(a==b&&bitcast<u32>(a)<bitcast<u32>(b));}fn orderedConstraintSum(input:array<f32,4>,count:u32)->f32{var t=input;for(var x=0u;x<count;x+=1u){for(var y=x+1u;y<count;y+=1u){if(before(t[y],t[x])){let s=t[x];t[x]=t[y];t[y]=s;}}}if(count==2u){return t[0]+t[1];}return(t[0]+t[1])+(t[2]+t[3]);}
@compute @workgroup_size(64)fn projectTransportedBand(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;let count=min(atomicLoad(&transportControl[2]),capacity());if(work>=count){return;}let i=worklist[capacity()+work]&0x7fffffffu;if(i>=nodeCount()){atomicOr(&control[12],ERR_GRAPH);return;}let c=constraints[i];let masters=c.header.y;let next=1u-bank();if((masters!=2u&&masters!=4u)||c.header.z==0u){atomicOr(&control[12],ERR_CONSTRAINT);return;}var terms:array<f32,4>;var weights=0u;for(var j=0u;j<masters;j+=1u){let m=c.masters[j];if(m>=nodeCount()){atomicOr(&control[12],ERR_CONSTRAINT);return;}terms[j]=f32(c.numerators[j])*phi[m][next];weights+=c.numerators[j];}let value=orderedConstraintSum(terms,masters)/f32(c.header.z);if(!finite(value)||weights!=c.header.z){atomicOr(&control[12],ERR_CONSTRAINT);return;}phi[i][next]=value;atomicAdd(&control[18],1u);}
`;

const adaptivePhiRedistanceParamsWGSL = /* wgsl */ `struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}`;

export const octreeLosassoAdaptivePhiRedistanceInitializeWGSL = /* wgsl */ `${adaptivePhiRedistanceParamsWGSL}
const INVALID:u32=0xffffffffu;const ERR_GRAPH:u32=1u;const FROZEN:u32=0x80000000u;const FAR:f32=3.402823e38;struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>leaves:array<LeafRecord>;@group(0)@binding(3)var<storage,read>incidentLeaves:array<u32>;@group(0)@binding(4)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(5)var<storage,read>phi:array<vec2f>;@group(0)@binding(6)var<storage,read_write>distanceA:array<f32>;@group(0)@binding(7)var<storage,read_write>distanceB:array<f32>;@group(0)@binding(8)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(9)var<storage,read_write>workControl:array<atomic<u32>>;@group(0)@binding(10)var<storage,read>worklist:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<FAR;}fn count()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&phi)),min(arrayLength(&distanceA),arrayLength(&distanceB)));}fn leafCount()->u32{return min(min(graph[1],p.limits.y),arrayLength(&leaves));}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn compactAccepted()->bool{return p.volume.w>.5;}fn cornerSlot(r:LeafRecord,c:u32)->u32{if(c<4u){return r.corners0[c];}return r.corners1[c-4u];}
struct EdgeSeed{distance:f32,frozen:u32}fn changesSign(a:f32,b:f32)->bool{return(a<=0.&&b>=0.)||(a>=0.&&b<=0.);}
fn exactEdgeSeed(i:u32,b:u32)->EdgeSeed{if(i>=count()||8u*i+7u>=arrayLength(&incidentLeaves)){atomicOr(&control[12],ERR_GRAPH);return EdgeSeed(FAR,0u);}let centre=phi[i][b];if(!finite(centre)){atomicOr(&control[12],ERR_GRAPH);return EdgeSeed(FAR,0u);}let centreMagnitude=abs(centre);var seed=FAR;var frozen=0u;for(var incident=0u;incident<8u;incident+=1u){let leaf=incidentLeaves[8u*i+incident];if(leaf==INVALID){continue;}if(leaf>=leafCount()){atomicOr(&control[12],ERR_GRAPH);continue;}let r=leaves[leaf];var localCorner=INVALID;for(var c=0u;c<8u;c+=1u){let slot=cornerSlot(r,c);if(slot>=count()){atomicOr(&control[12],ERR_GRAPH);continue;}if(slot==i){localCorner=c;}}if(localCorner==INVALID){atomicOr(&control[12],ERR_GRAPH);continue;}let edgeLength=f32(r.originSpan.w)*p.originCell.w;for(var axis=0u;axis<3u;axis+=1u){let other=cornerSlot(r,localCorner^(1u<<axis));if(other>=count()){atomicOr(&control[12],ERR_GRAPH);continue;}let otherPhi=phi[other][b];if(!finite(otherPhi)){atomicOr(&control[12],ERR_GRAPH);continue;}if(changesSign(centre,otherPhi)){let denominator=centreMagnitude+abs(otherPhi);var crossing=0.;if(denominator>0.){crossing=edgeLength*centreMagnitude/denominator;}if(!finite(crossing)||crossing<0.){atomicOr(&control[12],ERR_GRAPH);continue;}seed=min(seed,crossing);frozen=1u;}}}return EdgeSeed(seed,frozen);}
fn initializeNode(i:u32,taggedFrozen:bool,validateTag:bool){if(atomicLoad(&control[17])==0u||i>=count()){return;}let b=1u-(atomicLoad(&control[6])&1u);let crossing=exactEdgeSeed(i,b);let frozen=crossing.frozen!=0u;if(validateTag&&taggedFrozen!=frozen){atomicOr(&control[12],ERR_GRAPH);}var seed=FAR;if(frozen){seed=bitcast<f32>(bitcast<u32>(crossing.distance)|FROZEN);}distanceA[i]=seed;distanceB[i]=seed;if(frozen){atomicAdd(&control[14],1u);}}
@compute @workgroup_size(64)fn initializeRedistance(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i==0u){atomicStore(&redistanceReceipt[1],0u);}if(compactAccepted()||i>=count()){return;}initializeNode(i,false,false);}
@compute @workgroup_size(64)fn initializeAcceptedRedistanceIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;if(!compactAccepted()||work>=min(atomicLoad(&workControl[0]),capacity())){return;}let raw=worklist[work];initializeNode(raw&0x7fffffffu,(raw&FROZEN)!=0u,true);}
@compute @workgroup_size(64)fn initializeAcceptedRedistanceConstrained(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;if(!compactAccepted()||work>=min(atomicLoad(&workControl[2]),capacity())){return;}let raw=worklist[capacity()+work];initializeNode(raw&0x7fffffffu,(raw&FROZEN)!=0u,true);}
`;

export const octreeLosassoAdaptivePhiRedistanceResetWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(1)var<storage,read_write>transportControl:array<atomic<u32>>;
@compute @workgroup_size(1)fn resetRedistanceResidual(){atomicStore(&redistanceReceipt[0],0u);}
@compute @workgroup_size(1)fn prepareAcceptedRedistance(){atomicStore(&redistanceReceipt[0],0u);atomicStore(&redistanceReceipt[1],0u);}
`;

const makeAdaptivePhiRedistanceSweepWGSL = (entry: string) => /* wgsl */ `${adaptivePhiRedistanceParamsWGSL}
const INVALID:u32=0xffffffffu;const FROZEN:u32=0x80000000u;struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read>adjacency:array<u32>;@group(0)@binding(4)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(5)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(6)var<storage,read>distanceInput:array<f32>;@group(0)@binding(7)var<storage,read_write>distanceOutput:array<f32>;@group(0)@binding(8)var<storage,read>bandMask:array<u32>;@group(0)@binding(9)var<storage,read_write>workControl:array<atomic<u32>>;@group(0)@binding(10)var<storage,read>worklist:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn count()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&constraints)),min(arrayLength(&distanceInput),arrayLength(&distanceOutput)));}fn compactAccepted()->bool{return p.volume.w>.5;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn bandActive(i:u32)->bool{return p.volume.z>.5||(i<arrayLength(&bandMask)&&(bandMask[i]&1u)!=0u);}fn neighbor(i:u32,a:u32,pos:bool)->u32{return adjacency[12u*i+a+select(0u,3u,pos)];}fn span(i:u32,a:u32,pos:bool)->f32{return bitcast<f32>(adjacency[12u*i+6u+a+select(0u,3u,pos)]);}
fn axisMinimum(i:u32,axis:u32)->vec2f{var result=vec2f(3.402823e38);let negative=neighbor(i,axis,false);if(negative!=INVALID&&negative<count()&&bandActive(negative)){result=vec2f(abs(distanceInput[negative]),span(i,axis,false));}let positive=neighbor(i,axis,true);if(positive!=INVALID&&positive<count()&&bandActive(positive)){let candidate=vec2f(abs(distanceInput[positive]),span(i,axis,true));if(candidate.x<result.x||(candidate.x==result.x&&candidate.y<result.y)){result=candidate;}}return result;}
fn pairBefore(a:vec2f,b:vec2f)->bool{return a.x<b.x||(a.x==b.x&&a.y<b.y);}fn solveQuadratic(sw:f32,saw:f32,saa:f32)->f32{return(saw+sqrt(max(0.,saw*saw-sw*(saa-1.))))/sw;}
fn eikonal(i:u32)->f32{var a=axisMinimum(i,0u);var b=axisMinimum(i,1u);var c=axisMinimum(i,2u);if(pairBefore(b,a)){let t=a;a=b;b=t;}if(pairBefore(c,b)){let t=b;b=c;c=t;}if(pairBefore(b,a)){let t=a;a=b;b=t;}if(!finite(a.x)||!finite(a.y)||!(a.y>0.)){return 3.402823e38;}var sw=1./(a.y*a.y);var saw=a.x*sw;var saa=a.x*a.x*sw;var result=solveQuadratic(sw,saw,saa);if(!finite(b.x)||!finite(b.y)||!(b.y>0.)||result<=b.x){return result;}let wb=1./(b.y*b.y);sw+=wb;saw+=b.x*wb;saa+=b.x*b.x*wb;result=solveQuadratic(sw,saw,saa);if(!finite(c.x)||!finite(c.y)||!(c.y>0.)||result<=c.x){return result;}let wc=1./(c.y*c.y);return solveQuadratic(sw+wc,saw+c.x*wc,saa+c.x*c.x*wc);}
@compute @workgroup_size(64)fn ${entry}(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;var i=work;if(compactAccepted()){if(work>=min(atomicLoad(&workControl[0]),capacity())){return;}i=worklist[work]&0x7fffffffu;}if(atomicLoad(&control[17])==0u||i>=count()||constraints[i].header.y!=0u||!bandActive(i)){return;}let prior=distanceInput[i];if((bitcast<u32>(prior)&FROZEN)!=0u){distanceOutput[i]=prior;return;}let update=min(prior,eikonal(i));distanceOutput[i]=update;if(finite(prior)&&finite(update)&&prior<3.402823e38){atomicMax(&redistanceReceipt[0],bitcast<u32>(abs(prior-update)));}}
`;
export const octreeLosassoAdaptivePhiRedistanceAtoBWGSL = makeAdaptivePhiRedistanceSweepWGSL("redistanceAtoB");
export const octreeLosassoAdaptivePhiRedistanceBtoAWGSL = makeAdaptivePhiRedistanceSweepWGSL("redistanceBtoA");

const makeAdaptivePhiRedistanceProjectWGSL = (entry: string) => /* wgsl */ `${adaptivePhiRedistanceParamsWGSL}
const ERR_CONSTRAINT:u32=8u;const FROZEN:u32=0x80000000u;struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>constraints:array<Constraint>;@group(0)@binding(3)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>distance:array<f32>;@group(0)@binding(5)var<storage,read>bandMask:array<u32>;@group(0)@binding(6)var<storage,read_write>workControl:array<atomic<u32>>;@group(0)@binding(7)var<storage,read>worklist:array<u32>;
const FAR:f32=3.402823e38;fn finite(v:f32)->bool{return v==v&&abs(v)<FAR;}fn count()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&constraints)),arrayLength(&distance));}fn compactAccepted()->bool{return p.volume.w>.5;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn bandActive(i:u32)->bool{return p.volume.z>.5||(i<arrayLength(&bandMask)&&(bandMask[i]&1u)!=0u);}fn before(a:f32,b:f32)->bool{return a<b||(a==b&&bitcast<u32>(a)<bitcast<u32>(b));}
fn orderedFour(v:vec4f)->f32{let p01=before(v.x,v.y);let a0=select(v.y,v.x,p01);let a1=select(v.x,v.y,p01);let p23=before(v.z,v.w);let a2=select(v.w,v.z,p23);let a3=select(v.z,v.w,p23);let p02=before(a0,a2);let b0=select(a2,a0,p02);let b2=select(a0,a2,p02);let p13=before(a1,a3);let b1=select(a3,a1,p13);let b3=select(a1,a3,p13);let p12=before(b1,b2);let c1=select(b2,b1,p12);let c2=select(b1,b2,p12);return(b0+c1)+(c2+b3);}
@compute @workgroup_size(64)fn ${entry}(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;var i=work;if(compactAccepted()){if(work>=min(atomicLoad(&workControl[2]),capacity())){return;}i=worklist[capacity()+work]&0x7fffffffu;}let nodeCount=count();if(atomicLoad(&control[17])==0u||i>=nodeCount||constraints[i].header.y==0u||!bandActive(i)){return;}if((bitcast<u32>(distance[i])&FROZEN)!=0u){return;}let c=constraints[i];let n=c.header.y;if((n!=2u&&n!=4u)||c.header.z==0u||c.masters.x>=nodeCount||c.masters.y>=nodeCount){atomicOr(&control[12],ERR_CONSTRAINT);return;}let d0=abs(distance[c.masters.x]);let d1=abs(distance[c.masters.y]);let k0=finite(d0);let k1=finite(d1);if(n==2u){if(!k0&&!k1){distance[i]=FAR;return;}if(!k0||!k1){distance[i]=min(abs(distance[i]),select(d1,d0,k0));return;}distance[i]=(f32(c.numerators.x)*d0+f32(c.numerators.y)*d1)/f32(c.header.z);return;}if(c.masters.z>=nodeCount||c.masters.w>=nodeCount){atomicOr(&control[12],ERR_CONSTRAINT);return;}let d2=abs(distance[c.masters.z]);let d3=abs(distance[c.masters.w]);let k2=finite(d2);let k3=finite(d3);if(!(k0&&k1&&k2&&k3)){let reached=min(min(select(FAR,d0,k0),select(FAR,d1,k1)),min(select(FAR,d2,k2),select(FAR,d3,k3)));distance[i]=min(abs(distance[i]),reached);return;}let t0=f32(c.numerators.x)*d0;let t1=f32(c.numerators.y)*d1;let t2=f32(c.numerators.z)*d2;let t3=f32(c.numerators.w)*d3;distance[i]=orderedFour(vec4f(t0,t1,t2,t3))/f32(c.header.z);}
`;
export const octreeLosassoAdaptivePhiRedistanceProjectAWGSL = makeAdaptivePhiRedistanceProjectWGSL("projectDistanceA");
export const octreeLosassoAdaptivePhiRedistanceProjectBWGSL = makeAdaptivePhiRedistanceProjectWGSL("projectDistanceB");

export const octreeLosassoAdaptivePhiRedistanceFinishWGSL = /* wgsl */ `${adaptivePhiRedistanceParamsWGSL}
const ERR_REDISTANCE:u32=16u;const FROZEN:u32=0x80000000u;@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(3)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>phi:array<vec2f>;@group(0)@binding(5)var<storage,read>distance:array<f32>;@group(0)@binding(6)var<storage,read>bandMask:array<u32>;@group(0)@binding(7)var<storage,read_write>workControl:array<atomic<u32>>;@group(0)@binding(8)var<storage,read>worklist:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn quantizePhi(v:f32)->f32{return round(v*65536.)/65536.;}fn count()->u32{return min(min(min(graph[2],p.limits.x),arrayLength(&phi)),arrayLength(&distance));}fn compactAccepted()->bool{return p.volume.w>.5;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn bandActive(i:u32)->bool{return p.volume.z>.5||(i<arrayLength(&bandMask)&&(bandMask[i]&1u)!=0u);}fn finishNode(i:u32){if(atomicLoad(&control[17])==0u||i>=count()||!bandActive(i)){return;}let b=1u-(atomicLoad(&control[6])&1u);let encoded=distance[i];if((bitcast<u32>(encoded)&FROZEN)!=0u){atomicAdd(&redistanceReceipt[1],1u);return;}let sign=select(1.,-1.,phi[i][b]<=0.);var d=encoded;if(!finite(d)||d>=3.402823e38){if(abs(phi[i][b])>=p.numerics.y){d=p.numerics.y;}else{atomicOr(&control[12],ERR_REDISTANCE);return;}}phi[i][b]=quantizePhi(sign*min(d,p.numerics.y));atomicAdd(&redistanceReceipt[1],1u);}
@compute @workgroup_size(64)fn finishRedistance(@builtin(global_invocation_id)gid:vec3u){if(compactAccepted()){return;}finishNode(gid.x);}
@compute @workgroup_size(64)fn finishAcceptedRedistanceIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;if(!compactAccepted()||work>=min(atomicLoad(&workControl[0]),capacity())){return;}finishNode(worklist[work]&0x7fffffffu);}
@compute @workgroup_size(64)fn finishAcceptedRedistanceConstrained(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;if(!compactAccepted()||work>=min(atomicLoad(&workControl[2]),capacity())){return;}finishNode(worklist[capacity()+work]&0x7fffffffu);}
`;

export const octreeLosassoAdaptivePhiRedistancePublishAcceptedWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(1)var<storage,read_write>receipts:array<atomic<u32>>;@group(0)@binding(2)var<storage,read_write>workControl:array<atomic<u32>>;
@compute @workgroup_size(1)fn publishAcceptedRedistanceReceipt(){atomicStore(&receipts[6],atomicLoad(&redistanceReceipt[0]));let reached=atomicLoad(&redistanceReceipt[1]);let activeIndependent=atomicLoad(&workControl[0]);let activeConstrained=atomicLoad(&workControl[2]);atomicStore(&receipts[7],select(0u,atomicLoad(&receipts[32])+atomicLoad(&receipts[34]),reached!=0u));atomicStore(&receipts[52],select(0u,activeIndependent,reached!=0u));atomicStore(&receipts[53],select(0u,activeConstrained,reached!=0u));}
`;

export const octreeLosassoAdaptivePhiRedistancePublishCandidateWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>redistanceReceipt:array<atomic<u32>>;@group(0)@binding(1)var<storage,read_write>receipts:array<atomic<u32>>;
@compute @workgroup_size(1)fn publishCandidateRedistanceReceipt(){atomicStore(&receipts[24],atomicLoad(&redistanceReceipt[0]));atomicStore(&receipts[26],atomicLoad(&redistanceReceipt[1]));}
`;


export const octreeLosassoAdaptivePhiBacktraceWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const ERR_GRAPH:u32=1u;const ERR_SAMPLE:u32=2u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>leaves:array<LeafRecord>;@group(0)@binding(3)var<storage,read>nodes:array<vec4u>;@group(0)@binding(4)var<storage,read>incidentLeaves:array<u32>;@group(0)@binding(5)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(6)var<storage,read>velocity:array<vec4u>;@group(0)@binding(7)var<storage,read_write>transportControl:array<atomic<u32>>;@group(0)@binding(8)var<storage,read>worklist:array<u32>;@group(0)@binding(9)var<storage,read_write>departures:array<vec4f>;
fn nodeCount()->u32{return min(graph[2],p.limits.x);}fn leafCount()->u32{return min(min(graph[1],p.limits.y),arrayLength(&leaves));}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn nodePosition(i:u32)->vec3u{let item=nodes[i].x;let d=p.dims.xyz+vec3u(1);return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}fn packedVelocity(slot:u32)->vec4f{if(slot>=nodeCount()||2u*slot>=arrayLength(&velocity)){return vec4f(0);}let raw=velocity[2u*slot];let v=bitcast<vec3f>(raw.xyz);if((raw.w&7u)!=7u||any(v!=v)){return vec4f(0);}return vec4f(v,1);}
fn productD4(a:f32,b:f32,c:f32)->f32{var x=a;var y=b;var z=c;if(y<x){let swap=x;x=y;y=swap;}if(z<y){let swap=y;y=z;z=swap;}if(y<x){let swap=x;x=y;y=swap;}return(x*y)*z;}
fn sumVelocityD4(a:vec3f,b:vec3f,c:vec3f,d:vec3f,e:vec3f,f:vec3f,g:vec3f,h:vec3f)->vec3f{return((a+f)+(b+e))+((c+h)+(d+g));}
fn sampleVelocity(q:vec3f,leaf:u32)->vec4f{if(leaf>=leafCount()){return vec4f(0);}let r=leaves[leaf];let v0=packedVelocity(r.corners0.x);let v1=packedVelocity(r.corners0.y);let v2=packedVelocity(r.corners0.z);let v3=packedVelocity(r.corners0.w);let v4=packedVelocity(r.corners1.x);let v5=packedVelocity(r.corners1.y);let v6=packedVelocity(r.corners1.z);let v7=packedVelocity(r.corners1.w);if(v0.w*v1.w*v2.w*v3.w*v4.w*v5.w*v6.w*v7.w!=1.){return vec4f(0);}let t=clamp((q-vec3f(r.originSpan.xyz))/f32(r.originSpan.w),vec3f(0),vec3f(1));let lo=vec3f(1)-t;let value=sumVelocityD4(productD4(lo.x,lo.y,lo.z)*v0.xyz,productD4(t.x,lo.y,lo.z)*v1.xyz,productD4(lo.x,t.y,lo.z)*v2.xyz,productD4(t.x,t.y,lo.z)*v3.xyz,productD4(lo.x,lo.y,t.z)*v4.xyz,productD4(t.x,lo.y,t.z)*v5.xyz,productD4(lo.x,t.y,t.z)*v6.xyz,productD4(t.x,t.y,t.z)*v7.xyz);return vec4f(value,1);}
fn snapDisplacement(q:vec3f)->vec3f{return round(q*65536.)/65536.;}
@compute @workgroup_size(64)fn backtraceIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;let count=min(atomicLoad(&transportControl[0]),capacity());if(work>=count){return;}let i=worklist[work];if(i>=nodeCount()||8u*i+7u>=arrayLength(&incidentLeaves)){atomicOr(&control[12],ERR_GRAPH);return;}departures[2u*i]=vec4f(0);let q=vec3f(nodePosition(i));let v0=packedVelocity(i);if(v0.w!=1.){atomicAdd(&control[14],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let mid=q-snapDisplacement(.5*p.numerics.x*v0.xyz/p.originCell.w);let boundedMid=clamp(mid,vec3f(0),vec3f(p.dims.xyz));let midpointNegative=vec3<bool>(boundedMid.x<q.x||(boundedMid.x==q.x&&q.x==f32(p.dims.x)),boundedMid.y<q.y||(boundedMid.y==q.y&&q.y==f32(p.dims.y)),boundedMid.z<q.z||(boundedMid.z==q.z&&q.z==f32(p.dims.z)));let midpointOctant=select(0u,1u,midpointNegative.x)|select(0u,2u,midpointNegative.y)|select(0u,4u,midpointNegative.z);let vm=sampleVelocity(boundedMid,incidentLeaves[8u*i+midpointOctant]);if(vm.w!=1.){atomicAdd(&control[15],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let departure=q-snapDisplacement(p.numerics.x*vm.xyz/p.originCell.w);let departureNegative=vec3<bool>(q.x>0.&&(departure.x<q.x||q.x==f32(p.dims.x)),q.y>0.&&(departure.y<q.y||q.y==f32(p.dims.y)),q.z>0.&&(departure.z<q.z||q.z==f32(p.dims.z)));let departureOctant=select(0u,1u,departureNegative.x)|select(0u,2u,departureNegative.y)|select(0u,4u,departureNegative.z);let leaf=incidentLeaves[8u*i+departureOctant];if(leaf==INVALID){atomicAdd(&control[15],1u);atomicOr(&control[12],ERR_SAMPLE);return;}departures[2u*i]=vec4f(departure,bitcast<f32>(leaf));}
`;

/** The +dt companion trace writes only the odd record for each compact node. */
export const octreeLosassoAdaptivePhiReverseBacktraceWGSL = octreeLosassoAdaptivePhiBacktraceWGSL
  .replace(/@compute @workgroup_size\(64\)fn backtraceIndependent[\s\S]*$/,
    `@compute @workgroup_size(64)fn reverseBacktraceIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;let count=min(atomicLoad(&transportControl[0]),capacity());if(work>=count){return;}let i=worklist[work];if(i>=nodeCount()||8u*i+7u>=arrayLength(&incidentLeaves)){atomicOr(&control[12],ERR_GRAPH);return;}departures[2u*i+1u]=vec4f(0);let q=vec3f(nodePosition(i));let v0=packedVelocity(i);if(v0.w!=1.){atomicAdd(&control[14],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let mid=q+snapDisplacement(.5*p.numerics.x*v0.xyz/p.originCell.w);let boundedMid=clamp(mid,vec3f(0),vec3f(p.dims.xyz));let midpointNegative=vec3<bool>(boundedMid.x<q.x||(boundedMid.x==q.x&&q.x==f32(p.dims.x)),boundedMid.y<q.y||(boundedMid.y==q.y&&q.y==f32(p.dims.y)),boundedMid.z<q.z||(boundedMid.z==q.z&&q.z==f32(p.dims.z)));let midpointOctant=select(0u,1u,midpointNegative.x)|select(0u,2u,midpointNegative.y)|select(0u,4u,midpointNegative.z);let vm=sampleVelocity(boundedMid,incidentLeaves[8u*i+midpointOctant]);if(vm.w!=1.){atomicAdd(&control[15],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let reverse=q+snapDisplacement(p.numerics.x*vm.xyz/p.originCell.w);let reverseNegative=vec3<bool>(q.x>0.&&(reverse.x<q.x||q.x==f32(p.dims.x)),q.y>0.&&(reverse.y<q.y||q.y==f32(p.dims.y)),q.z>0.&&(reverse.z<q.z||q.z==f32(p.dims.z)));let reverseOctant=select(0u,1u,reverseNegative.x)|select(0u,2u,reverseNegative.y)|select(0u,4u,reverseNegative.z);let leaf=incidentLeaves[8u*i+reverseOctant];if(leaf==INVALID){atomicAdd(&control[15],1u);atomicOr(&control[12],ERR_SAMPLE);return;}departures[2u*i+1u]=vec4f(reverse,bitcast<f32>(leaf));}`);

/** Compact scalar sampling at validated departures; no velocity field is bound. */
export const octreeLosassoAdaptivePhiTransportWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const ERR_GRAPH:u32=1u;const ERR_SAMPLE:u32=2u;const ERR_CONSTRAINT:u32=8u;
struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f}struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}struct Sample{value:f32,valid:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read>leaves:array<LeafRecord>;@group(0)@binding(3)var<storage,read>nodes:array<vec4u>;@group(0)@binding(4)var<storage,read>constraints:array<Constraint>;@group(0)@binding(5)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(6)var<storage,read_write>phi:array<vec2f>;@group(0)@binding(7)var<storage,read_write>transportControl:array<atomic<u32>>;@group(0)@binding(8)var<storage,read>worklist:array<u32>;@group(0)@binding(9)var<storage,read>departures:array<vec4f>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn quantizePhi(v:f32)->f32{return trunc(v*65536.)/65536.;}fn nodeCount()->u32{return min(graph[2],p.limits.x);}fn leafCount()->u32{return min(min(graph[1],p.limits.y),arrayLength(&leaves));}fn bank()->u32{return atomicLoad(&control[6])&1u;}fn capacity()->u32{return min(p.limits.x,arrayLength(&worklist)/2u);}fn nodePosition(i:u32)->vec3u{let item=nodes[i].x;let d=p.dims.xyz+vec3u(1);return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}fn before(a:f32,b:f32)->bool{return a<b||(a==b&&bitcast<u32>(a)<bitcast<u32>(b));}
fn orderedFour(v:vec4f)->f32{let p01=before(v.x,v.y);let a0=select(v.y,v.x,p01);let a1=select(v.x,v.y,p01);let p23=before(v.z,v.w);let a2=select(v.w,v.z,p23);let a3=select(v.z,v.w,p23);let p02=before(a0,a2);let b0=select(a2,a0,p02);let b2=select(a0,a2,p02);let p13=before(a1,a3);let b1=select(a3,a1,p13);let b3=select(a1,a3,p13);let p12=before(b1,b2);let c1=select(b2,b1,p12);let c2=select(b1,b2,p12);return(b0+c1)+(c2+b3);}fn constrained(i:u32,b:u32)->Sample{if(i>=nodeCount()){return Sample(0,0);}let c=constraints[i];let n=c.header.y;if(n==0u){let v=phi[i][b];return Sample(v,select(0u,1u,finite(v)));}if((n!=2u&&n!=4u)||c.header.z==0u||c.masters.x>=nodeCount()||c.masters.y>=nodeCount()){return Sample(0,0);}let t0=f32(c.numerators.x)*phi[c.masters.x][b];let t1=f32(c.numerators.y)*phi[c.masters.y][b];if(n==2u){let value=(t0+t1)/f32(c.header.z);return Sample(value,select(0u,1u,finite(value)));}if(c.masters.z>=nodeCount()||c.masters.w>=nodeCount()){return Sample(0,0);}let terms=vec4f(t0,t1,f32(c.numerators.z)*phi[c.masters.z][b],f32(c.numerators.w)*phi[c.masters.w][b]);let value=orderedFour(terms)/f32(c.header.z);return Sample(value,select(0u,1u,finite(value)));}
fn productD4(a:f32,b:f32,c:f32)->f32{var x=a;var y=b;var z=c;if(y<x){let swap=x;x=y;y=swap;}if(z<y){let swap=y;y=z;z=swap;}if(y<x){let swap=x;x=y;y=swap;}return(x*y)*z;}fn sumD4(a:f32,b:f32,c:f32,d:f32,e:f32,f:f32,g:f32,h:f32)->f32{return((a+f)+(b+e))+((c+h)+(d+g));}fn samplePhi(q:vec3f,b:u32,leaf:u32)->Sample{let bounded=clamp(q,vec3f(0),vec3f(p.dims.xyz));let exterior=length((q-bounded)*p.originCell.w);if(leaf>=leafCount()){return Sample(0,0);}let r=leaves[leaf];let s0=constrained(r.corners0.x,b);let s1=constrained(r.corners0.y,b);let s2=constrained(r.corners0.z,b);let s3=constrained(r.corners0.w,b);let s4=constrained(r.corners1.x,b);let s5=constrained(r.corners1.y,b);let s6=constrained(r.corners1.z,b);let s7=constrained(r.corners1.w,b);if(s0.valid*s1.valid*s2.valid*s3.valid*s4.valid*s5.valid*s6.valid*s7.valid!=1u){return Sample(0,0);}let t=clamp((bounded-vec3f(r.originSpan.xyz))/f32(r.originSpan.w),vec3f(0),vec3f(1));let lo=vec3f(1)-t;let value=sumD4(productD4(lo.x,lo.y,lo.z)*s0.value,productD4(t.x,lo.y,lo.z)*s1.value,productD4(lo.x,t.y,lo.z)*s2.value,productD4(t.x,t.y,lo.z)*s3.value,productD4(lo.x,lo.y,t.z)*s4.value,productD4(t.x,lo.y,t.z)*s5.value,productD4(lo.x,t.y,t.z)*s6.value,productD4(t.x,t.y,t.z)*s7.value)+exterior;return Sample(value,select(0u,1u,finite(value)));}
fn inflowPhi(world:vec3f)->f32{let direction=p.inflowVelocityStrength.xyz;if(dot(direction,direction)<.5||p.inflowPositionRadius.w<=0.){return 3.402823e38;}let relative=world-p.inflowPositionRadius.xyz;let axial=dot(relative,direction);let radial=relative-axial*direction;return max(sqrt(max(0.,dot(radial,radial)))-p.inflowPositionRadius.w,max(-axial,axial-2.*p.originCell.w));}
@compute @workgroup_size(64)fn transportIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;let count=min(atomicLoad(&transportControl[0]),capacity());if(work>=count){return;}let i=worklist[work];if(i>=nodeCount()){atomicOr(&control[12],ERR_GRAPH);return;}let departure=departures[2u*i];if(any(departure.xyz!=departure.xyz)){atomicAdd(&control[19],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let current=bank();let sampled=samplePhi(departure.xyz,current,bitcast<u32>(departure.w));if(sampled.valid==0u){atomicAdd(&control[19],1u);atomicOr(&control[12],ERR_SAMPLE);return;}var value=sampled.value;let q=vec3f(nodePosition(i));let source=inflowPhi(p.originCell.xyz+q*p.originCell.w);if(p.inflowVelocityStrength.w>0.&&finite(source)){value=min(value,max(source,-.5*p.originCell.w*clamp(p.inflowVelocityStrength.w,0,1)));}value=quantizePhi(value);let prior=phi[i][current];phi[i][1u-current]=value;atomicMax(&control[13],bitcast<u32>(abs(value-prior)));atomicAdd(&control[18],1u);}
`;

export const octreeLosassoAdaptivePhiPredictorSnapshotWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;const ERR_SAMPLE:u32=2u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflowPositionRadius:vec4f,inflowVelocityStrength:vec4f}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;@group(0)@binding(2)var<storage,read_write>control:array<atomic<u32>>;@group(0)@binding(3)var<storage,read>phi:array<vec2f>;@group(0)@binding(4)var<storage,read_write>predictorScratch:array<f32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn nodeCount()->u32{return min(graph[2],p.limits.x);}@compute @workgroup_size(64)fn snapshotProjectedPredictor(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=nodeCount()){return;}let value=phi[i][1u-(atomicLoad(&control[6])&1u)];if(!finite(value)){atomicOr(&control[12],ERR_SAMPLE);return;}predictorScratch[i]=value;}
`;

/** Monotone correction samples the immutable projected predictor in distanceA. */
export const octreeLosassoAdaptivePhiCorrectionWGSL = octreeLosassoAdaptivePhiTransportWGSL
  .replace(/@group\(0\)@binding\(9\)var<storage,read>departures:array<vec4f>;/,
    "@group(0)@binding(9)var<storage,read>departures:array<vec4f>;@group(0)@binding(10)var<storage,read>predictorScratch:array<f32>;")
  .replace(/@compute @workgroup_size\(64\)fn transportIndependent[\s\S]*$/,
  `fn cornerSlot(r:LeafRecord,c:u32)->u32{if(c<4u){return r.corners0[c];}return r.corners1[c-4u];}
fn donorBounds(leaf:u32,b:u32)->vec3f{if(leaf>=leafCount()){return vec3f(0);}let r=leaves[leaf];var minimumValue=3.402823e38;var maximumValue=-3.402823e38;for(var c=0u;c<8u;c+=1u){let slot=cornerSlot(r,c);if(slot>=nodeCount()){return vec3f(0);}let value=phi[slot][b];if(!finite(value)){return vec3f(0);}minimumValue=min(minimumValue,value);maximumValue=max(maximumValue,value);}return vec3f(minimumValue,maximumValue,1);}
fn scratchValue(i:u32)->Sample{if(i>=nodeCount()){return Sample(0,0);}let value=predictorScratch[i];return Sample(value,select(0u,1u,finite(value)));}
fn samplePredictor(q:vec3f,leaf:u32)->Sample{let bounded=clamp(q,vec3f(0),vec3f(p.dims.xyz));let exterior=length((q-bounded)*p.originCell.w);if(leaf>=leafCount()){return Sample(0,0);}let r=leaves[leaf];let s0=scratchValue(r.corners0.x);let s1=scratchValue(r.corners0.y);let s2=scratchValue(r.corners0.z);let s3=scratchValue(r.corners0.w);let s4=scratchValue(r.corners1.x);let s5=scratchValue(r.corners1.y);let s6=scratchValue(r.corners1.z);let s7=scratchValue(r.corners1.w);if(s0.valid*s1.valid*s2.valid*s3.valid*s4.valid*s5.valid*s6.valid*s7.valid!=1u){return Sample(0,0);}let t=clamp((bounded-vec3f(r.originSpan.xyz))/f32(r.originSpan.w),vec3f(0),vec3f(1));let lo=vec3f(1)-t;let value=sumD4(productD4(lo.x,lo.y,lo.z)*s0.value,productD4(t.x,lo.y,lo.z)*s1.value,productD4(lo.x,t.y,lo.z)*s2.value,productD4(t.x,t.y,lo.z)*s3.value,productD4(lo.x,lo.y,t.z)*s4.value,productD4(t.x,lo.y,t.z)*s5.value,productD4(lo.x,t.y,t.z)*s6.value,productD4(t.x,t.y,t.z)*s7.value)+exterior;return Sample(value,select(0u,1u,finite(value)));}
@compute @workgroup_size(64)fn correctTransportIndependent(@builtin(global_invocation_id)gid:vec3u){let work=gid.x;let count=min(atomicLoad(&transportControl[0]),capacity());if(work>=count){return;}let i=worklist[work];if(i>=nodeCount()){atomicOr(&control[12],ERR_GRAPH);return;}let current=bank();let targetBank=1u-current;let backward=departures[2u*i];let reverse=departures[2u*i+1u];let backwardLeaf=bitcast<u32>(backward.w);let reversed=samplePredictor(reverse.xyz,bitcast<u32>(reverse.w));let bounds=donorBounds(backwardLeaf,current);if(reversed.valid==0u||bounds.z!=1.){atomicAdd(&control[19],1u);atomicOr(&control[12],ERR_SAMPLE);return;}let predictor=predictorScratch[i];let prior=phi[i][current];var corrected=clamp(predictor+.5*(prior-reversed.value),bounds.x,bounds.y);let q=vec3f(nodePosition(i));let source=inflowPhi(p.originCell.xyz+q*p.originCell.w);if(p.inflowVelocityStrength.w>0.&&finite(source)){corrected=predictor;}corrected=quantizePhi(corrected);phi[i][targetBank]=corrected;atomicMax(&control[13],bitcast<u32>(abs(corrected-prior)));}`);

export const octreeLosassoAdaptivePhiHandoffWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const ERR_GRAPH:u32=1u;const ERR_SAMPLE:u32=2u;const ERR_CONSTRAINT:u32=8u;
struct Params { dims:vec4u, originCell:vec4f, limits:vec4u, numerics:vec4f, policy:vec4u, volume:vec4f }
struct Constraint { header:vec4u,masters:vec4u,numerators:vec4u }
struct LeafRecord { originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u }
struct LeafLookup { cell:u32,span:u32,slot:u32,flags:u32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(2)var<storage,read> acceptedGraph:array<u32>;
@group(0)@binding(3)var<storage,read> acceptedLeaves:array<LeafRecord>;@group(0)@binding(4)var<storage,read> acceptedNodeDirectory:array<vec2u>;
@group(0)@binding(5)var<storage,read> acceptedLeafDirectory:array<LeafLookup>;@group(0)@binding(6)var<storage,read> acceptedPhi:array<vec2f>;
@group(0)@binding(7)var<storage,read> candidateGraph:array<u32>;@group(0)@binding(8)var<storage,read> candidateNodes:array<vec4u>;
@group(0)@binding(9)var<storage,read> candidateConstraints:array<Constraint>;@group(0)@binding(10)var<storage,read_write> candidatePhi:array<vec2f>;
@group(0)@binding(11)var<storage,read_write> state:array<atomic<u32>>;
@group(0)@binding(17)var<storage,read_write> receipt:array<atomic<u32>>;
fn quantizePhi(v:f32)->f32{return round(v*65536.)/65536.;}fn exact(item:u32)->u32{var lo=0u;var hi=atomicLoad(&state[4]);while(lo<hi){let mid=lo+(hi-lo)/2u;if(acceptedNodeDirectory[mid].x<item){lo=mid+1u;}else{hi=mid;}}if(lo<atomicLoad(&state[4])&&acceptedNodeDirectory[lo].x==item){return acceptedNodeDirectory[lo].y;}return INVALID;}
fn cell(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}fn logSpan(v:u32)->u32{var n=0u;var x=v;while(x>1u){x>>=1u;n+=1u;}return n;}
fn mortonPart(v:u32)->u32{var x=v&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}fn morton(q:vec3u)->u32{return mortonPart(q.x)|(mortonPart(q.y)<<1u)|(mortonPart(q.z)<<2u);}
fn less(ca:u32,sa:u32,cb:u32,sb:u32)->bool{return ca<cb||(ca==cb&&sa<sb);}
fn leafAtHalfOpen(q:vec3f)->u32{var span=1u;loop{let o=vec3u(floor(q/vec3f(f32(span))))*span;let wanted=cell(o);var lo=0u;var hi=atomicLoad(&state[5]);while(lo<hi){let mid=lo+(hi-lo)/2u;let e=acceptedLeafDirectory[mid];if(less(e.cell,e.span,wanted,span)){lo=mid+1u;}else{hi=mid;}}if(lo<atomicLoad(&state[5])){let e=acceptedLeafDirectory[lo];if(e.cell==wanted&&e.span==span&&e.flags!=0u){return e.slot;}}if(span>=p.dims.w){break;}span*=2u;}return INVALID;}
fn leafAt(q:vec3f)->u32{let onPlane=abs(q-round(q))<=vec3f(1e-4);for(var mask=0u;mask<8u;mask+=1u){var probe=q;var eligible=true;for(var axis=0u;axis<3u;axis+=1u){if((mask&(1u<<axis))!=0u){if(!onPlane[axis]||q[axis]<=0.){eligible=false;}else{probe[axis]-=1e-5;}}}if(eligible){let leaf=leafAtHalfOpen(min(probe,vec3f(p.dims.xyz)-vec3f(1e-5)));if(leaf!=INVALID){return leaf;}}}return INVALID;}
fn corner(r:LeafRecord,c:u32)->u32{if(c<4u){return r.corners0[c];}return r.corners1[c-4u];}
fn orderedBefore(a:f32,b:f32)->bool{return a<b||(a==b&&bitcast<u32>(a)<bitcast<u32>(b));}
fn orderedSum4(input:array<f32,4>)->f32{var terms=input;for(var x=0u;x<4u;x+=1u){for(var y=x+1u;y<4u;y+=1u){if(orderedBefore(terms[y],terms[x])){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}return(terms[0]+terms[1])+(terms[2]+terms[3]);}
fn orderedConstraintSum(terms:array<f32,4>,count:u32)->f32{if(count==2u){return terms[0]+terms[1];}return orderedSum4(terms);}
fn orderedSum8(input:array<f32,8>)->f32{var terms=input;for(var x=0u;x<8u;x+=1u){for(var y=x+1u;y<8u;y+=1u){if(orderedBefore(terms[y],terms[x])){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}let a0=terms[0]+terms[1];let a1=terms[2]+terms[3];let a2=terms[4]+terms[5];let a3=terms[6]+terms[7];return(a0+a1)+(a2+a3);}
fn sample(q:vec3f)->f32{let leaf=leafAt(min(clamp(q,vec3f(0),vec3f(p.dims.xyz)),vec3f(p.dims.xyz)-vec3f(.00001)));if(leaf==INVALID){return 3.402823e38;}let r=acceptedLeaves[leaf];let t=clamp((q-vec3f(r.originSpan.xyz))/f32(r.originSpan.w),vec3f(0),vec3f(1));let bank=atomicLoad(&state[6])&1u;var terms:array<f32,8>;for(var c=0u;c<8u;c+=1u){let value=acceptedPhi[corner(r,c)][bank];let w=select(1.-t.x,t.x,(c&1u)!=0u)*select(1.-t.y,t.y,(c&2u)!=0u)*select(1.-t.z,t.z,(c&4u)!=0u);terms[c]=w*value;}return orderedSum8(terms);}
fn nodePosition(item:u32)->vec3u{let d=p.dims.xyz+vec3u(1u);return vec3u(item%d.x,(item/d.x)%d.y,item/(d.x*d.y));}
@compute @workgroup_size(1)fn prepareCandidateHandoff(){atomicStore(&state[14],0u);for(var i=18u;i<20u;i+=1u){atomicStore(&state[i],0u);}atomicStore(&state[8],candidateGraph[0]);atomicStore(&state[10],candidateGraph[2]);atomicStore(&state[11],0u);
 if(atomicLoad(&state[7])!=1u||acceptedGraph[0]==0u||acceptedGraph[3]!=acceptedGraph[0]||candidateGraph[0]==0u||candidateGraph[3]!=candidateGraph[0]||candidateGraph[2]>p.limits.x){atomicOr(&state[12],ERR_GRAPH);}}
@compute @workgroup_size(64)fn handoffCandidate(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=atomicLoad(&state[10])||i>=p.limits.x){return;}let item=candidateNodes[i].x;let retained=exact(item);var value=0.;if(retained!=INVALID){let bank=atomicLoad(&state[6])&1u;value=acceptedPhi[retained][bank];atomicAdd(&state[18],1u);}else{value=quantizePhi(sample(vec3f(nodePosition(item))));atomicAdd(&state[19],1u);}
 if(value!=value||abs(value)>=3.402823e38){atomicOr(&state[12],ERR_SAMPLE);return;}candidatePhi[i]=vec2f(value);}
@compute @workgroup_size(64)fn projectCandidate(@builtin(global_invocation_id)gid:vec3u){let i=gid.x;if(i>=candidateGraph[2]||i>=p.limits.x){return;}let c=candidateConstraints[i];let count=c.header.y;if(count==0u){return;}if((count!=2u&&count!=4u)||c.header.z==0u){atomicOr(&state[12],ERR_CONSTRAINT);return;}var terms:array<f32,4>;var sum=0u;for(var j=0u;j<count;j+=1u){if(c.masters[j]>=candidateGraph[2]){atomicOr(&state[12],ERR_CONSTRAINT);return;}terms[j]=f32(c.numerators[j])*candidatePhi[c.masters[j]].x;sum+=c.numerators[j];}if(sum!=c.header.z){atomicOr(&state[12],ERR_CONSTRAINT);return;}let value=orderedConstraintSum(terms,count)/f32(c.header.z);candidatePhi[i]=vec2f(value);atomicAdd(&state[14],1u);}
@compute @workgroup_size(1)fn finalizeCandidateHandoff(){let valid=atomicLoad(&state[12])==0u&&atomicLoad(&state[18])+atomicLoad(&state[19])==candidateGraph[2];atomicStore(&state[11],select(0u,1u,valid));atomicStore(&receipt[16],atomicLoad(&state[18]));atomicStore(&receipt[17],atomicLoad(&state[19]));atomicStore(&receipt[18],atomicLoad(&state[14]));atomicStore(&receipt[19],select(0u,1u,valid));}
`;

export const octreeLosassoAdaptivePhiCommitWGSL = /* wgsl */ `
const ERR_GRAPH:u32=1u;
const ADAPTIVE_MASS_MAGIC:u32=0x414d4153u;
struct Params { dims:vec4u, originCell:vec4f, limits:vec4u, numerics:vec4f, policy:vec4u, volume:vec4f, inflowPositionRadius:vec4f, inflowVelocityStrength:vec4f }
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write> candidateGraph:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read> joint:array<u32>;@group(0)@binding(5)var<storage,read_write> state:array<atomic<u32>>;
@group(0)@binding(6)var<storage,read_write> receipts:array<atomic<u32>>;
@compute @workgroup_size(1)fn commitCandidate(){let epoch=atomicLoad(&candidateGraph[0]);let ready=joint[0]==ADAPTIVE_MASS_MAGIC&&joint[1]==epoch&&joint[7]==1u&&joint[12]==0u&&epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u;
 if(!ready){atomicStore(&receipts[20],0u);return;}let generation=max(max(atomicLoad(&state[2]),atomicLoad(&candidateGraph[5])),epoch)+1u;atomicStore(&candidateGraph[5],generation);atomicStore(&state[9],generation);atomicStore(&receipts[20],1u);}
@compute @workgroup_size(1)fn stampCandidateBootstrap(){let epoch=atomicLoad(&candidateGraph[0]);let ready=epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u&&atomicLoad(&state[7])==1u&&atomicLoad(&state[1])==epoch;
 if(ready){atomicStore(&candidateGraph[5],epoch);atomicStore(&state[2],epoch);atomicStore(&receipts[20],1u);}else{atomicStore(&receipts[20],0u);}}
@compute @workgroup_size(1)fn syncAcceptedCommit(){let epoch=atomicLoad(&candidateGraph[0]);let priorEpoch=atomicLoad(&state[1]);let priorBank=atomicLoad(&state[6]);let generation=atomicLoad(&candidateGraph[5]);let valid=epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u&&generation!=0u&&atomicLoad(&candidateGraph[6])==generation;let reused=valid&&epoch==priorEpoch;
 atomicStore(&state[1],select(0u,epoch,valid));atomicStore(&state[2],select(0u,generation,valid));atomicStore(&state[3],select(0u,atomicLoad(&candidateGraph[6]),valid));atomicStore(&state[4],select(0u,atomicLoad(&candidateGraph[2]),valid));atomicStore(&state[5],select(0u,atomicLoad(&candidateGraph[1]),valid));atomicStore(&state[6],select(0u,priorBank,reused));atomicStore(&state[7],select(0u,1u,valid));atomicStore(&state[12],select(ERR_GRAPH,0u,valid));atomicStore(&receipts[20],select(0u,1u,valid));}
@compute @workgroup_size(1)fn stampCandidateRepair(){let epoch=atomicLoad(&candidateGraph[0]);let ready=epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u&&atomicLoad(&state[11])==1u&&atomicLoad(&state[12])==0u&&atomicLoad(&state[13])==1u&&bitcast<f32>(atomicLoad(&receipts[24]))<=p.numerics.z;
 if(ready){let generation=atomicLoad(&state[2]);atomicStore(&candidateGraph[5],generation);atomicStore(&state[9],generation);atomicStore(&receipts[21],1u);}else{atomicStore(&receipts[21],0u);}}
@compute @workgroup_size(1)fn stampAcceptedAdvance(){let epoch=atomicLoad(&candidateGraph[0]);let ready=epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u&&atomicLoad(&state[7])==1u&&atomicLoad(&state[1])==epoch&&atomicLoad(&state[12])==0u;
 if(ready){atomicStore(&candidateGraph[5],atomicLoad(&state[2]));atomicStore(&receipts[22],1u);}else{atomicStore(&receipts[22],0u);}}
// Surface mass is the authority on accepted steps.  Both phi components have
// already been materialized from that mass when this singleton runs, so it
// only publishes a fresh scalar clock; it never transports or redistances phi.
// Binding 2 is the adaptive-mass control block for this entry point.
@compute @workgroup_size(1)fn publishExternalAccepted(){let epoch=atomicLoad(&candidateGraph[0]);let targeted=joint[0]==ADAPTIVE_MASS_MAGIC&&joint[1]==epoch;let ready=epoch!=0u&&atomicLoad(&candidateGraph[3])==epoch&&atomicLoad(&candidateGraph[4])==0u&&targeted&&joint[7]==1u&&joint[12]==0u;
 if(ready){let generation=max(atomicLoad(&state[2]),atomicLoad(&candidateGraph[5]))+1u;atomicStore(&candidateGraph[5],generation);atomicStore(&state[1],epoch);atomicStore(&state[2],generation);atomicStore(&state[4],atomicLoad(&candidateGraph[2]));atomicStore(&state[5],atomicLoad(&candidateGraph[1]));atomicStore(&state[7],1u);atomicStore(&state[12],0u);atomicStore(&receipts[1],0u);atomicStore(&receipts[2],0u);atomicStore(&receipts[3],0u);atomicStore(&receipts[4],0u);atomicStore(&receipts[5],0u);atomicStore(&receipts[7],0u);atomicStore(&receipts[12],0u);atomicStore(&receipts[14],1u);atomicStore(&receipts[15],atomicLoad(&candidateGraph[1]));atomicStore(&receipts[22],1u);atomicStore(&receipts[23],generation);atomicStore(&receipts[32],0u);atomicStore(&receipts[33],atomicLoad(&candidateGraph[2]));atomicStore(&receipts[34],0u);atomicStore(&receipts[35],0u);atomicStore(&receipts[54],0u);}else if(targeted){let reason=select(0u,1u,epoch==0u)|select(0u,2u,atomicLoad(&candidateGraph[3])!=epoch)|select(0u,4u,atomicLoad(&candidateGraph[4])!=0u)|select(0u,8u,joint[7]!=1u)|select(0u,16u,joint[12]!=0u)|(joint[12]<<8u);atomicStore(&receipts[54],reason);atomicStore(&receipts[14],0u);atomicStore(&receipts[22],0u);}}
`;

export const octreeLosassoAdaptivePhiGhostWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=4u;
const FACE_SOLID_NEUMANN:u32=0x08000000u;const FACE_INTERFACE_NEARBY:u32=0x10000000u;
const FACE_SEPARATED:u32=0x20000000u;const FACE_CLOSED_BOUNDARY:u32=0x40000000u;
const FACE_ROW_ON_POSITIVE_SIDE:u32=0x80000000u;
struct Params { dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f }
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}
struct LeafLookup{cell:u32,span:u32,slot:u32,flags:u32}struct Sample{value:f32,valid:u32}
@group(0)@binding(0)var<uniform> p:Params;@group(0)@binding(1)var<storage,read> graph:array<u32>;@group(0)@binding(2)var<storage,read> leaves:array<LeafRecord>;
@group(0)@binding(3)var<storage,read> rowPhi:array<vec4u>;@group(0)@binding(4)var<storage,read> faceControl:array<u32>;@group(0)@binding(5)var<storage,read_write> faces:array<Face>;
@group(0)@binding(6)var<storage,read_write> ghosts:array<vec4u>;
@group(0)@binding(7)var<storage,read> pressureRowToGraphLeaf:array<u32>;
@group(0)@binding(8)var<storage,read> leafDirectory:array<LeafLookup>;
@group(0)@binding(9)var<storage,read> phi:array<vec2f>;
@group(0)@binding(10)var<storage,read> state:array<u32>;
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn leafCount()->u32{return min(min(graph[1],p.limits.y),arrayLength(&leaves));}fn nodeCount()->u32{return min(graph[2],min(p.limits.x,arrayLength(&phi)));}
fn cellItem(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}fn less(cellA:u32,spanA:u32,cellB:u32,spanB:u32)->bool{return cellA<cellB||(cellA==cellB&&spanA<spanB);}
fn findLeafHalfOpen(q:vec3f)->u32{if(any(q<vec3f(0))||any(q>=vec3f(p.dims.xyz))){return INVALID;}var span=1u;loop{let origin=vec3u(floor(q/f32(span)))*span;let cell=cellItem(origin);var lo=0u;var hi=min(leafCount(),arrayLength(&leafDirectory));while(lo<hi){let mid=lo+(hi-lo)/2u;let e=leafDirectory[mid];if(less(e.cell,e.span,cell,span)){lo=mid+1u;}else{hi=mid;}}if(lo<min(leafCount(),arrayLength(&leafDirectory))){let e=leafDirectory[lo];if(e.cell==cell&&e.span==span&&e.flags!=0u){return e.slot;}}if(span>=p.dims.w){break;}span*=2u;}return INVALID;}
fn cornerSlot(r:LeafRecord,c:u32)->u32{if(c<4u){return r.corners0[c];}return r.corners1[c-4u];}
fn symmetricSum8(terms:array<f32,8>)->f32{return((terms[0]+terms[1])+(terms[2]+terms[3]))+((terms[4]+terms[5])+(terms[6]+terms[7]));}
fn samplePhi(q:vec3f)->Sample{if(q.y>f32(p.dims.y)&&p.policy.x!=0u){let value=p.numerics.w+(q.y-f32(p.dims.y))*p.originCell.w;return Sample(value,select(0u,1u,finite(value)));}let bounded=clamp(q,vec3f(0),vec3f(p.dims.xyz));let exterior=length((q-bounded)*p.originCell.w);let ownerPoint=min(bounded,vec3f(p.dims.xyz)-vec3f(.00001));let leaf=findLeafHalfOpen(ownerPoint);if(leaf==INVALID||leaf>=leafCount()){return Sample(0.,0u);}let r=leaves[leaf];let t=clamp((bounded-vec3f(r.originSpan.xyz))/f32(r.originSpan.w),vec3f(0),vec3f(1));let bank=state[6]&1u;var terms:array<f32,8>;for(var c=0u;c<8u;c+=1u){let slot=cornerSlot(r,c);if(slot>=nodeCount()){return Sample(0.,0u);}let v=phi[slot][bank];if(!finite(v)){return Sample(0.,0u);}let w=select(1.-t.x,t.x,(c&1u)!=0u)*select(1.-t.y,t.y,(c&2u)!=0u)*select(1.-t.z,t.z,(c&4u)!=0u);terms[c]=w*v;}let value=symmetricSum8(terms)+exterior;return Sample(value,select(0u,1u,finite(value)));}
fn leafCentre(leaf:u32)->vec3f{let r=leaves[leaf].originSpan;return vec3f(r.xyz)+vec3f(.5*f32(r.w));}
@compute @workgroup_size(64)fn deriveGhosts(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=faceControl[2]||i>=arrayLength(&faces)||i>=arrayLength(&ghosts)||i>=p.policy.z){return;}let rows=min(min(graph[28],arrayLength(&pressureRowToGraphLeaf)),arrayLength(&rowPhi));var f=faces[i];if(f.axis>2u){ghosts[i]=vec4u(0);return;}
 f.reserved&=~FACE_INTERFACE_NEARBY;let negativeNearby=f.negativeRow<rows&&(rowPhi[f.negativeRow].w&INTERFACE)!=0u;let positiveNearby=f.positiveRow<rows&&(rowPhi[f.positiveRow].w&INTERFACE)!=0u;if(negativeNearby||positiveNearby){f.reserved|=FACE_INTERFACE_NEARBY;}
 var negativeLeaf=INVALID;var positiveLeaf=INVALID;if(f.negativeRow<rows){negativeLeaf=pressureRowToGraphLeaf[f.negativeRow];}if(f.positiveRow<rows){positiveLeaf=pressureRowToGraphLeaf[f.positiveRow];}if((negativeLeaf!=INVALID&&negativeLeaf>=leafCount())||(positiveLeaf!=INVALID&&positiveLeaf>=leafCount())||(negativeLeaf==INVALID&&positiveLeaf==INVALID)){faces[i]=f;ghosts[i]=vec4u(0);return;}
 var negativePoint=vec3f(0);var positivePoint=vec3f(0);var negativeSample=Sample(0.,0u);var positiveSample=Sample(0.,0u);if(negativeLeaf!=INVALID){negativePoint=leafCentre(negativeLeaf);negativeSample=samplePhi(negativePoint);}if(positiveLeaf!=INVALID){positivePoint=leafCentre(positiveLeaf);positiveSample=samplePhi(positivePoint);}
 if(negativeLeaf==INVALID){negativePoint=positivePoint;negativePoint[f.axis]-=f32(leaves[positiveLeaf].originSpan.w);negativeSample=samplePhi(negativePoint);}if(positiveLeaf==INVALID){let wetSpan=f32(leaves[negativeLeaf].originSpan.w);let direction=select(1.,-1.,(f.reserved&FACE_ROW_ON_POSITIVE_SIDE)!=0u);var probe=negativePoint;probe[f.axis]+=direction*(.5*wetSpan+.0001);let airLeaf=findLeafHalfOpen(probe);if(airLeaf!=INVALID&&airLeaf<leafCount()){positivePoint=leafCentre(airLeaf);}else{positivePoint=negativePoint;positivePoint[f.axis]+=direction*wetSpan;}positiveSample=samplePhi(positivePoint);}
 let dual=max(abs(positivePoint[f.axis]-negativePoint[f.axis])*p.originCell.w,1e-9);let closed=(f.reserved&FACE_CLOSED_BOUNDARY)!=0u;let solid=(f.reserved&FACE_SOLID_NEUMANN)!=0u;let separated=closed&&(f.reserved&FACE_SEPARATED)!=0u;let geometricOpen=clamp(f.openFraction,0.,1.);f.inverseDistance=1./dual;f.openFraction=geometricOpen;if(solid||(closed&&!separated)){f.openFraction=0.;}else if(separated){f.openFraction=1.;}
 if(solid||(positiveLeaf==INVALID&&closed&&!separated)){faces[i]=f;ghosts[i]=vec4u(0u,bitcast<u32>(1.),0u,2u);return;}if(negativeSample.valid==0u||positiveSample.valid==0u){faces[i]=f;ghosts[i]=vec4u(0);return;}var a=0.;var b=0.;var wetPoint=vec3f(0);var airPoint=vec3f(0);if(negativeSample.value<=0.&&positiveSample.value>0.){a=negativeSample.value;b=positiveSample.value;wetPoint=negativePoint;airPoint=positivePoint;}else if(positiveSample.value<=0.&&negativeSample.value>0.){a=positiveSample.value;b=negativeSample.value;wetPoint=positivePoint;airPoint=negativePoint;}else{faces[i]=f;ghosts[i]=vec4u(0);return;}
 let physicalDual=max(abs(airPoint[f.axis]-wetPoint[f.axis])*p.originCell.w,1e-9);let theta=clamp(-a/max(b-a,1e-20),.01,1.);let ghostDistance=max(theta*physicalDual,1e-9);f.inverseDistance=1./ghostDistance;faces[i]=f;ghosts[i]=vec4u(bitcast<u32>(ghostDistance),bitcast<u32>(theta),bitcast<u32>(b),select(VALID,3u,closed&&separated));}
`;

/** One-binding canonical adaptive row evidence for topology consumers. */
export const octreeLosassoAdaptivePhiEvidenceWGSL = /* wgsl */ `
const MAGIC:u32=0x4c504849u;const ADAPTIVE_CORNERS:u32=0x10000000u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f}
struct LeafRecord{originSpan:vec4u,info:vec4u,corners0:vec4u,corners1:vec4u}
struct LeafLookup{cell:u32,span:u32,slot:u32,flags:u32}
struct Constraint{header:vec4u,masters:vec4u,numerators:vec4u}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>graph:array<u32>;
@group(0)@binding(2)var<storage,read>leaves:array<LeafRecord>;@group(0)@binding(3)var<storage,read>leafDirectory:array<LeafLookup>;
@group(0)@binding(4)var<storage,read_write>arena:array<atomic<u32>>;
@group(0)@binding(5)var<storage,read_write>renderer:array<u32>;
@group(0)@binding(6)var<storage,read>phi:array<vec2f>;@group(0)@binding(7)var<storage,read>state:array<u32>;
@group(0)@binding(8)var<storage,read_write>receipts:array<atomic<u32>>;
@group(0)@binding(9)var<storage,read>constraints:array<Constraint>;
@group(0)@binding(10)var<storage,read>surfaceMass:array<f32>;
fn hash(cell:u32,span:u32)->u32{return ((cell*0x9e3779b1u)^span)*0x85ebca6bu;}
fn rendererCorner(leaf:LeafRecord,corner:u32)->u32{if(corner<4u){return leaf.corners0[corner];}return leaf.corners1[corner-4u];}
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}fn canonicalPublishedZero(v:f32)->f32{let bits=bitcast<u32>(v);return bitcast<f32>(select(bits,0u,(bits&0x7fffffffu)==0u));}
fn publishedNodeValue(slot:u32,bank:u32)->f32{let invalid=3.402823e38;if(slot>=graph[2]||slot>=arrayLength(&phi)||slot>=arrayLength(&constraints)){return invalid;}let c=constraints[slot];let count=c.header.y;if(count==0u){return phi[slot][bank];}if((count!=2u&&count!=4u)||c.header.z==0u){return invalid;}var terms:array<f32,4>;var weight=0u;for(var term=0u;term<count;term+=1u){let master=c.masters[term];if(master>=graph[2]||master>=arrayLength(&phi)){return invalid;}terms[term]=f32(c.numerators[term])*phi[master][bank];weight+=c.numerators[term];}if(weight!=c.header.z){return invalid;}if(count==2u){return(terms[0]+terms[1])/f32(c.header.z);}for(var x=0u;x<4u;x+=1u){for(var y=x+1u;y<4u;y+=1u){if(terms[y]<terms[x]){let swap=terms[x];terms[x]=terms[y];terms[y]=swap;}}}return((terms[0]+terms[1])+(terms[2]+terms[3]))/f32(c.header.z);}
fn evidenceCell(q:vec3u)->u32{return q.x+p.dims.x*(q.y+p.dims.y*q.z);}
fn evidenceLowerBound(cell:u32)->u32{var low=0u;var high=min(graph[1],p.limits.y);while(low<high){let middle=low+(high-low)/2u;if(leafDirectory[middle].cell<cell){low=middle+1u;}else{high=middle;}}return low;}
fn evidenceHashRow(cell:u32,span:u32)->u32{if(atomicLoad(&arena[19])!=graph[0]||graph[0]==0u){return 0xffffffffu;}let cap=p.policy.w;let wanted=hash(cell,span);for(var probe=0u;probe<32u;probe+=1u){let at=20u+8u*p.limits.y+4u*((wanted+probe)&(cap-1u));if(at+3u>=arrayLength(&arena)){return 0xffffffffu;}let key=atomicLoad(&arena[at]);if(key==0u){return 0xffffffffu;}if(key==cell+1u&&atomicLoad(&arena[at+1u])==span&&atomicLoad(&arena[at+3u])==wanted){let encoded=atomicLoad(&arena[at+2u]);if(encoded==0u){return 0xffffffffu;}let row=encoded-1u;if(row<min(graph[1],p.limits.y)){let d=leafDirectory[row];if(d.cell==cell&&d.span==span&&d.slot<graph[1]&&d.flags!=0u){return row;}}return 0xffffffffu;}}return 0xffffffffu;}
fn evidenceLocate(cell:vec3u)->u32{let cached=atomicLoad(&arena[19])==graph[0]&&graph[0]!=0u;var span=1u;loop{let origin=(cell/span)*span;let wanted=evidenceCell(origin);if(cached){let row=evidenceHashRow(wanted,span);if(row!=0xffffffffu){return leafDirectory[row].slot;}}else{let at=evidenceLowerBound(wanted);if(at<min(graph[1],p.limits.y)){let d=leafDirectory[at];if(d.cell==wanted&&d.span==span&&d.slot<graph[1]&&d.flags!=0u){return d.slot;}}}if(span>=p.dims.w){break;}span*=2u;}return 0xffffffffu;}
fn evidenceRho(slot:u32)->f32{if(slot>=graph[1]||slot>=arrayLength(&leaves)||slot>=arrayLength(&surfaceMass)){return -1.;}let leaf=leaves[slot];let width=f32(leaf.originSpan.w)*p.originCell.w;let volume=width*width*width;let rho=surfaceMass[slot]/max(volume,1e-30);return select(-1.,rho,finite(rho)&&rho>=0.);}
// Chentanez--Mueller mass is an integral leaf field. Its physical interface is
// rho=.5, so retain the finest topology only where the local 26-neighbour shell
// brackets that value (or one shell member lies in the predictive [.25,.75]
// band). This is not the retired 0<rho<1 rule: low-density donor tails whose
// neighbourhood stays on the air side remain ordinary coarsening evidence.
fn rhoHalfInterface(leaf:LeafRecord,slot:u32)->bool{let own=evidenceRho(slot);if(own<0.){return false;}var minimum=own;var maximum=own;var nearHalf=abs(own-.5)<=.25;let span=leaf.originSpan.w;let centre=vec3i(leaf.originSpan.xyz+vec3u((span-1u)/2u));for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){if(x==0&&y==0&&z==0){continue;}let direction=vec3i(x,y,z);var q=centre;for(var axis=0u;axis<3u;axis+=1u){if(direction[axis]<0){q[axis]=i32(leaf.originSpan[axis])-1;}else if(direction[axis]>0){q[axis]=i32(leaf.originSpan[axis]+span);}}if(any(q<vec3i(0))||any(q>=vec3i(p.dims.xyz))){continue;}let neighbor=evidenceLocate(vec3u(q));if(neighbor==0xffffffffu){continue;}let rho=evidenceRho(neighbor);if(rho>=0.){minimum=min(minimum,rho);maximum=max(maximum,rho);nearHalf=nearHalf||abs(rho-.5)<=.25;}}}}return(minimum<=.5&&maximum>=.5)||nearHalf;}
fn evidenceValid(rows:u32,cap:u32)->bool{var rendererRowCapacity=0u;let rendererWords=arrayLength(&renderer);if(rendererWords>=8u){rendererRowCapacity=(rendererWords-8u)/16u;}return graph[0]!=0u&&graph[3]==graph[0]&&graph[4]==0u&&rows<=arrayLength(&leafDirectory)&&cap!=0u&&(cap&(cap-1u))==0u&&rows<=rendererRowCapacity;}
// Word 19 is the topology-only hash epoch. The scalar generation in words
// 12/14 changes every accepted step, but the row keys do not. Gate reuse in a
// singleton before the parallel refresh so no workgroup races header mutation.
fn topologyHashMatches(rows:u32,cap:u32)->bool{let epoch=graph[0];return atomicLoad(&arena[0])==MAGIC&&epoch!=0u&&atomicLoad(&arena[1])==epoch&&atomicLoad(&arena[2])==rows&&atomicLoad(&arena[4])==p.dims.w&&atomicLoad(&arena[5])==p.dims.x&&atomicLoad(&arena[6])==p.dims.y&&atomicLoad(&arena[7])==p.dims.z&&atomicLoad(&arena[9])==20u&&atomicLoad(&arena[10])==20u+8u*p.limits.y&&atomicLoad(&arena[11])==cap&&atomicLoad(&arena[13])==0u&&atomicLoad(&arena[18])==20u+8u*p.limits.y+4u*cap&&atomicLoad(&arena[19])==epoch;}
@compute @workgroup_size(1)fn prepareTopologyEvidenceEpoch(){let rows=min(graph[1],p.limits.y);let cap=p.policy.w;let valid=evidenceValid(rows,cap);let reuse=valid&&topologyHashMatches(rows,cap);atomicStore(&receipts[15],0u);atomicStore(&arena[0],0u);renderer[0]=0u;atomicStore(&arena[13],select(1u,0u,valid));atomicStore(&arena[19],select(0u,graph[0],reuse));if(!valid){return;}atomicStore(&arena[1],graph[0]);atomicStore(&arena[2],rows);atomicStore(&arena[3],0u);atomicStore(&arena[4],p.dims.w);atomicStore(&arena[5],p.dims.x);atomicStore(&arena[6],p.dims.y);atomicStore(&arena[7],p.dims.z);atomicStore(&arena[8],bitcast<u32>(p.originCell.w));atomicStore(&arena[9],20u);atomicStore(&arena[10],20u+8u*p.limits.y);atomicStore(&arena[11],cap);atomicStore(&arena[12],state[2]);atomicStore(&arena[14],graph[5]);atomicStore(&arena[15],bitcast<u32>(p.originCell.x));atomicStore(&arena[16],bitcast<u32>(p.originCell.y));atomicStore(&arena[17],bitcast<u32>(p.originCell.z));atomicStore(&arena[18],20u+8u*p.limits.y+4u*cap);}
@compute @workgroup_size(64)fn prepareTopologyEvidence(@builtin(global_invocation_id)gid:vec3u){let item=gid.x;let cap=p.policy.w;if(atomicLoad(&arena[19])==graph[0]&&graph[0]!=0u){return;}if(item<arrayLength(&renderer)){renderer[item]=0u;}if(item<cap){let at=20u+8u*p.limits.y+4u*item;atomicStore(&arena[at],0u);atomicStore(&arena[at+1u],0u);atomicStore(&arena[at+2u],0u);atomicStore(&arena[at+3u],0u);}}
@compute @workgroup_size(64)
fn publishTopologyEvidenceRows(@builtin(global_invocation_id)gid:vec3u){
 let row=gid.x;let rows=min(graph[1],p.limits.y);
 if(row>=rows||atomicLoad(&arena[13])!=0u){return;}
 let entry=leafDirectory[row];
 if(entry.slot>=rows||entry.flags==0u
   ||(row>0u&&(leafDirectory[row-1u].cell>entry.cell
   ||(leafDirectory[row-1u].cell==entry.cell&&leafDirectory[row-1u].span>=entry.span)))){
  atomicOr(&arena[13],2u);return;
 }
 let leaf=leaves[entry.slot];var values:array<f32,8>;
 var mn=3.402823e38;var mx=-3.402823e38;let bank=state[6]&1u;
 let aux=8u+8u*rows+8u*row;
 let cornerBase=20u+8u*p.limits.y+4u*p.policy.w+8u*row;
 for(var corner=0u;corner<8u;corner+=1u){
  let slot=rendererCorner(leaf,corner);let value=publishedNodeValue(slot,bank);
  if(!finite(value)||cornerBase+corner>=arrayLength(&arena)){
   atomicOr(&arena[13],8u);return;
  }
  values[corner]=value;mn=min(mn,value);mx=max(mx,value);
  renderer[aux+corner]=bitcast<u32>(value);
  atomicStore(&arena[cornerBase+corner],bitcast<u32>(value));
 }
 for(var x=0u;x<8u;x+=1u){for(var y=x+1u;y<8u;y+=1u){
  if(values[y]<values[x]){let swap=values[x];values[x]=values[y];values[y]=swap;}
 }}
 let a0=values[0]+values[1];let a1=values[2]+values[3];
 let a2=values[4]+values[5];let a3=values[6]+values[7];
 let rawCentre=clamp(.125*((a0+a1)+(a2+a3)),mn,mx);
 let centre=canonicalPublishedZero(rawCentre);
 mn=canonicalPublishedZero(mn);mx=canonicalPublishedZero(mx);
 let width=f32(entry.span)*p.originCell.w;let geometricVolume=width*width*width;
 // Phi is metric reach evidence; rho contributes only the local physical
 // rho=.5 crossing. In particular, partial-density transport tails do not
 // become a second distance band.
 let densityDetail=rhoHalfInterface(leaf,entry.slot);
 let flags=3u|ADAPTIVE_CORNERS|select(0u,4u,mn<=0.&&mx>=0.)
   |select(0u,0x20000000u,densityDetail);
 let base=20u+8u*row;
 atomicStore(&arena[base],entry.cell+1u);atomicStore(&arena[base+1u],entry.span);
 atomicStore(&arena[base+2u],bitcast<u32>(centre));atomicStore(&arena[base+3u],bitcast<u32>(mn));
 atomicStore(&arena[base+4u],bitcast<u32>(mx));atomicStore(&arena[base+5u],flags);
 atomicStore(&arena[base+6u],row);let volumeBits=bitcast<u32>(geometricVolume);
 atomicStore(&arena[base+7u],volumeBits);
 let rb=8u+8u*row;renderer[rb]=entry.cell+1u;renderer[rb+1u]=entry.span;
 renderer[rb+2u]=bitcast<u32>(centre);renderer[rb+3u]=bitcast<u32>(mn);
 renderer[rb+4u]=bitcast<u32>(mx);renderer[rb+5u]=9u|ADAPTIVE_CORNERS;
 renderer[rb+6u]=row;renderer[rb+7u]=volumeBits;
}
@compute @workgroup_size(1)fn finishTopologyEvidence(){let rows=min(graph[1],p.limits.y);let cap=p.policy.w;let epoch=graph[0];let publicationMatches=evidenceValid(rows,cap)&&atomicLoad(&arena[1])==epoch&&atomicLoad(&arena[2])==rows&&atomicLoad(&arena[12])==state[2]&&atomicLoad(&arena[14])==graph[5];if(!publicationMatches){atomicOr(&arena[13],16u);}if(atomicLoad(&arena[13])!=0u){atomicStore(&arena[0],0u);atomicStore(&arena[19],0u);return;}let reuse=atomicLoad(&arena[19])==epoch;if(!reuse){for(var row=0u;row<rows;row+=1u){let entry=leafDirectory[row];let wanted=hash(entry.cell,entry.span);var installed=false;for(var probe=0u;probe<32u;probe+=1u){let at=20u+8u*p.limits.y+4u*((wanted+probe)&(cap-1u));if(atomicLoad(&arena[at])==0u){atomicStore(&arena[at],entry.cell+1u);atomicStore(&arena[at+1u],entry.span);atomicStore(&arena[at+2u],row+1u);atomicStore(&arena[at+3u],wanted);installed=true;break;}}if(!installed){atomicOr(&arena[13],4u);break;}}}if(atomicLoad(&arena[13])==0u){renderer[0]=0x80000000u;renderer[1]=state[2];renderer[2]=rows;renderer[3]=p.dims.w;renderer[4]=p.dims.x;renderer[5]=p.dims.y;renderer[6]=p.dims.z;renderer[7]=bitcast<u32>(p.originCell.w);atomicStore(&arena[19],epoch);atomicStore(&arena[0],MAGIC);atomicStore(&receipts[15],rows);atomicStore(&receipts[23],state[2]);}else{atomicStore(&arena[0],0u);atomicStore(&arena[19],0u);}}
`;

/**
 * Accepted graph work scheduler.  The graph publishes node/leaf dispatches,
 * but phi also needs matching pressure-row and face schedules.  Re-derive all
 * four records from one validated epoch so no recurring consumer can launch
 * from host capacity or from a stale mixed-generation tuple.
 */
export const octreeLosassoAdaptivePhiAcceptedScheduleWGSL = /* wgsl */ `
const MAGIC:u32=0x41504849u;const ERR_GRAPH:u32=1u;
const ERROR_GRAPH:u32=1u;const ERROR_STATE:u32=2u;const ERROR_COUNT:u32=4u;
const ERROR_DISPATCH:u32=8u;const ERROR_FACE:u32=16u;
struct Limits{counts:vec4u}
@group(0)@binding(0)var<uniform>limits:Limits;
@group(0)@binding(1)var<storage,read>graph:array<u32>;
@group(0)@binding(2)var<storage,read_write>state:array<atomic<u32>>;
@group(0)@binding(3)var<storage,read>pressureRowToLeaf:array<u32>;
@group(0)@binding(4)var<storage,read>faceControl:array<u32>;
@group(0)@binding(5)var<storage,read_write>nodeDispatch:array<u32>;
@group(0)@binding(6)var<storage,read_write>leafDispatch:array<u32>;
@group(0)@binding(7)var<storage,read_write>rowDispatch:array<u32>;
@group(0)@binding(8)var<storage,read_write>faceDispatch:array<u32>;
@group(0)@binding(9)var<storage,read_write>schedule:array<atomic<u32>>;
fn groups(count:u32)->u32{return(count+63u)/64u;}
@compute @workgroup_size(1)fn scheduleAcceptedWork(){
 nodeDispatch[0]=0u;nodeDispatch[1]=1u;nodeDispatch[2]=1u;
 leafDispatch[0]=0u;leafDispatch[1]=1u;leafDispatch[2]=1u;
 rowDispatch[0]=0u;rowDispatch[1]=1u;rowDispatch[2]=1u;
 faceDispatch[0]=0u;faceDispatch[1]=1u;faceDispatch[2]=1u;
 for(var i=0u;i<8u;i+=1u){atomicStore(&schedule[i],0u);}
 if(arrayLength(&graph)<29u||arrayLength(&state)<20u){
  atomicStore(&schedule[2],ERROR_GRAPH);if(arrayLength(&state)>12u){atomicOr(&state[12],ERR_GRAPH);}return;
 }
 let epoch=graph[0];let leaves=graph[1];let nodes=graph[2];let rows=graph[28];
 var errors=0u;
 if(epoch==0u||graph[3]!=epoch||graph[4]!=0u){errors|=ERROR_GRAPH;}
 if(atomicLoad(&state[0])!=MAGIC||atomicLoad(&state[1])!=epoch||atomicLoad(&state[7])!=1u
    ||atomicLoad(&state[12])!=0u||atomicLoad(&state[2])==0u||graph[5]!=atomicLoad(&state[2])
    ||atomicLoad(&state[4])!=nodes||atomicLoad(&state[5])!=leaves){errors|=ERROR_STATE;}
 if(nodes>limits.counts.x||leaves>limits.counts.y||rows>limits.counts.z
    ||rows>arrayLength(&pressureRowToLeaf)||graph[14]!=limits.counts.y
    ||graph[15]!=limits.counts.x){errors|=ERROR_COUNT;}
 let expectedLeaves=groups(leaves);let expectedNodes=groups(nodes);
 if(graph[17]!=expectedLeaves||graph[18]!=1u||graph[19]!=1u
    ||graph[20]!=expectedNodes||graph[21]!=1u||graph[22]!=1u){errors|=ERROR_DISPATCH;}
 var faces=0u;
 if(arrayLength(&faceControl)>=5u){
  faces=faceControl[2];
  if(faceControl[0]!=epoch||faceControl[1]!=rows||faceControl[3]!=1u
      ||faceControl[4]!=0u||faces>limits.counts.w){errors|=ERROR_FACE;}
 }
 atomicStore(&schedule[0],epoch);atomicStore(&schedule[1],atomicLoad(&state[2]));
 atomicStore(&schedule[2],errors);atomicStore(&schedule[4],nodes);
 atomicStore(&schedule[5],leaves);atomicStore(&schedule[6],rows);
 atomicStore(&schedule[7],faces);
 if(errors!=0u){atomicOr(&state[12],ERR_GRAPH);return;}
 nodeDispatch[0]=groups(nodes);nodeDispatch[1]=1u;nodeDispatch[2]=1u;
 leafDispatch[0]=groups(leaves);leafDispatch[1]=1u;leafDispatch[2]=1u;
 rowDispatch[0]=groups(rows);rowDispatch[1]=1u;rowDispatch[2]=1u;
 faceDispatch[0]=groups(faces);faceDispatch[1]=1u;faceDispatch[2]=1u;
 atomicStore(&schedule[3],1u);
}
`;

export const octreeLosassoAdaptivePhiScheduleWGSL = /* wgsl */ `
const MAGIC:u32=0x41504849u;const ERR_GRAPH:u32=1u;struct Params{dims:vec4u,originCell:vec4f,limits:vec4u,numerics:vec4f,policy:vec4u,volume:vec4f,inflow0:vec4f,inflow1:vec4f}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>acceptedGraph:array<u32>;@group(0)@binding(2)var<storage,read>candidateGraph:array<u32>;
@group(0)@binding(3)var<storage,read>topologyControl:array<u32>;@group(0)@binding(4)var<storage,read_write>state:array<atomic<u32>>;@group(0)@binding(5)var<storage,read_write>dispatch:array<u32>;@group(0)@binding(6)var<storage,read_write>receipts:array<atomic<u32>>;
fn groups(count:u32)->u32{return min((count+63u)/64u,65535u);}fn setDispatch(word:u32,x:u32){dispatch[word]=x;dispatch[word+1u]=1u;dispatch[word+2u]=1u;}
@compute @workgroup_size(1)fn scheduleCandidateSource(){let warm=acceptedGraph[0]!=0u&&acceptedGraph[3]==acceptedGraph[0]&&acceptedGraph[4]==0u&&acceptedGraph[5]!=0u;let candidateValid=candidateGraph[0]!=0u&&candidateGraph[3]==candidateGraph[0]&&candidateGraph[4]==0u&&candidateGraph[2]<=p.limits.x&&candidateGraph[1]<=p.limits.y;
 if(warm){for(var i=16u;i<=21u;i+=1u){atomicStore(&receipts[i],0u);}for(var i=24u;i<32u;i+=1u){atomicStore(&receipts[i],0u);}}else{for(var i=0u;i<32u;i+=1u){atomicStore(&receipts[i],0u);}}if(!warm){for(var i=0u;i<20u;i+=1u){atomicStore(&state[i],0u);}atomicStore(&state[0],MAGIC);atomicStore(&state[1],candidateGraph[0]);atomicStore(&state[4],candidateGraph[2]);atomicStore(&state[5],candidateGraph[1]);}
 else{atomicStore(&state[1],acceptedGraph[0]);atomicStore(&state[2],acceptedGraph[5]);atomicStore(&state[3],acceptedGraph[6]);atomicStore(&state[4],acceptedGraph[2]);atomicStore(&state[5],acceptedGraph[1]);atomicStore(&state[7],1u);atomicStore(&state[12],0u);}
 atomicStore(&state[8],candidateGraph[0]);atomicStore(&state[10],candidateGraph[2]);atomicStore(&state[11],0u);atomicStore(&state[14],0u);atomicStore(&state[18],0u);atomicStore(&state[19],0u);
 // An inactive candidate is speculative. Once an accepted scalar exists, a
 // missing/rejected candidate publishes zero work and candidate-local invalid
 // receipts without poisoning accepted state. Cold bootstrap has no accepted
 // fallback and therefore retains the original fail-closed error.
 if(!candidateValid&&!warm){atomicOr(&state[12],ERR_GRAPH);}
 let wn=select(0u,groups(candidateGraph[2]),warm&&candidateValid);let cn=select(groups(candidateGraph[2]),0u,warm||!candidateValid);let cl=select(groups(candidateGraph[1]),0u,warm||!candidateValid);setDispatch(0u,wn);setDispatch(3u,select(0u,1u,warm&&candidateValid));setDispatch(6u,cn);setDispatch(9u,cl);setDispatch(12u,select(1u,0u,warm||!candidateValid));}
@compute @workgroup_size(1)fn scheduleCandidateRepair(){let warm=acceptedGraph[0]!=0u&&acceptedGraph[3]==acceptedGraph[0]&&acceptedGraph[4]==0u;let changed=topologyControl[5]==0u&&warm;let valid=candidateGraph[0]!=0u&&candidateGraph[3]==candidateGraph[0]&&candidateGraph[4]==0u;atomicStore(&state[17],select(0u,1u,changed&&valid));setDispatch(15u,select(0u,groups(candidateGraph[2]),changed&&valid));setDispatch(18u,select(0u,1u,changed&&valid));setDispatch(21u,select(0u,groups(candidateGraph[2]),valid));setDispatch(24u,select(0u,groups(candidateGraph[1]),valid));setDispatch(27u,select(0u,1u,valid));}
`;
