/** Read-only relocation of a prepared CM12 seam into a candidate world image. */
export const SPARSE_CM12_DYNAMIC_SEAM_BINDING_WGSL = /* wgsl */ `
fn cm12PreparedSeamHeader(own:u32,other:u32,side:u32)->u32{
  if(!validBrickResolution(own)||(other!=0u&&!validBrickResolution(other))||side>=6u){return INVALID;}
  let neighbor=select(0u,1u+cm12DynamicRungIndex(max(1u,other)),other!=0u);
  let at=CM12_DYNAMIC_SEAM_CATALOGUE+4u*((cm12DynamicRungIndex(own)*5u+neighbor)*6u+side);
  return select(INVALID,at,ta(at+3u)==1u);
}
fn cm12PreparedPageReady(leaf:u32)->bool{
  if(leaf<CM12_WDR_INITIAL_LEAVES||leaf>=p.dispatch.w){return false;}
  let page=leaf-CM12_WDR_INITIAL_LEAVES;let base=candidateTopologyPageBase(page);
  return ta(base)==leaf&&(ta(base+3u)&0x8000001fu)==0x8000001fu
    &&ta(base+4u)==CM12_DYNAMIC_ROWS&&ta(base+5u)==6u*CM12_DYNAMIC_CELLS;
}
// (stable row, stable first term, prepared relative row, other leaf).
// The negative dynamic page owns a shared face; a dynamic page always owns
// its host seam. This choice is independent of which side is finer.
fn cm12PreparedDynamicFace(leaf:u32,resolution:u32,side:u32,point:vec3f)->vec4u{
  if(brickSpan(leaf)!=1u){return vec4u(INVALID);}
  let axis=side/2u;var direction=vec3i(0);direction[axis]=select(-1,1,(side&1u)!=0u);
  var other=cm12WorldOwnerAt(cm12WorldLeafCoordinate(leaf)+direction);
  if(other!=INVALID&&!scheduledBrickActive(other)){other=INVALID;}
  var owner=leaf;var ownerResolution=resolution;var ownerSide=side;
  if(leaf<CM12_WDR_INITIAL_LEAVES||(other!=INVALID
    &&other>=CM12_WDR_INITIAL_LEAVES&&(side&1u)==0u)){
    if(other==INVALID||other<CM12_WDR_INITIAL_LEAVES){return vec4u(INVALID);}
    owner=other;ownerResolution=scheduledBrickResolution(owner);ownerSide=side^1u;other=leaf;
  }
  if(!cm12PreparedPageReady(owner)){return vec4u(INVALID);}
  var otherResolution=0u;
  if(other!=INVALID){
    if(brickSpan(other)!=1u){return vec4u(INVALID);}
    otherResolution=scheduledBrickResolution(other);
    if(other>=CM12_WDR_INITIAL_LEAVES&&!cm12PreparedPageReady(other)){return vec4u(INVALID);}
  }
  let header=cm12PreparedSeamHeader(ownerResolution,otherResolution,ownerSide);
  if(header==INVALID){return vec4u(INVALID);}
  let origin=8.0*vec3f(cm12WorldLeafCoordinate(owner));
  let relative=point-origin;let width=8.0/f32(ownerResolution);
  if(relative[axis]!=select(0.0,8.0,(ownerSide&1u)!=0u)){return vec4u(INVALID);}
  let u=relative[(axis+1u)%3u];let v=relative[(axis+2u)%3u];
  if(u<0.0||v<0.0||u>=8.0||v>=8.0){return vec4u(INVALID);}
  let uv=u32(floor(u/width))+ownerResolution*u32(floor(v/width));
  let rowRecord=CM12_DYNAMIC_SEAM_CATALOGUE+ta(CM12_DYNAMIC_SEAM_CATALOGUE+ta(header+2u)+2u*uv);
  let row=ta(rowRecord);let page=owner-CM12_WDR_INITIAL_LEAVES;
  return vec4u(ta(3u)+page*CM12_DYNAMIC_ROWS+cm12DynamicRowOffset(ownerResolution)+row,
    ta(4u)+page*CM12_DYNAMIC_TERMS+cm12DynamicRowFirstTerm(ownerResolution,row),rowRecord,other);
}
fn cm12ScheduledDynamicRowBinding(row:u32)->vec4u{
  if(row<ta(3u)){return vec4u(INVALID);}
  let page=(row-ta(3u))/CM12_DYNAMIC_ROWS;let within=(row-ta(3u))%CM12_DYNAMIC_ROWS;
  let resolution=cm12DynamicRowRung(within);let local=within-cm12DynamicRowOffset(resolution);
  let perAxis=(resolution+1u)*resolution*resolution;let axis=local/perAxis;
  let face=local%(resolution+1u);if(face!=0u&&face!=resolution){return vec4u(INVALID);}
  let uv=(local%perAxis)/(resolution+1u);let width=8.0/f32(resolution);
  let leaf=CM12_WDR_INITIAL_LEAVES+page;var point=8.0*vec3f(cm12WorldLeafCoordinate(leaf));
  point[axis]+=f32(face)*width;
  point[(axis+1u)%3u]+=(f32(uv%resolution)+0.5)*width;
  point[(axis+2u)%3u]+=(f32(uv/resolution)+0.5)*width;
  return cm12PreparedDynamicFace(leaf,resolution,2u*axis+select(0u,1u,face==resolution),point);
}
fn cm12PreparedDynamicTerm(binding:vec4u,ordinal:u32)->u32{
  if(binding.x==INVALID||ordinal>=ta(binding.z+6u)){return INVALID;}
  let owner=CM12_WDR_INITIAL_LEAVES+(binding.x-ta(3u))/CM12_DYNAMIC_ROWS;
  let at=binding.z+8u+3u*ordinal;let leaf=select(owner,binding.w,ta(at)!=0u);
  if(leaf==INVALID){return INVALID;}
  let cells=templateBrickCellRange(leaf,scheduledBrickResolution(leaf));let local=ta(at+1u);
  return select(INVALID,cells.x+local,cells.y>local);
}
fn cm12PreparedDynamicTermForCell(binding:vec4u,cell:u32)->u32{
  if(binding.x==INVALID){return INVALID;}
  for(var ordinal=0u;ordinal<ta(binding.z+6u);ordinal+=1u){
    if(cm12PreparedDynamicTerm(binding,ordinal)==cell){return ordinal;}}
  return INVALID;
}
fn compiledHostDynamicSeam(cell:u32,incidence:u32)->vec4u{
  if(cell>=ta(2u)||incidence>=ta(5u)){return vec4u(INVALID);}
  let row=ta(IMMUTABLE_HOST_INCIDENCE_BASE+2u*incidence);
  let term=ta(IMMUTABLE_HOST_INCIDENCE_BASE+2u*incidence+1u);
  if(rowKind(row)!=3u||termCell(term)!=cell){return vec4u(INVALID);}
  let leaf=cellBrick(cell);if(!scheduledBrickActive(leaf)
    ||cellResolution(cell)!=scheduledBrickResolution(leaf)){return vec4u(INVALID);}
  let side=2u*rowAxis(row)+select(0u,1u,termCoefficient(term)<0.0);
  let binding=cm12PreparedDynamicFace(leaf,cellResolution(cell),side,rowCenter(row));
  let ordinal=cm12PreparedDynamicTermForCell(binding,cell);
  if(ordinal==INVALID){return vec4u(INVALID);}
  return vec4u(binding.x,binding.y+ordinal,binding.z,term);
}
fn compiledDynamicBoundaryFace(cell:u32,incidence:u32)->u32{
  if(cell<ta(2u)){return 0u;}
  let resolution=cellResolution(cell);let leaf=cellBrick(cell);
  let side=(incidence-ta(5u))%6u;let axis=side/2u;let positive=(side&1u)!=0u;
  let local=dynamicCellWithin(dynamicCellLocal(cell))-cm12DynamicCellOffset(resolution);
  let q=vec3u(local%resolution,(local/resolution)%resolution,local/(resolution*resolution));
  if(q[axis]!=select(0u,resolution-1u,positive)){return 0u;}
  var point=cellCenter(cell);point[axis]+=select(-0.5,0.5,positive)*cellWidths(cell)[axis];
  let binding=cm12PreparedDynamicFace(leaf,resolution,side,point);
  return select(2u,1u,cm12PreparedDynamicTermForCell(binding,cell)!=INVALID);
}
`;

export const SPARSE_CM12_DYNAMIC_SEAM_PUBLICATION_WGSL = /* wgsl */ `
fn cm12InstallPreparedDynamicRow(page:u32,resolution:u32,record:u32,other:u32){
  let localRow=ta(record);let rowWithin=cm12DynamicRowOffset(resolution)+localRow;
  let pageBase=candidateTopologyPageBase(page);let rowBase=pageBase+ta(pageBase+7u);
  let first=cm12DynamicRowFirstTerm(resolution,localRow);
  let termBase=pageBase+ta(pageBase+8u)+2u*first;
  let stableRow=ta(3u)+page*CM12_DYNAMIC_ROWS+rowWithin;
  let stableTerm=ta(4u)+page*CM12_DYNAMIC_TERMS+first;
  let owner=CM12_WDR_INITIAL_LEAVES+page;
  let axis=localRow/((resolution+1u)*resolution*resolution);
  let count=ta(record+6u);let area=taf(record+4u);let distance=taf(record+5u);
  for(var ordinal=0u;ordinal<count;ordinal+=1u){
    let source=record+8u+3u*ordinal;let neighbor=ta(source)!=0u;
    let leaf=select(owner,other,neighbor);let rung=select(resolution,scheduledBrickResolution(leaf),neighbor);
    let cells=templateBrickCellRange(leaf,rung);let local=ta(source+1u);
    if(leaf==INVALID||local>=cells.y){cm12RecordFailure(8u,stableRow,vec4u(leaf,rung,local,cells.y));return;}
    atomicStore(&topologyArena[termBase+2u*ordinal],cells.x+local);
    atomicStore(&topologyArena[termBase+2u*ordinal+1u],ta(source+2u));
  }
  let kind=select(select(1u,2u,count>2u),3u,other==INVALID);
  atomicStore(&topologyArena[rowBase+rowWithin],stableTerm|(count<<23u));
  atomicStore(&topologyArena[rowBase+CM12_DYNAMIC_ROWS+rowWithin],(axis<<30u)|(kind<<28u));
  atomicStore(&topologyArena[rowBase+2u*CM12_DYNAMIC_ROWS+rowWithin],bitcast<u32>(distance));
  atomicStore(&topologyArena[rowBase+3u*CM12_DYNAMIC_ROWS+rowWithin],bitcast<u32>(area*distance));
  for(var a=0u;a<3u;a+=1u){atomicStore(&topologyArena[rowBase+(4u+a)*CM12_DYNAMIC_ROWS+rowWithin],ta(record+1u+a));}
  atomicStore(&topologyArena[rowBase+7u*CM12_DYNAMIC_ROWS+rowWithin],bitcast<u32>(area));
}
@compute @workgroup_size(64)
fn resetSparseWorldFrontierBindings(@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  let page=wid.x;if(page>=ta(topologyWorklistBase()+27u)||ta(topologyWorklistBase()+3u)!=2u){return;}
  let leaf=CM12_WDR_INITIAL_LEAVES+page;if(!cm12PreparedPageReady(leaf)){return;}
  let pageBase=candidateTopologyPageBase(page);
  // Restore host slots from their immutable authority before any page binds
  // next-generation seams. This dispatch precedes all new bindings.
  for(var rung=1u;rung<=8u;rung*=2u){
    for(var side=0u;side<6u;side+=1u){
      let header=cm12PreparedSeamHeader(rung,0u,side);
      for(var ordinal=lane;ordinal<ta(header+1u);ordinal+=64u){
        let record=CM12_DYNAMIC_SEAM_CATALOGUE+ta(header)+23u*ordinal;
        let localRow=ta(record);let row=ta(3u)+page*CM12_DYNAMIC_ROWS+cm12DynamicRowOffset(rung)+localRow;
        let range=rowTermRange(row);
        for(var term=range.x;term<range.y;term+=1u){let cell=termCell(term);
          if(cell<ta(2u)){restoreHostExteriorIncidence(cell,row,term);}}
        cm12InstallPreparedDynamicRow(page,rung,record,INVALID);
        let local=cm12DynamicCellOffset(rung)+ta(record+9u);
        let incidence=dynamicIncidenceOverrideAt(pageBase,local,side);
        atomicStore(&topologyArena[incidence],row);
        atomicStore(&topologyArena[incidence+1u],ta(4u)+page*CM12_DYNAMIC_TERMS+cm12DynamicRowFirstTerm(rung,localRow));
      }
    }
  }
}
@compute @workgroup_size(64)
fn connectSparseWorldFrontierPages(@builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  let page=wid.x;if(page>=ta(topologyWorklistBase()+27u)||ta(topologyWorklistBase()+3u)!=2u){return;}
  let leaf=CM12_WDR_INITIAL_LEAVES+page;
  if(!cm12PreparedPageReady(leaf)||!candidateBrickActive(leaf)){return;}
  let resolution=scheduledBrickResolution(leaf);let origin=8.0*vec3f(cm12WorldLeafCoordinate(leaf));
  for(var side=0u;side<6u;side+=1u){
    let axis=side/2u;var delta=vec3i(0);delta[axis]=select(-1,1,(side&1u)!=0u);
    var other=cm12WorldOwnerAt(cm12WorldLeafCoordinate(leaf)+delta);
    if(other!=INVALID&&!scheduledBrickActive(other)){other=INVALID;}
    if(other!=INVALID&&other>=CM12_WDR_INITIAL_LEAVES&&(side&1u)==0u){continue;}
    let otherRung=select(0u,scheduledBrickResolution(select(leaf,other,other!=INVALID)),other!=INVALID);
    let header=cm12PreparedSeamHeader(resolution,otherRung,side);
    if(header==INVALID){cm12RecordFailure(8u,leaf,vec4u(resolution,otherRung,side,other));return;}
    for(var ordinal=lane;ordinal<ta(header+1u);ordinal+=64u){
      let record=CM12_DYNAMIC_SEAM_CATALOGUE+ta(header)+23u*ordinal;
      cm12InstallPreparedDynamicRow(page,resolution,record,other);
      let row=ta(3u)+page*CM12_DYNAMIC_ROWS+cm12DynamicRowOffset(resolution)+ta(record);
      let first=ta(4u)+page*CM12_DYNAMIC_TERMS+cm12DynamicRowFirstTerm(resolution,ta(record));
      for(var t=0u;t<ta(record+6u);t+=1u){
        let cell=termCell(first+t);let coefficient=termCoefficient(first+t);
        let cellSide=2u*axis+select(0u,1u,coefficient<0.0);
        if(cell>=ta(2u)){
          let local=dynamicCellLocal(cell);let targetPage=dynamicCellPage(local);
          let at=dynamicIncidenceOverrideAt(candidateTopologyPageBase(targetPage),dynamicCellWithin(local),cellSide);
          atomicStore(&topologyArena[at],row);atomicStore(&topologyArena[at+1u],first+t);
        }else{
          var patched=false;
          for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
            let source=ta(IMMUTABLE_HOST_INCIDENCE_BASE+2u*incidence);
            let sourceTerm=ta(IMMUTABLE_HOST_INCIDENCE_BASE+2u*incidence+1u);
            if(rowKind(source)!=3u||rowAxis(source)!=axis||termCell(sourceTerm)!=cell
              ||termCoefficient(sourceTerm)*coefficient<=0.0){continue;}
            atomicStore(&topologyArena[ta(10u)+2u*incidence],row);
            atomicStore(&topologyArena[ta(10u)+2u*incidence+1u],first+t);patched=true;break;
          }
          if(!patched){cm12RecordFailure(8u,cell,vec4u(row,first+t,side,leaf));return;}
        }
      }
    }
  }
}
`;
