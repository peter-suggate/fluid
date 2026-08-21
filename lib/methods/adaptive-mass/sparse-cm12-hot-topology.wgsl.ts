import {
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_COMMON_TWO_TERM,
  SPARSE_CM12_HOT_TOPOLOGY_COMPLETE,
  SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_HEADER,
  SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_INVALID,
  SPARSE_CM12_HOT_TOPOLOGY_MAGIC,
  SPARSE_CM12_HOT_TOPOLOGY_ROW,
  SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_VALIDATED,
  SPARSE_CM12_HOT_TOPOLOGY_VERSION,
  type SparseCM12HotTopologyLayout,
} from "./sparse-cm12-hot-topology";
import { createSparseCM12LogicalOwnerDirectoryWGSL } from
  "./sparse-cm12-logical-owner-directory.wgsl";

export interface SparseCM12HotTopologyWGSLOptions {
  readonly layout: SparseCM12HotTopologyLayout;
  /** Existing immutable `array<u32>` containing the combined LOD1/HTP1 arena. */
  readonly arenaName?: string;
  readonly brickActiveFunction?: string;
  readonly acceptedBrickResolutionFunction?: string;
  readonly templateBrickCellRangeFunction?: string;
  readonly cellResolutionFunction?: string;
  readonly cellOpenVolumeFunction?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Binding-free accessors for the immutable combined LOD1/HTP1 arena. */
export function createSparseCM12HotTopologyWGSL(
  options: SparseCM12HotTopologyWGSLOptions,
): string {
  const { layout: l } = options;
  const arena = identifier(options.arenaName ?? "hotTopology", "arenaName");
  const brickActive = identifier(options.brickActiveFunction ?? "brickActive",
    "brickActiveFunction");
  const acceptedResolution = identifier(
    options.acceptedBrickResolutionFunction ?? "acceptedBrickResolution",
    "acceptedBrickResolutionFunction",
  );
  const lod = createSparseCM12LogicalOwnerDirectoryWGSL({
    layout: l.logicalOwner, directoryName: arena,
    brickActiveFunction: brickActive,
    acceptedBrickResolutionFunction: acceptedResolution,
    templateBrickCellRangeFunction: options.templateBrickCellRangeFunction,
    cellResolutionFunction: options.cellResolutionFunction,
    cellOpenVolumeFunction: options.cellOpenVolumeFunction,
  });
  const h = (word: number) => `${l.headerBaseWords + word}u`;
  return /* wgsl */ `
${lod}

const cm12HotInvalid:u32=0x${SPARSE_CM12_HOT_TOPOLOGY_INVALID.toString(16)}u;
const cm12HotMagic:u32=0x${SPARSE_CM12_HOT_TOPOLOGY_MAGIC.toString(16)}u;
const cm12HotVersion:u32=${SPARSE_CM12_HOT_TOPOLOGY_VERSION}u;
const cm12HotHeaderBase:u32=${l.headerBaseWords}u;
const cm12HotTotalWords:u32=${l.totalWords}u;
const cm12HotCellCount:u32=${l.cellCount}u;
const cm12HotRowCount:u32=${l.rowCount}u;
const cm12HotVariableTermCount:u32=${l.variableTermCount}u;
const cm12HotIncidenceCount:u32=${l.incidenceCount}u;
const cm12HotDirectedEdgeCount:u32=${l.directedEdgeCount}u;
const cm12HotRequirementCount:u32=${l.requirementCount}u;
const cm12HotCellBase:u32=${l.cellBaseWords}u;
const cm12HotRowBase:u32=${l.rowBaseWords}u;
const cm12HotVariableTermBase:u32=${l.variableTermBaseWords}u;
const cm12HotIncidenceOffsetBase:u32=${l.incidenceOffsetBaseWords}u;
const cm12HotIncidenceBase:u32=${l.incidenceBaseWords}u;
const cm12HotDirectedEdgeOffsetBase:u32=${l.directedEdgeOffsetBaseWords}u;
const cm12HotDirectedEdgeBase:u32=${l.directedEdgeBaseWords}u;
const cm12HotRequirementBase:u32=${l.requirementBaseWords}u;
const cm12HotCommonTwoTerm:u32=${SPARSE_CM12_HOT_TOPOLOGY_COMMON_TWO_TERM}u;

struct CM12HotCell {
  valid:u32,
  center:vec3f,
  volume:f32,
  widths:vec3f,
  brick:u32,
  resolution:u32,
}

fn cm12HotInvalidCell()->CM12HotCell{
  return CM12HotCell(0u,vec3f(0.0),0.0,vec3f(0.0),cm12HotInvalid,0u);
}

fn cm12HotHeaderValid()->bool{
  let flags=${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.flags)}];
  return arrayLength(&${arena})>=cm12HotTotalWords
    &&cm12LogicalOwnerHeaderValid()
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.magic)}]==cm12HotMagic
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.version)}]==cm12HotVersion
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.headerWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_HEADER_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.brickFineResolution)}]
      ==${l.brickFineResolution}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.presentationPageResolution)}]
      ==${l.presentationPageResolution}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.atlasGeneration)}]
      ==${l.atlasGeneration}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.logicalOwnerWords)}]
      ==${l.logicalOwnerWords}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellCount)}]==cm12HotCellCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowCount)}]==cm12HotRowCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.variableTermCount)}]
      ==cm12HotVariableTermCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceCount)}]
      ==cm12HotIncidenceCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeCount)}]
      ==cm12HotDirectedEdgeCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.requirementCount)}]
      ==cm12HotRequirementCount
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellBase)}]==cm12HotCellBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowBase)}]==cm12HotRowBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.variableTermBase)}]
      ==cm12HotVariableTermBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceOffsetBase)}]
      ==cm12HotIncidenceOffsetBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceBase)}]
      ==cm12HotIncidenceBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeOffsetBase)}]
      ==cm12HotDirectedEdgeOffsetBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.directedEdgeBase)}]
      ==cm12HotDirectedEdgeBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.requirementBase)}]
      ==cm12HotRequirementBase
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.totalWords)}]==cm12HotTotalWords
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.cellWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.rowWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.termWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.incidenceWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.edgeWords)}]
      ==${SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS}u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved0)}]==0u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved1)}]==0u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved2)}]==0u
    &&${arena}[${h(SPARSE_CM12_HOT_TOPOLOGY_HEADER.reserved3)}]==0u
    &&(flags&${SPARSE_CM12_HOT_TOPOLOGY_COMPLETE}u)!=0u
    &&(flags&${SPARSE_CM12_HOT_TOPOLOGY_VALIDATED}u)!=0u;
}

fn cm12HotCellAt(cell:u32)->CM12HotCell{
  if(!cm12HotHeaderValid()||cell>=cm12HotCellCount){return cm12HotInvalidCell();}
  let at=cm12HotCellBase+${SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS}u*cell;
  let metadata=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution}u];
  let brick=metadata>>5u;let resolution=metadata&31u;
  if(brick>=cm12LogicalOwnerResidentCount||resolution==0u
    ||resolution>cm12LogicalOwnerBrickFine||(resolution&(resolution-1u))!=0u){
    return cm12HotInvalidCell();
  }
  return CM12HotCell(1u,
    vec3f(bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.centerX}u]),
      bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.centerY}u]),
      bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.centerZ}u])),
    bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.volume}u]),
    vec3f(bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.widthX}u]),
      bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.widthY}u]),
      bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_CELL.widthZ}u])),brick,resolution);
}

fn cm12HotRowTag(row:u32)->u32{
  if(!cm12HotHeaderValid()||row>=cm12HotRowCount){return cm12HotInvalid;}
  return ${arena}[cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row];
}
fn cm12HotRowTermCount(row:u32)->u32{return cm12HotRowTag(row)&0xffffu;}
fn cm12HotRowAxis(row:u32)->u32{return (cm12HotRowTag(row)>>16u)&3u;}
fn cm12HotRowKind(row:u32)->u32{return (cm12HotRowTag(row)>>18u)&3u;}
fn cm12HotRowIsCommonTwoTerm(row:u32)->bool{
  return (cm12HotRowTag(row)&cm12HotCommonTwoTerm)!=0u;
}

fn cm12HotRowValid(row:u32)->bool{
  let tag=cm12HotRowTag(row);
  if(tag==cm12HotInvalid){return false;}
  let count=tag&0xffffu;let axis=(tag>>16u)&3u;let kind=(tag>>18u)&3u;
  let isCommon=(tag&cm12HotCommonTwoTerm)!=0u;
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  let first=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm}u];
  let requirementFirst=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.firstRequirement}u];
  let requirementCount=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.requirementCount}u];
  var variableRangeValid=true;
  if(!isCommon){
    variableRangeValid=count<=cm12HotVariableTermCount
      &&first<=cm12HotVariableTermCount-count;
  }
  return count>0u&&axis<3u&&kind<4u&&(tag&0xffe00000u)==0u
    &&isCommon==(count==2u)
    &&variableRangeValid
    &&requirementFirst<=cm12HotRequirementCount
    &&requirementCount<=cm12HotRequirementCount-requirementFirst
    &&${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.reserved}u]==0u
    &&${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.identity}u]==row;
}

fn cm12HotRowTermCell(row:u32,ordinal:u32)->u32{
  if(!cm12HotRowValid(row)||ordinal>=cm12HotRowTermCount(row)){return cm12HotInvalid;}
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  var cell=cm12HotInvalid;
  if(cm12HotRowIsCommonTwoTerm(row)){
    cell=select(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm}u],
      ${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.cell1}u],ordinal==1u);
  }else{
    let term=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm}u]+ordinal;
    cell=${arena}[cm12HotVariableTermBase+${SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS}u*term];
  }
  return select(cell,cm12HotInvalid,cell>=cm12HotCellCount);
}

fn cm12HotRowTermCoefficient(row:u32,ordinal:u32)->f32{
  if(!cm12HotRowValid(row)||ordinal>=cm12HotRowTermCount(row)){return 0.0;}
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  if(cm12HotRowIsCommonTwoTerm(row)){
    return bitcast<f32>(select(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient0}u],
      ${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.coefficient1}u],ordinal==1u));
  }
  let term=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm}u]+ordinal;
  return bitcast<f32>(${arena}[cm12HotVariableTermBase
    +${SPARSE_CM12_HOT_TOPOLOGY_TERM_WORDS}u*term+1u]);
}

fn cm12HotRowDualWeight(row:u32)->f32{
  if(!cm12HotRowValid(row)){return 0.0;}
  return bitcast<f32>(${arena}[cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row
    +${SPARSE_CM12_HOT_TOPOLOGY_ROW.dualWeight}u]);
}
fn cm12HotRowArea(row:u32)->f32{
  if(!cm12HotRowValid(row)){return 0.0;}
  return bitcast<f32>(${arena}[cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row
    +${SPARSE_CM12_HOT_TOPOLOGY_ROW.area}u]);
}
fn cm12HotRowDistance(row:u32)->f32{
  if(!cm12HotRowValid(row)){return 0.0;}
  return bitcast<f32>(${arena}[cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row
    +${SPARSE_CM12_HOT_TOPOLOGY_ROW.distance}u]);
}
fn cm12HotRowExteriorPhi(row:u32)->f32{
  if(!cm12HotRowValid(row)){return 0.0;}
  return bitcast<f32>(${arena}[cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row
    +${SPARSE_CM12_HOT_TOPOLOGY_ROW.exteriorPhi}u]);
}
fn cm12HotRowCenter(row:u32)->vec3f{
  if(!cm12HotRowValid(row)){return vec3f(0.0);}
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  return vec3f(bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.centerX}u]),
    bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.centerY}u]),
    bitcast<f32>(${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.centerZ}u]));
}

fn cm12HotRowRequirement(row:u32,ordinal:u32)->u32{
  if(!cm12HotRowValid(row)){return cm12HotInvalid;}
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  let count=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.requirementCount}u];
  if(ordinal>=count){return cm12HotInvalid;}
  return ${arena}[cm12HotRequirementBase
    +${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.firstRequirement}u]+ordinal];
}

fn cm12HotRowRequirementsMatch(row:u32)->bool{
  if(!cm12HotRowValid(row)){return false;}
  let at=cm12HotRowBase+${SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS}u*row;
  let count=${arena}[at+${SPARSE_CM12_HOT_TOPOLOGY_ROW.requirementCount}u];
  for(var ordinal=0u;ordinal<count;ordinal++){
    let packed=cm12HotRowRequirement(row,ordinal);
    let brick=packed>>5u;let resolution=packed&31u;
    if(brick>=cm12LogicalOwnerResidentCount||resolution==0u
      ||!${brickActive}(brick)||${acceptedResolution}(brick)!=resolution){return false;}
  }
  return true;
}

fn cm12HotIncidenceRange(cell:u32)->vec2u{
  if(!cm12HotHeaderValid()||cell>=cm12HotCellCount){return vec2u(cm12HotInvalid,0u);}
  let first=${arena}[cm12HotIncidenceOffsetBase+cell];
  let end=${arena}[cm12HotIncidenceOffsetBase+cell+1u];
  if(first>end||end>cm12HotIncidenceCount){return vec2u(cm12HotInvalid,0u);}
  return vec2u(first,end-first);
}

// Compact row + term ordinal. Coefficient and tag come from the row packet.
fn cm12HotIncidence(index:u32)->vec2u{
  if(!cm12HotHeaderValid()||index>=cm12HotIncidenceCount){return vec2u(cm12HotInvalid);}
  let at=cm12HotIncidenceBase+${SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS}u*index;
  let record=vec2u(${arena}[at],${arena}[at+1u]);
  if(!cm12HotRowValid(record.x)||record.y>=cm12HotRowTermCount(record.x)){
    return vec2u(cm12HotInvalid);
  }
  return record;
}

fn cm12HotDirectedEdgeRange(cell:u32)->vec2u{
  if(!cm12HotHeaderValid()||cell>=cm12HotCellCount){return vec2u(cm12HotInvalid,0u);}
  let first=${arena}[cm12HotDirectedEdgeOffsetBase+cell];
  let end=${arena}[cm12HotDirectedEdgeOffsetBase+cell+1u];
  if(first>end||end>cm12HotDirectedEdgeCount){return vec2u(cm12HotInvalid,0u);}
  return vec2u(first,end-first);
}

// neighbor, row, pressure-base-weight bits, positive extrapolation-weight bits.
fn cm12HotDirectedEdge(index:u32)->vec4u{
  if(!cm12HotHeaderValid()||index>=cm12HotDirectedEdgeCount){return vec4u(cm12HotInvalid);}
  let at=cm12HotDirectedEdgeBase+${SPARSE_CM12_HOT_TOPOLOGY_EDGE_WORDS}u*index;
  let record=vec4u(${arena}[at],${arena}[at+1u],${arena}[at+2u],${arena}[at+3u]);
  if(record.x>=cm12HotCellCount||!cm12HotRowValid(record.y)
    ||(record.w&0x80000000u)!=0u){return vec4u(cm12HotInvalid);}
  return record;
}
`;
}
