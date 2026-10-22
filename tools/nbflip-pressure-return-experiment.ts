/**
 * Experimental dead-output elimination in the pressure return transform.
 * This module is deliberately not imported by the product. It rewrites the
 * transfer's WGSL before compilation; the experiment supplies a read-only
 * view of the current pressure arena's tile-slot map.
 *
 * Let B be the current pressure-band tiles, R the coarse-to-simulation return,
 * C the band's pCopy, and P its pProject. The candidate R' leaves B untouched:
 *
 *   P(C(R(u)), pool) = P(C(R'(u)), pool).
 *
 * pProject writes all 64 texels and low-domain boundary entries of each B tile
 * from its pressure pool. pCopy and pProject's preserved-component reads only
 * address texels outside B. No consumer between R and P reads B. Arithmetic
 * for every surviving value is unchanged. The comparison point is after
 * pProject, never immediately after R.
 */
export interface PressureDirectTransferOptions {
 /** Byte offset returned by UniformPressureArena.slotMapOffset. */
 readonly slotMapOffset:number;
 /** An unused binding in the transfer resources group. Defaults to 8. */
 readonly binding?:number;
 /** Transfer resources group, normally 1. */
 readonly group?:number;
}

function replaceExactly(source:string,before:string,after:string):string {
 if(source.split(before).length!==2)throw new Error(`Pressure direct transfer: expected exactly one source fragment: ${before}`);
 return source.replace(before,after);
}

export function pressureDirectTransferWGSL(source:string,options:PressureDirectTransferOptions):string {
 const group=options.group??1,binding=options.binding??8;
 if(!Number.isInteger(options.slotMapOffset)||options.slotMapOffset<0||options.slotMapOffset%4!==0)throw new Error("Pressure direct transfer: slot map must have a nonnegative, word-aligned byte offset");
 if(!Number.isInteger(group)||group<0||!Number.isInteger(binding)||binding<0)throw new Error("Pressure direct transfer: invalid group or binding");
 const collision=new RegExp(`@group\\(\\s*${group}\\s*\\)\\s*@binding\\(\\s*${binding}\\s*\\)`);
 if(collision.test(source))throw new Error(`Pressure direct transfer: group ${group} binding ${binding} already exists`);
 // This replacement is intentionally confined to the fine return entry. The
 // coarse path, restriction, and all freeze/rejection checks remain intact.
 source=replaceExactly(source,
  "if(!umFrozenRegion(umPageOfTile(tile))){tFineToSimulation(tile,lane);}",
  "if(!umFrozenRegion(umPageOfTile(tile))&&pdBandIndex[PD_SLOT_WORD+tile]==0u){tFineToSimulation(tile,lane);}");
 return `${source}\n// Current-frame band membership; read only after the pressure verdict.\n@group(${group}) @binding(${binding}) var<storage,read> pdBandIndex:array<u32>;\nconst PD_SLOT_WORD:u32=${options.slotMapOffset/4}u;\n`;
}

/** Add this to the exact resources layout used by the candidate pipeline. */
export function pressureDirectTransferLayoutEntry(binding=8):GPUBindGroupLayoutEntry {
 return {binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}};
}

/** Bind the whole current-frame arena index; the shader applies slotMapOffset. */
export function pressureDirectTransferBinding(index:GPUBuffer,binding=8):GPUBindGroupEntry {
 return {binding,resource:{buffer:index}};
}

export const pressureDirectTransferContract={
 entryPoint:"toSimulationFine",
 compareAfter:"pProject",
 requiredChain:["toSimulationCoarse","toSimulationFine","pCopy","pProject"],
 source:"uniform-mixed-remap.ts:tFineToSimulation/toSimulationFine",
 killedWrites:"All fine velocity texels and negative-domain boundary entries on current band tiles",
 survivingReads:"pCopy and pProject preserve components only at non-band texels",
 unchanged:["Root restriction","Root pressure solve","Root projection","Coarse return","Frozen tiles","Frame-failure return","Band projection and publication"],
 caution:[
  "The current arena index must match this frame and stay immutable through return/projection.",
  "The caller must append the index layout/binding only to the candidate resources group.",
  "Compare physical velocity storage and negative boundary buffers after pProject; compare status/presentation/warm state too.",
  "Do not compare the intentionally unmaterialized intermediate band texels immediately after transfer.",
  "This removes a subset of a historically 0.262 ms pass; no large speedup claim follows.",
 ],
} as const;
