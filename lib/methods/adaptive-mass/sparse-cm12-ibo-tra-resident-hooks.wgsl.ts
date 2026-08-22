const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Resident adapters for stable IBO packets and direct two-phase gamma scatter. */
export function createSparseCM12IboTRAResidentHooksWGSL(options: Readonly<{
  iboPrefix?: string;
  residentPrefix?: string;
  arenaName?: string;
}> = {}): string {
  const ibo = identifier(options.iboPrefix ?? "cm12Resident", "ITR resident IBO prefix");
  const resident = identifier(options.residentPrefix ?? "cm12Resident",
    "ITR resident hook prefix");
  const arena = identifier(options.arenaName ?? "topologyArena", "ITR resident arena");
  return /* wgsl */ `
fn ${resident}DynamicClosureGeneration()->u32{
  return ${ibo}IBOAcceptedGeneration();
}
fn ${resident}DynamicClosureSurfaceMask(packet:u32)->vec2u{
  if(!tpm1PacketReady(packet)){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[TPM1_SURFACE_LOW+packet]),
    atomicLoad(&${arena}[TPM1_SURFACE_HIGH+packet]));
}
fn ${resident}DynamicClosureDensityMask(packet:u32)->vec2u{
  if(!tpm1PacketReady(packet)){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[TPM1_DENSITY_LOW+packet]),
    atomicLoad(&${arena}[TPM1_DENSITY_HIGH+packet]));
}
fn ${resident}IBOTRAPacketDescriptor(packet:u32,slot:u32)->vec4u{
  let leaf=${ibo}IBOStablePacketLeaf(packet);
  if(!${ibo}IBOLeafActive(slot,leaf)){return vec4u(0xffffffffu);}
  let descriptor=${ibo}IBOLeafDescriptorId(slot,leaf);
  let resolution=${ibo}IBOCanonicalWord(descriptor,2u);
  let dimensions=${ibo}IBOCanonicalWord(descriptor,4u);
  return vec4u(leaf,resolution,dimensions,${ibo}IBOStablePacketLocal(packet));
}
fn ${resident}IBOTRAPacketLocal(packet:u32,lane:u32,slot:u32)->vec4u{
  let descriptor=${resident}IBOTRAPacketDescriptor(packet,slot);
  if(descriptor.x==0xffffffffu||lane>=64u){return vec4u(0xffffffffu);}
  let packetAxis=max(1u,(descriptor.y+3u)/4u);let localPacket=descriptor.w;
  if(localPacket>=packetAxis*packetAxis*packetAxis){return vec4u(0xffffffffu);}
  let pz=localPacket/(packetAxis*packetAxis);
  let remainder=localPacket-pz*packetAxis*packetAxis;
  let py=remainder/packetAxis;let px=remainder-py*packetAxis;
  let local=4u*vec3u(px,py,pz)+vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  let dimensions=vec3u(descriptor.z&1023u,(descriptor.z>>10u)&1023u,
    (descriptor.z>>20u)&1023u);
  return select(vec4u(local,descriptor.x),vec4u(0xffffffffu),any(local>=dimensions));
}
fn ${resident}IBOTRAPacketForLocal(leaf:u32,local:vec3u,slot:u32)->vec2u{
  if(!${ibo}IBOLeafActive(slot,leaf)){return vec2u(0xffffffffu);}
  let descriptor=${ibo}IBOLeafDescriptorId(slot,leaf);
  let resolution=${ibo}IBOCanonicalWord(descriptor,2u);
  let packed=${ibo}IBOCanonicalWord(descriptor,4u);
  let dimensions=vec3u(packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u);
  if(any(local>=dimensions)){return vec2u(0xffffffffu);}
  let packetAxis=max(1u,(resolution+3u)/4u);let packetLocal=local/4u;
  let packet=leaf*IBO1_PACKETS_PER_LEAF+packetLocal.x
    +packetAxis*(packetLocal.y+packetAxis*packetLocal.z);
  let lane=(local.x&3u)+4u*(local.y&3u)+16u*(local.z&3u);
  return vec2u(packet,lane);
}
fn ${resident}IBOTRAScatterSnapshot(row:u32){
  scatterGammaRow(row,destinationDensity(),destinationGamma());
}
fn ${resident}IBOTRAScatterRefinement(row:u32){
  scatterGammaRow(row,p.stateOffsets2.x,p.stateOffsets2.y);
}
`;
}
