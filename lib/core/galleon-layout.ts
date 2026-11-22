// Blender Y-up export: the bowsprit points along +Z and the waterline is Y=0.
export const galleonAsset = "sailing-galleon";
export const galleonOcean = { width_m: 160, height_m: 48, depth_m: 160, cell_m: 1, waterline_m: 8 };
export const galleons = [
  { id: "jackdaw", x: -15, z: 5, heading: -1 },
  { id: "kestrel", x: 15, z: -4, heading: 1 },
] as const;
