/** Shared semantic catalog. Order is the composition order, never click order. */
export const VISUAL_LAYERS = [
  { id: "phi", requires: ["phi"], label: "Signed distance", color: "#ed7829", opacity: 0.65, mode: 3, description: "Signed distance in cell widths: blue liquid, white interface, orange air." },
  { id: "density", requires: ["volume"], label: "Surface density", color: "#23a186", opacity: 0.65, mode: 10, description: "Transported volume per cell volume, including dilute residue and overfull cells." },
  { id: "volume", requires: ["volume", "capacity"], label: "Conserved volume", color: "#4f9ae0", opacity: 0.7, mode: 21, description: "Conserved volume divided by open cell capacity; amber marks overcapacity." },
  { id: "pressure", requires: ["pressure"], label: "Pressure", color: "#7e9ee5", opacity: 0.65, mode: 5, description: "Signed pressure: blue positive, amber negative; full intensity at 10 kPa." },
  { id: "tiles", requires: ["tiles"], label: "Work tiles", color: "#3fae8f", opacity: 0.45, mode: 22, description: "Fine velocity tiles, extension shell, and transport reach where published." },
  { id: "pages", requires: ["pages"], label: "Volume pages", color: "#b39bea", opacity: 0.65, mode: 27, description: "Actual transport/sharpening scratch pages. Purple cells own a physical page; empty pages remain transparent. Enable paged volume record storage in SIM." },
  { id: "window", requires: ["window"], label: "Working window", color: "#5fb4e6", opacity: 0.8, mode: 23, description: "Actual dispatched window, seed box, launch slack and clipping. Dense scheduling uses the whole domain." },
  { id: "surface", requires: ["phi"], label: "Liquid surface · φ = 0", color: "#ef9f35", opacity: 0.9, mode: 24, description: "Reconstructed liquid and its zero level-set interface." },
  { id: "grid", requires: ["dimensions"], label: "Grid", color: "#a8c7d8", opacity: 0.7, mode: 0, description: "Cell boundaries, independently of field fills." },
  { id: "velocity", requires: ["velocity"], label: "Velocity", color: "#dce9ee", opacity: 0.9, mode: 26, description: "Cell velocity magnitude and in-plane direction; full scale at 1 m/s." },
  { id: "release", requires: ["releasedFaces"], label: "Released faces", color: "#f5be52", opacity: 1, mode: 25, description: "Solid faces released by the pressure projection." },
] as const;
export type VisualLayerId = typeof VISUAL_LAYERS[number]["id"];
export interface VisualLayerState {
  enabled: VisualLayerId[];
  visible: boolean;
  opacity: Partial<Record<VisualLayerId, number>>;
}
export function visualLayers(enabled: readonly VisualLayerId[] = ["surface"]): VisualLayerState {
  return { enabled: VISUAL_LAYERS.filter(l => enabled.includes(l.id)).map(l => l.id), visible: true, opacity: {} };
}
export function layerOpacity(state: VisualLayerState, id: VisualLayerId): number {
  return state.opacity[id] ?? VISUAL_LAYERS.find(l => l.id === id)!.opacity;
}
export function toggleVisualLayer(state: VisualLayerState, id: VisualLayerId): VisualLayerState {
  return { ...state, visible: true, enabled: VISUAL_LAYERS.filter(l => l.id === id ? !state.enabled.includes(id) : state.enabled.includes(l.id)).map(l => l.id) };
}
export function legacyVisualLayers(mode: string, grid = false): VisualLayerState {
  const aliases: Record<string, VisualLayerId[]> = {
    // Uniform defaults omit the lattice; it remains an explicit layer toggle.
    structure: ["surface"], "volume-levelset": ["volume", "surface"],
    "fine-tiles": ["tiles", "surface"], "solve-window": ["window", "surface"],
    speed: ["velocity"], "face-velocity": ["velocity"], release: ["surface", "release"],
  };
  return visualLayers([...((Object.hasOwn(aliases, mode) ? aliases[mode] : undefined) ?? VISUAL_LAYERS.filter(l => l.id === mode).map(l => l.id)), ...(grid ? ["grid" as const] : [])]);
}
export function readVisualLayers(raw: string | null, fallback = visualLayers()): VisualLayerState {
  if (raw === null) return fallback;
  try {
    const value = JSON.parse(raw);
    if (!Array.isArray(value.enabled)) return fallback;
    const state = visualLayers(value.enabled);
    state.visible = value.visible !== false;
    for (const layer of VISUAL_LAYERS) {
      const n = value.opacity?.[layer.id];
      if (typeof n === "number" && Number.isFinite(n)) state.opacity[layer.id] = Math.max(0, Math.min(1, n));
    }
    return state;
  } catch { return fallback; }
}
export const writeVisualLayers = (state: VisualLayerState): string => JSON.stringify(state);

/** Display-space ramps, shared with WGSL; physical scales don't jump per frame. */
export const LAYER_PALETTE = {
  liquid: [79, 154, 224], empty: [245, 245, 230], positive: [126, 158, 229], negative: [221, 153, 85],
  phiLiquid: [26, 115, 235], phiAir: [237, 120, 41], residue: [70, 50, 126], dilute: [35, 161, 134],
  densityLiquid: [126, 163, 193], excess: [201, 86, 63], fine: [63, 174, 143], shell: [166, 216, 198], transport: [136, 100, 46],
} as const;
export const LAYER_PRESSURE_SCALE = 10_000;
export const LAYER_SPEED_SCALE = 1;
export function scalarLayerPaint(id: VisualLayerId, value: number): { color: readonly number[]; alpha: number } {
  const p = LAYER_PALETTE;
  const mix = (a: readonly number[], b: readonly number[], t: number) => a.map((v, i) => v + (b[i]! - v) * Math.max(0, Math.min(1, t)));
  switch (id) {
    case "phi": return { color: mix(p.empty, value < 0 ? p.phiLiquid : p.phiAir, Math.abs(value) / 4), alpha: 0.8 };
    case "pressure": return { color: value < 0 ? p.negative : p.positive, alpha: Math.min(1, Math.abs(value) / LAYER_PRESSURE_SCALE) };
    case "density": return { color: value > 1 ? p.excess : value >= 0.5 ? p.densityLiquid : mix(p.residue, p.dilute, (Math.log10(Math.max(value, 1e-6)) + 6) / Math.log10(500000)), alpha: value > 1e-6 ? 0.85 : 0 };
    case "volume": return { color: p.liquid, alpha: Math.max(0, Math.min(1, value)) };
    case "tiles": return { color: value & 1 ? p.fine : value & 2 ? p.shell : p.transport, alpha: value & 1 ? 0.66 : value & 2 ? 0.3 : value & 4 ? 0.35 : 0 };
    case "velocity": return { color: p.liquid, alpha: Math.min(1, Math.max(0, value) / LAYER_SPEED_SCALE) * 0.25 };
    default: return { color: p.liquid, alpha: 1 };
  }
}
export const visualLayerPaintWGSL = `
${Object.entries(LAYER_PALETTE).map(([key, rgb]) => `const LP_${key}:vec3f=vec3f(${rgb.map(n => (n / 255).toFixed(8)).join(",")});`).join("\n")}
fn scalarLayerPaint(mode:i32,value:f32)->vec4f {
  if(mode==3){return vec4f(mix(LP_empty,select(LP_phiAir,LP_phiLiquid,value<0.0),clamp(abs(value)/4.0,0.0,1.0)),0.8);}
  if(mode==5){return vec4f(select(LP_positive,LP_negative,value<0.0),min(1.0,abs(value)/${LAYER_PRESSURE_SCALE}.0));}
  if(mode==10){let dilute=mix(LP_residue,LP_dilute,clamp((log2(max(value,1e-6))/log2(10.0)+6.0)/log2(500000.0)*log2(10.0),0.0,1.0));return vec4f(select(select(dilute,LP_densityLiquid,value>=0.5),LP_excess,value>1.0),select(0.0,0.85,value>1e-6));}
  if(mode==21){return vec4f(LP_liquid,clamp(value,0.0,1.0));}
  if(mode==22){let bits=u32(value);return vec4f(select(select(LP_transport,LP_shell,(bits&2u)!=0u),LP_fine,(bits&1u)!=0u),select(select(select(0.0,0.35,(bits&4u)!=0u),0.3,(bits&2u)!=0u),0.66,(bits&1u)!=0u));}
  return vec4f(LP_liquid,min(1.0,max(0.0,value)/${LAYER_SPEED_SCALE}.0)*0.25);
}
`;
