import { UNIFORM_STAGE_REASON as R } from "../methods/uniform/uniform-stage-grids";

/** Shared semantic catalog. Order is the composition order, never click order. */
export const VISUAL_LAYERS = [
  { id: "phi", requires: ["phi"], label: "Signed distance", color: "#ed7829", opacity: 0.65, mode: 3, description: "Signed distance in cell widths: blue liquid, white interface, orange air. Mixed Uniform keeps it at owner resolution: h vertices in h tiles, 4h corners under 4h owners (interpolated between), as in an all-4h run." },
  { id: "volume", requires: ["volume"], label: "Liquid volume", color: "#4f9ae0", opacity: 0.7, mode: 21, description: "Transported volume over the open capacity of each cell (the cell volume where no capacity is published): dilute residue on a log ramp from purple to green, liquid blue, overfull cells red. Opacity follows fullness with a floor so residue stays visible. Mixed Uniform draws the h and 4h bulk cells the volume lives on, with orange resolution seams." },
  { id: "pressure", requires: ["pressure"], label: "Pressure", color: "#7e9ee5", opacity: 0.65, mode: 5, description: "Signed pressure (blue positive, amber negative; full intensity at 10 kPa) on the exact cells the last solve ran on. Mixed Uniform draws each pressure cell at its own size: h cells (the two-stage h band re-solved over the 4h solve, with the band's own pressure) are tinted green, 4h cells untinted, with resolution seams in lilac. Tiles where the level set crosses zero are outlined in yellow; under dynamic coarsening every such tile should be green, so one held at 4h (a surface the head census missed) is outlined in magenta." },
  { id: "tiles", requires: ["tiles"], label: "Work tiles", color: "#3fae8f", opacity: 0.45, mode: 22, description: "Fine velocity tiles, extension shell, and transport reach where published. Mixed Uniform draws each tile's live h or 4h cells and colours every h tile by why it is h, the first rule the frame head's census fired: crossing, closure (shared vertices at dt = 0), travel within this frame's departure, source, host join, solid or body promotion, or static (authored / host layout). 4h tiles the census traced without banding are tinted faintly; untinted 4h tiles were skipped by the O(1) reach test. Hatched h tiles take the general velocity sampler because a 4h tile lies inside their signed reach box (cross-hatched: reach saturated); unhatched h tiles are certified regular. A crossing tile held at 4h is magenta." },
  { id: "pages", requires: ["pages"], label: "Domain pages", color: "#b39bea", opacity: 0.65, mode: 27, description: "Page states from the last step: teal = transport active; amber = sharpening only; faint purple = resident without volume work. Absent pages are hidden. Residency can include pressure/interface support; authored pages currently remain allocated." },
  { id: "window", requires: ["window"], label: "Working window", color: "#5fb4e6", opacity: 0.8, mode: 23, description: "Actual dispatched window, seed box, launch slack and clipping. Dense scheduling uses the whole domain." },
  { id: "surface", requires: ["phi"], label: "Liquid surface · φ = 0", color: "#ef9f35", opacity: 0.9, mode: 24, description: "Reconstructed liquid and its zero level-set interface. Mixed Uniform colours the interface by the velocity that advected it: teal where all three components were h velocity (h bulk, or retained and extended projected h velocity), amber where the 4h bulk sampler moved it, blended where only some components were h." },
  { id: "grid", requires: ["dimensions"], label: "Grid", color: "#a8c7d8", opacity: 0.7, mode: 0, description: "Represented cell boundaries, independently of field fills. Mixed Uniform draws its live h and 4h bulk owners with orange resolution seams, and hatches the tiles this frame's head relayout changed: orange refined to h, blue coarsened to 4h (a host relayout after the last transport hatches orange)." },
  { id: "velocity", requires: ["velocity"], label: "Velocity", color: "#dce9ee", opacity: 0.9, mode: 26, description: "Cell velocity magnitude and in-plane direction; full scale at 1 m/s. Mixed Uniform draws one arrow per h or 4h bulk owner, the grid momentum is advected on." },
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
    // Surface density and conserved volume are one liquid volume layer.
    density: ["volume"],
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
  surfaceBand: [245, 226, 122], overlayValid: [63, 193, 201], overlayFallback: [214, 132, 44],
  // Mixed Uniform band reasons (tiles layer), relayout marks (grid layer) and
  // the fault colour (a surface the layout held at 4h).
  reasonCrossing: [63, 174, 143], reasonClosure: [104, 136, 214], reasonTravel: [232, 170, 60],
  reasonJoin: [152, 110, 212], reasonSource: [72, 186, 232], reasonSolid: [196, 116, 78],
  reasonStatic: [168, 168, 160], reasonTraced: [146, 156, 170], coarsened: [92, 162, 236], fault: [255, 20, 140],
} as const;

/** The tiles layer's key on mixed Uniform: why each h tile is h (first rule
 * the frame head's census fired; UNIFORM_STAGE_REASON), plus the sampler
 * certificate hatch. Swatches are the WGSL colours (umReasonPaint). */
const rgb = (c: readonly number[]) => `rgb(${c.join(",")})`;
export const MIXED_TILE_LEGEND = [
  { color: rgb(LAYER_PALETTE.reasonCrossing), label: "Crossing", title: "The level set crosses the tile: the tiles the surface needs." },
  { color: rgb(LAYER_PALETTE.reasonClosure), label: "Closure", title: "Touches a crossing owner's face, edge or corner: shared vertices need it at h even at dt = 0." },
  { color: rgb(LAYER_PALETTE.reasonTravel), label: "Travel", title: "The surface can reach it within this frame's departure." },
  { color: rgb(LAYER_PALETTE.reasonSource), label: "Source", title: "This frame's drop or inflow plug can fill it." },
  { color: rgb(LAYER_PALETTE.reasonJoin), label: "Join", title: "Joined by the host: a solid edit or a body pose." },
  { color: rgb(LAYER_PALETTE.reasonSolid), label: "Solid", title: "Liquid-conditional solid or rigid-body promotion." },
  { color: rgb(LAYER_PALETTE.reasonStatic), label: "Static", title: "h without a band reason: an authored fine region or the host layout." },
  { color: rgb(LAYER_PALETTE.reasonTraced), label: "Traced 4h", title: "A 4h tile the census traced and did not band (untinted 4h tiles were skipped in O(1))." },
  { color: rgb(LAYER_PALETTE.fault), label: "Crossing at 4h", title: "A crossing tile the layout held at 4h: a coarse-only region, or a fault." },
  { color: "repeating-linear-gradient(135deg,#333 0 2px,transparent 2px 5px)", label: "General sampler", title: "Hatched h tiles take the general velocity sampler: a 4h tile lies inside their signed reach box (cross-hatched: the reach saturated). Unhatched h tiles are certified regular." },
] as const;
export const LAYER_PRESSURE_SCALE = 10_000;
export const LAYER_SPEED_SCALE = 1;
export function scalarLayerPaint(id: VisualLayerId, value: number): { color: readonly number[]; alpha: number } {
  const p = LAYER_PALETTE;
  const mix = (a: readonly number[], b: readonly number[], t: number) => a.map((v, i) => v + (b[i]! - v) * Math.max(0, Math.min(1, t)));
  switch (id) {
    case "phi": return { color: mix(p.empty, value < 0 ? p.phiLiquid : p.phiAir, Math.abs(value) / 4), alpha: 0.8 };
    case "pressure": return { color: value < 0 ? p.negative : p.positive, alpha: Math.min(1, Math.abs(value) / LAYER_PRESSURE_SCALE) };
    case "volume": return { color: value > 1 ? p.excess : value >= 0.5 ? p.densityLiquid : mix(p.residue, p.dilute, (Math.log10(Math.max(value, 1e-6)) + 6) / Math.log10(500000)), alpha: value > 1e-6 ? Math.max(0.35, Math.min(1, value)) : 0 };
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
  if(mode==21){let dilute=mix(LP_residue,LP_dilute,clamp((log2(max(value,1e-6))/log2(10.0)+6.0)/log2(500000.0)*log2(10.0),0.0,1.0));return vec4f(select(select(dilute,LP_densityLiquid,value>=0.5),LP_excess,value>1.0),select(0.0,clamp(value,0.35,1.0),value>1e-6));}
  if(mode==22){let bits=u32(value);return vec4f(select(select(LP_transport,LP_shell,(bits&2u)!=0u),LP_fine,(bits&1u)!=0u),select(select(select(0.0,0.35,(bits&4u)!=0u),0.3,(bits&2u)!=0u),0.66,(bits&1u)!=0u));}
  return vec4f(LP_liquid,min(1.0,max(0.0,value)/${LAYER_SPEED_SCALE}.0)*0.25);
}
// Mixed Uniform band reason (UNIFORM_STAGE_REASON) of a tile, h or 4h.
fn umReasonPaint(reason:u32,h:bool)->vec4f{
  var c=LP_reasonStatic;
  switch(reason){
    case ${R.crossing}u:{c=LP_reasonCrossing;}
    case ${R.closure}u:{c=LP_reasonClosure;}
    case ${R.travel}u:{c=LP_reasonTravel;}
    case ${R.join}u:{c=LP_reasonJoin;}
    case ${R.source}u:{c=LP_reasonSource;}
    case ${R.solid}u:{c=LP_reasonSolid;}
    default:{}
  }
  let band=reason>=${R.crossing}u;
  if(h){return vec4f(c,select(0.45,0.62,band));}
  if(reason==${R.crossing}u){return vec4f(LP_fault,0.85);}
  if(band){return vec4f(c,0.2);}
  return vec4f(LP_reasonTraced,select(0.0,0.16,reason==${R.traced}u));
}
`;
