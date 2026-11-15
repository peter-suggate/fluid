/**
 * The layout-neutral form of a frozen census (the old NB-FLIP engine's
 * "Uniform dynamic ownership census" pass, tools/nbflip-census-capture.ts),
 * shared by the capture tool and brick-census-dam.test.ts.
 *
 * A window of W^3 cells at the lattice's low corner, T = W/4 tiles an axis,
 * tile index x + T*(y + T*z). Pre-census, what the census reads:
 * - per tile: the owner (1 = h, 4 = 4h), its region's census mode (0 the
 *   census did not visit it: not a resident block of the pool; 1 replayed
 *   deep COLD records; 2 classified), the particle join bit (the census's
 *   umJoined), the deep COLD bit of its region and decide's band bit of the
 *   last census (CC_DECIDED, the decidedPrev a deep COLD tile keeps);
 * - φ (metres) at the stored vertices and the extension velocity (the
 *   census's velocity texture) at the stored faces, in remap-fixture.ts's
 *   storedVertices / storedFaces order under the owners. φ at a 4-aligned vertex of an h tile's
 *   closure is the plain (h store) load; where the old base (the canonical load, umLoadCorner)
 *   disagrees with it, split lists the vertex (an index into phi) and splitPhi the base value.
 * Post-census, per tile: decide's band bit (CC_DECIDED, before solid
 * promotion and before any layout limit), the band bit (census word 20+),
 * importance words 0 and 1 (word 1 bits 16..21 are the triggers), the six
 * velocity keys, the two crossing-cell words and the gap word.
 */
export interface CensusFixtureMeta {
  readonly scene: string;
  /** The frame whose census was captured (1-based, as solver.advanceTo(step * dt)), and its pass in the frame. */
  readonly step: number;
  readonly pass: number;
  readonly window: number;
  readonly lattice: readonly number[];
  /** Cell size (m), the census's dt (s, policy word 0) and gravity (m/s^2). */
  readonly h: number;
  readonly dt: number;
  readonly gravity: readonly number[];
  /** census[18]: the largest tile travel over the whole old lattice, h per step (f32 bits). */
  readonly reachBits: number;
  /** The census header after the pass (census words 0..19: 0 crossing, 2 band, 4 refined, 12 required tiles, 18 the reach). */
  readonly header: readonly number[];
  /** The old census policy as encoded (uniform-mixed-dynamic.ts encode): the 28 words. */
  readonly policy: readonly number[];
  /** Counts over the whole lattice and the window. */
  readonly counts: Readonly<Record<string, number>>;
  readonly sections: Readonly<Record<string, { readonly offset: number; readonly count: number; readonly type: "u8" | "u32" | "f32" }>>;
  readonly notes: readonly string[];
}

export const CENSUS_SECTIONS = ["owner", "mode", "joined", "deep", "decidedPrev", "phi", "US", "decided", "band", "w0", "w1", "keys", "cross", "gap", "split", "splitPhi"] as const;
export type CensusSection = (typeof CENSUS_SECTIONS)[number];

export function encodeCensusFixture(parts: Record<CensusSection, Uint8Array | Uint32Array | Float32Array>): { bin: Uint8Array; sections: CensusFixtureMeta["sections"] } {
  let at = 0; const sections: Record<string, { offset: number; count: number; type: "u8" | "u32" | "f32" }> = {};
  const chunks: Uint8Array[] = [];
  for (const name of CENSUS_SECTIONS) {
    const v = parts[name], type = v instanceof Uint8Array ? "u8" : v instanceof Uint32Array ? "u32" : "f32";
    const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength), pad = (4 - (bytes.length & 3)) & 3;
    sections[name] = { offset: at, count: v.length, type };
    chunks.push(bytes, new Uint8Array(pad)); at += bytes.length + pad;
  }
  const bin = new Uint8Array(at); let o = 0; for (const c of chunks) { bin.set(c, o); o += c.length; }
  return { bin, sections };
}

export function decodeCensusFixture(bin: Uint8Array, meta: CensusFixtureMeta) {
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const u8 = (n: CensusSection) => { const s = meta.sections[n]!; return new Uint8Array(buf, s.offset, s.count); };
  const u32 = (n: CensusSection) => { const s = meta.sections[n]!; return new Uint32Array(buf, s.offset, s.count); };
  const f32 = (n: CensusSection) => { const s = meta.sections[n]!; return new Float32Array(buf, s.offset, s.count); };
  return {
    owner: u8("owner"), mode: u8("mode"), joined: u8("joined"), deep: u8("deep"), decidedPrev: u8("decidedPrev"), phi: f32("phi"), US: f32("US"),
    decided: u8("decided"), band: u8("band"), w0: u32("w0"), w1: u32("w1"), keys: u32("keys"), cross: u32("cross"), gap: u32("gap"), split: u32("split"), splitPhi: f32("splitPhi"),
  };
}
