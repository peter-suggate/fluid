/**
 * The V/K readout, written by a shader.
 *
 * The 2-D advance lab writes `fractionReadout` into every cell with the
 * pixels to hold it; this is the same readout for the 3-D slice overlay, which
 * has no canvas to call `fillText` on and no readback to hand one. So the
 * formatting is done per pixel: the fill becomes a short run of glyph codes —
 * indices into {@link FRACTION_READOUT_ALPHABET} — and each code is drawn from a
 * 5×7 bitmap font. Monospace at the lab's own advance, and gated on the lab's
 * own rule (`fractionReadoutRoomPixels`), so a number arrives at the same cell
 * size in both renderers.
 *
 * The formatter is the one part that restates arithmetic, and it restates
 * nothing else: its thresholds are the `fractionViewShaderConstants`
 * names, which the including shader must also carry. The Dawn test
 * `tests/fraction-readout-dawn.test.ts` runs it against `fractionReadout`
 * itself, and draws the font back out against {@link FRACTION_READOUT_FONT}.
 */
import {
  FRACTION_READOUT_ADVANCE_PIXELS, FRACTION_READOUT_ALPHABET, fractionReadoutRoomPixels, wgslDisplayColor,
} from "./fluid-fraction-view";

/** Glyph rows, top first; `#` is ink. One entry per alphabet character. */
export const FRACTION_READOUT_FONT: Readonly<Record<string, readonly string[]>> = Object.freeze({
  "0": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
  "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
  "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
  "3": [".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."],
  "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
  "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
  "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
  "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
  "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
  "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
  ".": [".....", ".....", ".....", ".....", ".....", "..##.", "..##."],
  "e": [".....", ".....", ".###.", "#...#", "#####", "#....", ".###."],
  "-": [".....", ".....", ".....", ".###.", ".....", ".....", "....."],
});

/** One blank column between glyphs, inside the shared advance. */
export const FRACTION_READOUT_GLYPH_WIDTH = FRACTION_READOUT_ADVANCE_PIXELS - 1;
export const FRACTION_READOUT_GLYPH_HEIGHT = 7;
const FRACTION_READOUT_ADVANCE = FRACTION_READOUT_ADVANCE_PIXELS;
/** Glyph codes one packed word holds, four bits each. */
export const FRACTION_READOUT_MAX_GLYPHS = 8;

/**
 * The light the digits are written in: the lab's ink on its dark ground, which
 * is the ground this overlay always has. The casing is what keeps them legible
 * where the amber contour or the overcapacity hatch runs under them.
 */
const READOUT_INK_SWATCH = "#f4eee3";
const READOUT_CASING_SWATCH = "#0b0e10";

/**
 * A glyph as two words: rows 0–3 in the first, 4–6 in the second, five bits a
 * row from bit `5·row`, leftmost column in the row's high bit.
 */
export function packFractionReadoutGlyph(rows: readonly string[]): [number, number] {
  const words: [number, number] = [0, 0];
  rows.forEach((row, index) => {
    let bits = 0;
    for (const pixel of row) bits = (bits << 1) | (pixel === "#" ? 1 : 0);
    const word = index < 4 ? 0 : 1;
    words[word] = (words[word] | (bits << (5 * (index % 4)))) >>> 0;
  });
  return words;
}

const hex = (value: number): string => `0x${value.toString(16)}u`;
const glyphCode = (character: string): string => `${FRACTION_READOUT_ALPHABET.indexOf(character)}u`;

/**
 * `fractionReadoutGlyphs(fill)` formats; `fractionReadoutInk(pixel, centre,
 * glyphs, scale)` says whether a framebuffer pixel is ink (x) or casing (y) of
 * that readout centred on `centre`, at `scale` pixels per font pixel.
 */
export const fractionReadoutShaderLibrary = /* wgsl */ `
// Generated from lib/core/fraction-readout.wgsl.ts. A glyph code is a
// character's index in the readout alphabet, "${FRACTION_READOUT_ALPHABET}".
const READOUT_GLYPH_POINT: u32 = ${glyphCode(".")};
const READOUT_GLYPH_E: u32 = ${glyphCode("e")};
const READOUT_GLYPH_MINUS: u32 = ${glyphCode("-")};
const READOUT_INK_DISPLAY: vec3f = ${wgslDisplayColor(READOUT_INK_SWATCH)};
const READOUT_CASING_DISPLAY: vec3f = ${wgslDisplayColor(READOUT_CASING_SWATCH)};

// fractionReadout(fill) as glyph codes: x packs them four bits apiece, first
// character lowest; y is the count, and 0 means the cell gets no readout.
// Round-half-up to hundredths is toFixed's rule; f32 can land a hair either
// side of a tie that the lab's f64 resolves, which moves the last digit by one.
fn fractionReadoutGlyphs(fill:f32)->vec2u{
  // NaN fails both comparisons, and vacuum is not written, as in the lab.
  if(!(fill>FRACTION_FLOOR)||!(fill<FRACTION_READOUT_CEILING)){return vec2u(0u);}
  if(fill>FRACTION_OVERFULL){
    let hundredths=u32(floor(fill*100.0+0.5));
    var whole=hundredths/100u;var digits=1u;
    for(var probe=whole;probe>=10u;probe/=10u){digits+=1u;}
    var packed=0u;
    for(var at=digits;at>0u;at-=1u){packed|=(whole%10u)<<(4u*(at-1u));whole/=10u;}
    let fraction=hundredths%100u;
    packed|=(READOUT_GLYPH_POINT<<(4u*digits))|((fraction/10u)<<(4u*(digits+1u)))
      |((fraction%10u)<<(4u*(digits+2u)));
    return vec2u(packed,digits+3u);
  }
  if(fill>=FRACTION_READOUT_WHOLE){return vec2u(1u,1u);}
  if(fill>=FRACTION_READOUT_HUNDREDTH){
    let hundredths=u32(floor(fill*100.0+0.5));
    return vec2u(READOUT_GLYPH_POINT|((hundredths/10u)<<4u)|((hundredths%10u)<<8u),3u);
  }
  // 1e-N: Math.round(log10), which rounds a half toward +infinity.
  let decade=u32(-floor(log2(fill)*0.30102999566+0.5));
  return vec2u(1u|(READOUT_GLYPH_E<<4u)|(READOUT_GLYPH_MINUS<<8u)|(decade<<12u),4u);
}

fn fractionReadoutGlyph(code:u32)->vec2u{
  switch code {
${[...FRACTION_READOUT_ALPHABET].map((character, code) => {
  const [low, high] = packFractionReadoutGlyph(FRACTION_READOUT_FONT[character]!);
  return `    case ${code}u: { return vec2u(${hex(low)},${hex(high)}); }`;
}).join("\n")}
    default: { return vec2u(0u); }
  }
}

// Whether the text-local pixel is ink. \`scale\` framebuffer pixels per font pixel.
fn fractionReadoutCovers(glyphs:vec2u,local:vec2i,scale:i32)->bool{
  if(local.x<0||local.y<0){return false;}
  let font=vec2u(local/scale);
  let slot=font.x/${FRACTION_READOUT_ADVANCE}u;let column=font.x%${FRACTION_READOUT_ADVANCE}u;
  if(slot>=glyphs.y||column>=${FRACTION_READOUT_GLYPH_WIDTH}u||font.y>=${FRACTION_READOUT_GLYPH_HEIGHT}u){return false;}
  let rows=fractionReadoutGlyph((glyphs.x>>(4u*slot))&15u);
  let word=select(rows.x,rows.y,font.y>=4u);
  return ((word>>(5u*(font.y%4u)+${FRACTION_READOUT_GLYPH_WIDTH - 1}u-column))&1u)!=0u;
}

// fractionReadoutRoomPixels, at \`scale\` framebuffer pixels per font pixel.
fn fractionReadoutRoom(glyphs:vec2u,scale:i32)->f32{
  return f32(scale)*(f32(glyphs.y)*${FRACTION_READOUT_ADVANCE}.0+${fractionReadoutRoomPixels(0)}.0);
}

// Framebuffer pixel (y down) against a readout centred on \`centre\`: x is ink,
// y is its one-pixel casing. The origin snaps to a whole pixel so every font
// pixel is exactly scale×scale framebuffer pixels and nothing needs filtering.
fn fractionReadoutInk(pixel:vec2f,centre:vec2f,glyphs:vec2u,scale:i32)->vec2f{
  let size=vec2i(i32(glyphs.y)*${FRACTION_READOUT_ADVANCE}-1,${FRACTION_READOUT_GLYPH_HEIGHT})*scale;
  let local=vec2i(floor(pixel))-vec2i(round(centre-0.5*vec2f(size)));
  if(any(local<vec2i(-1))||any(local>size)){return vec2f(0.0);}
  if(fractionReadoutCovers(glyphs,local,scale)){return vec2f(1.0,0.0);}
  for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){
    if(fractionReadoutCovers(glyphs,local+vec2i(dx,dy),scale)){return vec2f(0.0,1.0);}
  }}
  return vec2f(0.0);
}
`;
