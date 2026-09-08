/** CPU oracle: actual analytic fields, no GPU lease or illustrative substitute.
 * OAK_TREE_JSON=... OAK_ANALYTIC_OUT=... OAK_AZIMUTH=... node --import tsx tools/render-oak-analytic.ts
 */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import sharp from "sharp";
import { heroGardenCloudTree } from "../lib/core/hero-garden-tree";
import type { SceneryNode } from "../lib/core/scenery-graph";
import { oakOracleDescriptors } from "./preview/oak-oracle";
import { intersectSvoPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
const out = process.env.OAK_ANALYTIC_OUT ?? "artifacts/oak-v2/analytic.png";
const tree: SceneryNode = process.env.OAK_TREE_JSON ? JSON.parse(readFileSync(process.env.OAK_TREE_JSON, "utf8")) : heroGardenCloudTree();
const descriptors = oakOracleDescriptors(tree, process.env.OAK_BARE === "1");
const width = 640, height = 520, pixelsPerMetre = 390;
const azimuth = Number(process.env.OAK_AZIMUTH ?? .45);
const right = { x: Math.cos(azimuth), y: 0, z: -Math.sin(azimuth) };
const direction = { x: -Math.sin(azimuth), y: 0, z: -Math.cos(azimuth) };
const bytes = Buffer.alloc(width * height * 3, 245);
const depth = new Float64Array(width * height).fill(Infinity);
const masks = new Uint8Array(width * height);
const start = performance.now();
for (const d of descriptors) {
  const r = Math.max(d.lobeRadii_m.x, d.lobeRadii_m.y, d.lobeRadii_m.z);
  const cx = width / 2 + (d.center_m.x * right.x + d.center_m.z * right.z) * pixelsPerMetre;
  const cy = height - 20 - d.center_m.y * pixelsPerMetre;
  const minX = Math.max(0, Math.floor(cx - r * pixelsPerMetre)), maxX = Math.min(width - 1, Math.ceil(cx + r * pixelsPerMetre));
  const minY = Math.max(0, Math.floor(cy - r * pixelsPerMetre)), maxY = Math.min(height - 1, Math.ceil(cy + r * pixelsPerMetre));
  for (let y = minY; y <= maxY; y++)
    for (let x = minX; x <= maxX; x++) {
      const u = (x + .5 - width / 2) / pixelsPerMetre, v = (height - 20 - y - .5) / pixelsPerMetre;
      const hit = intersectSvoPrimitive(d, { origin_m: { x: right.x * u - direction.x * 2, y: v, z: right.z * u - direction.z * 2 }, direction, tMax_m: Math.min(4, depth[y * width + x]) });
      if (!hit || hit.t_m >= depth[y * width + x])
        continue;
      depth[y * width + x] = hit.t_m;
      masks[y * width + x] = d.materialId;
      const diffuse = Math.max(0, hit.normal.x * -.45 + hit.normal.y * .8 + hit.normal.z * .4);
      const value = Math.round(110 + 125 * diffuse);
      const rgb = process.env.OAK_COLOR === "1" ? (d.materialId === 2 ? [.53, .73, .29] : [.61, .44, .29]) : [1, .98, .94];
      for (let c = 0; c < 3; c++)
        bytes[(y * width + x) * 3 + c] = Math.round(value * rgb[c]);
    }
}
mkdirSync(dirname(out), { recursive: true });
await sharp(bytes, { raw: { width, height, channels: 3 } }).png().toFile(out);
let occupied = 0, wood = 0, interiorGaps = 0, interior = 0;
for (let y = 0; y < height; y++) {
  let first = width, last = -1;
  for (let x = 0; x < width; x++) {
    const m = masks[y * width + x];
    if (m) {
      occupied++;
      if (m === 1)
        wood++;
      first = Math.min(first, x);
      last = x;
    }
  }
  if ((height - 20 - y) / pixelsPerMetre > .45)
    for (let x = first; x <= last; x++) {
      interior++;
      if (!masks[y * width + x])
        interiorGaps++;
    }
}
const report = { out, azimuth, primitives: descriptors.length, occupiedPixels: occupied, visibleWoodPixels: wood, crownGapFraction: interiorGaps / interior, elapsed_ms: performance.now() - start };
writeFileSync(out.replace(/\.png$/, ".json"), JSON.stringify(report, null, 2));
console.log(report);
