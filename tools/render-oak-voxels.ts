/** Independent centre-sampled voxel silhouette study. Uses the production
 * analytic field sign; this CPU diagnostic does not model GPU conservative
 * coverage, lighting, or screen-space LOD. Front view, identical camera/shape.
 */
import sharp from "sharp";
import { mkdirSync, writeFileSync } from "node:fs";
import { heroGardenCloudTree } from "../lib/core/hero-garden-tree";
import { sampleSvoPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
import { oakOracleDescriptors } from "./preview/oak-oracle";
const descriptors = oakOracleDescriptors(heroGardenCloudTree());
const width = 560, height = 480, ppm = 390;
mkdirSync("artifacts/oak-v2/voxels", { recursive: true });
for (const level of [0, 1, 2, 3]) {
  const h = .00625 / 2 ** level, start = performance.now();
  const zbuffer = new Float64Array(width * height).fill(-Infinity), material = new Uint8Array(width * height);
  for (const d of descriptors) {
    const c = d.center_m, r = d.lobeRadii_m;
    const minX = Math.max(0, Math.floor(width / 2 + (c.x - r.x) * ppm)), maxX = Math.min(width - 1, Math.ceil(width / 2 + (c.x + r.x) * ppm));
    const minY = Math.max(0, Math.floor(height - 20 - (c.y + r.y) * ppm)), maxY = Math.min(height - 1, Math.ceil(height - 20 - (c.y - r.y) * ppm));
    const cache = new Map<number, number>();
    for (let y = minY; y <= maxY; y++)
      for (let x = minX; x <= maxX; x++) {
        const ix = Math.floor(((x + .5 - width / 2) / ppm) / h), iy = Math.floor(((height - 20 - y - .5) / ppm) / h);
        const key = (ix + 4096) + (iy + 4096) * 8192;
        let front = cache.get(key);
        if (front === undefined) {
          front = -Infinity;
          const px = (ix + .5) * h, py = (iy + .5) * h;
          const normalizedXY = ((px - c.x) / r.x) ** 2 + ((py - c.y) / r.y) ** 2;
          if (normalizedXY <= 1) {
            const dz = r.z * Math.sqrt(1 - normalizedXY);
            const hi = Math.floor((c.z + dz) / h), lo = Math.floor((c.z - dz) / h);
            for (let iz = hi; iz >= lo; iz--) {
              if (sampleSvoPrimitive(d, { x: px, y: py, z: (iz + .5) * h }).signedDistance_m < 0) {
                front = (iz + 1) * h;
                break;
              }
            }
          }
          cache.set(key, front);
        }
        if (front > zbuffer[y * width + x]) {
          zbuffer[y * width + x] = front;
          material[y * width + x] = d.materialId;
        }
      }
  }
  const bytes = Buffer.alloc(width * height * 3, 245);
  let occupied = 0;
  for (let i = 0; i < material.length; i++)
    if (material[i]) {
      occupied++;
      const value = 165 + 60 * Math.max(-1, Math.min(1, zbuffer[i]));
      const rgb = material[i] === 2 ? [.53, .73, .29] : [.61, .44, .29];
      for (let c = 0; c < 3; c++)
        bytes[3 * i + c] = Math.round(value * rgb[c]);
    }
  const out = `artifacts/oak-v2/voxels/depth-${level}.png`;
  await sharp(bytes, { raw: { width, height, channels: 3 } }).png().toFile(out);
  const report = { depth: level, cell_m: h, occupiedPixels: occupied, elapsed_ms: performance.now() - start };
  writeFileSync(out.replace(/png$/, "json"), JSON.stringify(report, null, 2));
  console.log(report);
}
