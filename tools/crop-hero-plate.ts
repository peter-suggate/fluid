#!/usr/bin/env node
/**
 * Cut the reference plate into the pieces geometry work is judged against.
 *
 *   npm run plate:crops
 *
 * The hero garden's geometry is authored *from the photograph*, not from a
 * description of it, and the difference matters more than it sounds. Two
 * examples already found by looking rather than reading:
 *
 *   - `docs/hero-fidelity-1000x-handoff.md` §2.2 calls fracture "the strongest
 *     single 'this is stone' cue". The plate's boulders have **no** fracture:
 *     they are water-worn porcelain domes with soft rims and no arris anywhere.
 *     Authoring fracture into this set would take it further from the reference
 *     while ticking a box on the plan.
 *   - The rim reads as a fat circular tube in the render. In the plate it is a
 *     broad flattened bullnose, distinctly wider than tall. That proportion is
 *     worth more than any surface detail on it, and no aggregate metric was
 *     ever going to say so.
 *
 * Crops are written at the plate's native resolution — no downscale — because
 * the whole point is to see the grain, the arris and the floret. They land in
 * `artifacts/`, which is gitignored, so this tool exists to make them
 * reproducible rather than precious: the plate itself is tracked, and these are
 * a pure function of it.
 *
 * Regions are normalised so they survive the plate being re-rendered at another
 * size. Add one here when a new part of the set comes up for work.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { decodeRgbPng, encodeRgbPng } from "../lib/core/png-codec";

const platePath = process.env.FLUID_HERO_PLATE ?? "output/imagegen/garden-pond-hose-fill-simplified.png";
const outDirectory = process.env.FLUID_HERO_PLATE_CROPS ?? "artifacts/plate-crops";

/** `[x0, y0, x1, y1]` in normalised plate coordinates, with what each is for. */
const CROPS: Readonly<Record<string, { readonly box: readonly [number, number, number, number]; readonly subject: string }>> =
  Object.freeze({
    coping: {
      box: [0.08, 0.74, 0.62, 1.0],
      subject: "the rim's section — a broad flattened bullnose, wider than tall, crisp inner lip",
    },
    boulders: {
      box: [0.01, 0.11, 0.36, 0.52],
      subject: "capped stones — oblate domes on tapered stems, no fracture anywhere",
    },
    pebbles: {
      box: [0.72, 0.4, 1.0, 0.9],
      subject: "the pebble course — individual stones with gaps and contact shadows, not a merged blob",
    },
    canopy: {
      box: [0.38, 0.0, 0.88, 0.24],
      subject: "the floret cloud — three self-similar scales, and the coral limbs beneath it",
    },
    stepping: {
      box: [0.08, 0.44, 0.46, 0.86],
      subject: "stepping discs — thick coins with rounded edges, half in the water",
    },
    ground: {
      box: [0.0, 0.0, 0.32, 0.26],
      subject: "the backdrop sweep — a bounded plaster cove with no corner line",
    },
  });

const plate = decodeRgbPng(readFileSync(platePath));
mkdirSync(outDirectory, { recursive: true });

for (const [name, { box, subject }] of Object.entries(CROPS)) {
  const originX = Math.max(0, Math.round(box[0] * plate.width));
  const originY = Math.max(0, Math.round(box[1] * plate.height));
  const width = Math.min(plate.width - originX, Math.round(box[2] * plate.width) - originX);
  const height = Math.min(plate.height - originY, Math.round(box[3] * plate.height) - originY);
  const rgb = new Uint8Array(width * height * 3);
  for (let row = 0; row < height; row += 1) {
    const source = ((row + originY) * plate.width + originX) * 3;
    rgb.set(plate.rgb.subarray(source, source + width * 3), row * width * 3);
  }
  const outPath = path.join(outDirectory, `${name}.png`);
  writeFileSync(outPath, encodeRgbPng({ width, height, rgb }));
  process.stdout.write(`${name.padEnd(10)} ${String(width).padStart(4)}x${String(height).padStart(4)}  ${subject}\n`);
}
process.stdout.write(`\n[plate-crops] written to ${outDirectory} from ${platePath}\n`);
