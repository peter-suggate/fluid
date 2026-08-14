/** Immutable point-location tables compiled from the generated power catalog. */

import type { GeneratedOctreePowerCatalogViews } from "./generated/octree-power-catalog";
import {
  OCTREE_CUBE_TRANSFORMS,
  composeCubeTransforms,
  inverseCubeTransform,
  transformPowerVector,
} from "./octree-power-topology";

export const OCTREE_POWER_COMPILED_SAMPLER_MAGIC = 0x4353_414d;
export const OCTREE_POWER_COMPILED_SAMPLER_VERSION = 4;
export const OCTREE_POWER_COMPILED_SAMPLER_HEADER_WORDS = 12;
export const OCTREE_POWER_COMPILED_SAMPLER_OCTANTS = 8;
export const OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS = 48;
export const OCTREE_POWER_COMPILED_SAMPLER_INVALID = 0xff;
export const OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS = 256;
export const OCTREE_POWER_COMPILED_SAMPLER_FIXED_AXES = 3;

export interface OctreePowerCompiledSampler {
  readonly words: Uint32Array;
  readonly transformedSelectorOffsetWords: number;
  readonly adjacencyOffsetWords: number;
  readonly octantSeedOffsetWords: number;
  readonly barycentricOffsetWords: number;
  readonly canonicalFanTransformOffsetWords: number;
  readonly transformCompositionOffsetWords: number;
  readonly inverseTransformOffsetWords: number;
}

function vertex(catalog: Pick<GeneratedOctreePowerCatalogViews, "tetrahedronVertexData">,
  selector: number): readonly [number, number, number, number] {
  const at = 4 * selector;
  return [catalog.tetrahedronVertexData[at]!, catalog.tetrahedronVertexData[at + 1]!,
    catalog.tetrahedronVertexData[at + 2]!, catalog.tetrahedronVertexData[at + 3]!];
}

function selectorKey(value: readonly number[]): string {
  return `${value[0]},${value[1]},${value[2]},${value[3]}`;
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function cross3(a: readonly number[], b: readonly number[]): readonly [number, number, number] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!,
    a[0]! * b[1]! - a[1]! * b[0]!];
}

/**
 * Compile two searches out of every transition interpolation:
 *
 * - every cube transform maps a selector directly to another selector;
 * - every tetrahedron names its three neighbours through origin-incident faces.
 *
 * The latter is a walking point-location graph. Eight case-local seeds keep a
 * query close to the tetrahedron in its canonical octant without duplicating
 * the 367k-tetrahedron catalog for every symmetry.
 */
export function compileOctreePowerSampler(catalog: Pick<GeneratedOctreePowerCatalogViews,
  "tetrahedronHeaders" | "tetrahedronData" | "tetrahedronVertexData">): OctreePowerCompiledSampler {
  if (catalog.tetrahedronHeaders.length % 3 !== 0
    || catalog.tetrahedronVertexData.length % 4 !== 0) {
    throw new RangeError("Power sampler catalog arrays are truncated");
  }
  const entryCount = catalog.tetrahedronHeaders.length / 3;
  const selectorCount = catalog.tetrahedronVertexData.length / 4;
  if (selectorCount < 1 || selectorCount > 0xff) {
    throw new RangeError("Power sampler selector count exceeds its byte ABI");
  }
  const transformedSelectorOffsetWords = OCTREE_POWER_COMPILED_SAMPLER_HEADER_WORDS;
  const adjacencyOffsetWords = transformedSelectorOffsetWords
    + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS * selectorCount;
  const octantSeedOffsetWords = adjacencyOffsetWords + catalog.tetrahedronData.length;
  const barycentricOffsetWords = octantSeedOffsetWords
    + OCTREE_POWER_COMPILED_SAMPLER_OCTANTS * entryCount;
  const canonicalFanTransformOffsetWords = barycentricOffsetWords
    + 9 * catalog.tetrahedronData.length;
  const transformCompositionOffsetWords = canonicalFanTransformOffsetWords
    + OCTREE_POWER_COMPILED_SAMPLER_FIXED_AXES
      * OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS
      * OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS;
  const inverseTransformOffsetWords = transformCompositionOffsetWords
    + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS * OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS;
  const words = new Uint32Array(inverseTransformOffsetWords
    + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS);
  const floats = new Float32Array(words.buffer);
  words.fill(0xffff_ffff);
  words.set([OCTREE_POWER_COMPILED_SAMPLER_MAGIC, OCTREE_POWER_COMPILED_SAMPLER_VERSION,
    selectorCount, entryCount, transformedSelectorOffsetWords, adjacencyOffsetWords,
    octantSeedOffsetWords, catalog.tetrahedronData.length, barycentricOffsetWords,
    canonicalFanTransformOffsetWords, transformCompositionOffsetWords,
    inverseTransformOffsetWords], 0);

  const horizontalD4 = [
    0, 2, 4, 6, 8, 10, 12, 14,
    0, 1, 4, 5, 40, 41, 44, 45,
    0, 1, 2, 3, 16, 17, 18, 19,
  ] as const;
  for (let fixedAxis = 0; fixedAxis < OCTREE_POWER_COMPILED_SAMPLER_FIXED_AXES;
    fixedAxis += 1) {
    for (let mask = 0; mask < OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS; mask += 1) {
      for (const rowTransform of OCTREE_CUBE_TRANSFORMS) {
        let chosen = 0;
        let bestEffective = Number.POSITIVE_INFINITY;
        for (let symmetry = 0; symmetry < 8; symmetry += 1) {
          if ((mask & (1 << symmetry)) === 0) continue;
          const candidate = horizontalD4[8 * fixedAxis + symmetry]!;
          const effective = composeCubeTransforms(rowTransform,
            inverseCubeTransform(OCTREE_CUBE_TRANSFORMS[candidate]!)).code;
          if (effective < bestEffective || (effective === bestEffective && candidate < chosen)) {
            chosen = candidate;
            bestEffective = effective;
          }
        }
        const at = canonicalFanTransformOffsetWords
          + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS
            * (mask + OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS * fixedAxis)
          + rowTransform.code;
        words[at] = chosen;
      }
    }
  }
  for (const first of OCTREE_CUBE_TRANSFORMS) {
    words[inverseTransformOffsetWords + first.code] = inverseCubeTransform(first).code;
    for (const second of OCTREE_CUBE_TRANSFORMS) {
      words[transformCompositionOffsetWords
        + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS * first.code + second.code]
        = composeCubeTransforms(first, second).code;
    }
  }

  const selectorByGeometry = new Map<string, number>();
  for (let selector = 0; selector < selectorCount; selector += 1) {
    selectorByGeometry.set(selectorKey(vertex(catalog, selector)), selector);
  }
  for (let code = 0; code < OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS; code += 1) {
    const transform = OCTREE_CUBE_TRANSFORMS[code];
    if (!transform) throw new RangeError(`Missing cube transform ${code}`);
    for (let selector = 0; selector < selectorCount; selector += 1) {
      const source = vertex(catalog, selector);
      const transformed = transformPowerVector([source[0], source[1], source[2]], transform);
      const mapped = selectorByGeometry.get(selectorKey([...transformed, source[3]]));
      words[transformedSelectorOffsetWords + code * selectorCount + selector]
        = mapped ?? OCTREE_POWER_COMPILED_SAMPLER_INVALID;
    }
  }

  for (let entry = 0; entry < entryCount; entry += 1) {
    const first = catalog.tetrahedronHeaders[3 * entry]!;
    const count = catalog.tetrahedronHeaders[3 * entry + 1]!;
    if (first > catalog.tetrahedronData.length || count > catalog.tetrahedronData.length - first
      || count > 0xff) {
      throw new RangeError(`Power sampler tetrahedron range ${entry} is invalid`);
    }
    const selectors: [number, number, number][] = [];
    const faces = new Map<string, Array<readonly [number, number]>>();
    for (let local = 0; local < count; local += 1) {
      const packed = catalog.tetrahedronData[first + local]!;
      const tetra: [number, number, number] = [packed & 0xff, (packed >>> 8) & 0xff,
        (packed >>> 16) & 0xff];
      if (tetra.some((selector) => selector >= selectorCount)) {
        throw new RangeError(`Power sampler tetrahedron ${first + local} has an invalid selector`);
      }
      selectors.push(tetra);
      const x = vertex(catalog, tetra[0]);
      const y = vertex(catalog, tetra[1]);
      const z = vertex(catalog, tetra[2]);
      const yz = cross3(y, z), zx = cross3(z, x), xy = cross3(x, y);
      const determinant = x[0] * yz[0] + x[1] * yz[1] + x[2] * yz[2];
      if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) {
        throw new Error(`Power sampler tetrahedron ${first + local} is degenerate`);
      }
      const inverse = [...yz, ...zx, ...xy].map((coefficient) =>
        Math.fround(coefficient / determinant));
      floats.set(inverse, barycentricOffsetWords + 9 * (first + local));
      const keys = [pairKey(tetra[1], tetra[2]), pairKey(tetra[0], tetra[2]),
        pairKey(tetra[0], tetra[1])];
      keys.forEach((key, face) => {
        const incidence = faces.get(key) ?? [];
        incidence.push([local, face]); faces.set(key, incidence);
      });
    }
    for (const incidence of faces.values()) {
      if (incidence.length > 2) throw new Error(`Power sampler case ${entry} is non-manifold`);
      if (incidence.length !== 2) continue;
      const [[a, af], [b, bf]] = incidence;
      const aAt = adjacencyOffsetWords + first + a;
      const bAt = adjacencyOffsetWords + first + b;
      const aShift = 8 * af, bShift = 8 * bf;
      const aWord = words[aAt] === 0xffff_ffff ? 0x00ff_ffff : words[aAt]!;
      const bWord = words[bAt] === 0xffff_ffff ? 0x00ff_ffff : words[bAt]!;
      words[aAt] = (aWord & ~(0xff << aShift)) | (b << aShift);
      words[bAt] = (bWord & ~(0xff << bShift)) | (a << bShift);
    }
    for (let local = 0; local < count; local += 1) {
      const at = adjacencyOffsetWords + first + local;
      if (words[at] === 0xffff_ffff) words[at] = 0x00ff_ffff;
    }

    for (let octant = 0; octant < OCTREE_POWER_COMPILED_SAMPLER_OCTANTS; octant += 1) {
      let best = OCTREE_POWER_COMPILED_SAMPLER_INVALID;
      let bestScore = -Infinity;
      const wanted: [number, number, number] = [
        (octant & 1) === 0 ? -1 : 1,
        (octant & 2) === 0 ? -1 : 1,
        (octant & 4) === 0 ? -1 : 1,
      ];
      selectors.forEach((tetra, local) => {
        const sum: [number, number, number] = [0, 0, 0];
        for (const selector of tetra) {
          const value = vertex(catalog, selector);
          sum[0] += value[0]; sum[1] += value[1]; sum[2] += value[2];
        }
        const length = Math.hypot(...sum);
        const score = length > 0 ? (sum[0] * wanted[0] + sum[1] * wanted[1]
          + sum[2] * wanted[2]) / length : -Infinity;
        if (score > bestScore || (score === bestScore && local < best)) {
          best = local; bestScore = score;
        }
      });
      words[octantSeedOffsetWords + entry * OCTREE_POWER_COMPILED_SAMPLER_OCTANTS + octant]
        = best;
    }
  }

  // Prove the inverse transform table used by the shader cannot leave the
  // compiled selector domain. This also catches accidental transform-code ABI drift.
  for (const transform of OCTREE_CUBE_TRANSFORMS) {
    const inverse = inverseCubeTransform(transform);
    if (inverse.code >= OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS) {
      throw new Error("Power sampler inverse transform exceeds its compiled table");
    }
  }
  return { words, transformedSelectorOffsetWords, adjacencyOffsetWords, octantSeedOffsetWords,
    barycentricOffsetWords, canonicalFanTransformOffsetWords, transformCompositionOffsetWords,
    inverseTransformOffsetWords };
}
