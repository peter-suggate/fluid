#!/usr/bin/env node
/** Compare packed rgba16float dry-scene captures from exact and LOD runs. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { compareSvoScreenSpaceImages } from "../lib/svo/svo-screen-space-termination";

const [referencePath, candidatePath, widthRaw, heightRaw, outputPath] = process.argv.slice(2);
const width = Number(widthRaw), height = Number(heightRaw);
assert.ok(referencePath && candidatePath, "usage: compare-svo-screen-space-images reference.raw candidate.raw width height [output.json]");
assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0, "width and height must be positive integers");

function decodeF16(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function decode(path: string): Float32Array {
  const packed = readFileSync(path);
  const expectedBytes = width * height * 8;
  assert.equal(packed.byteLength, expectedBytes, `${path} must contain ${expectedBytes} packed rgba16float bytes`);
  const half = new Uint16Array(packed.buffer, packed.byteOffset, packed.byteLength / 2);
  const result = new Float32Array(half.length);
  for (let index = 0; index < half.length; index += 1) result[index] = decodeF16(half[index]);
  return result;
}

const report = {
  referencePath,
  candidatePath,
  width,
  height,
  ...compareSvoScreenSpaceImages(decode(referencePath), decode(candidatePath), { width, height }),
};
const json = `${JSON.stringify(report, null, 2)}\n`;
if (outputPath) writeFileSync(outputPath, json);
process.stdout.write(json);
