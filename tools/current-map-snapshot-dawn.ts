import { createWriteStream } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

/** Read-only packed checkpoint: omit unused archive capacity, retain every
 * accepted increment plus both working maps and both fine-measure banks.
 * Metadata explicitly relocates the count/footer and measure in the artifact;
 * sourceMap/sourceMeasure preserve the original GPU addresses for provenance.
 */
export async function captureCurrentMapSnapshot(device: GPUDevice,
  source: WebGPUAdaptiveMassSolver["fieldSnapshotSourceForQA"],
  directory: string, stem = "current-map"): Promise<void> {
  const sourceMap = source.currentMap, sourceMeasure = source.currentMapMeasure;
  if (!sourceMap || !sourceMeasure) return;
  const header = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let metadata: Float32Array, bits: Uint32Array;
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source.state, 4 * source.retainedControlBaseWords!, header, 0, 16);
    encoder.copyBufferToBuffer(source.state, 4 * sourceMap.chainCountBaseWords, header, 16, 4);
    encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.scalarParityWord), header, 20, 4);
    encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.faceParityWord), header, 24, 4);
    device.queue.submit([encoder.finish()]); await header.mapAsync(GPUMapMode.READ);
    metadata = new Float32Array(header.getMappedRange()).slice(); bits = new Uint32Array(metadata.buffer);
  } finally { if (header.mapState === "mapped") header.unmap(); header.destroy(); }
  const count = metadata[4]!;
  if (!Number.isInteger(count) || count < 0 || count > sourceMap.chainCapacity) throw new Error("Invalid current-map snapshot archive count");
  const prefixWords = sourceMap.chainBaseWords - sourceMap.baseWords;
  const archiveWords = 3 * sourceMap.nodeCount * count;
  const totalWords = prefixWords + archiveWords + 1;
  const map = { ...sourceMap, chainCountBaseWords: sourceMap.baseWords + totalWords - 1,
    totalWords, endWords: sourceMap.baseWords + totalWords };
  const measure = { ...sourceMeasure, baseWords: map.endWords };
  const measureWords = 8 * sourceMeasure.dimensions.reduce((a, b) => a * b, 1);
  const readback = device.createBuffer({ size: 4 * (totalWords + measureWords), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder({ label: "Packed current-field QA checkpoint" });
    encoder.copyBufferToBuffer(source.state, 4 * sourceMap.baseWords, readback, 0, 4 * (prefixWords + archiveWords));
    encoder.copyBufferToBuffer(source.state, 4 * sourceMap.chainCountBaseWords, readback, 4 * (totalWords - 1), 4);
    encoder.copyBufferToBuffer(source.state, 4 * sourceMeasure.baseWords, readback, 4 * totalWords, 4 * measureWords);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
    await pipeline(Readable.from([bytes]), createGzip(), createWriteStream(join(directory, `${stem}.bin.gz`)));
    await writeFile(join(directory, `${stem}.json`), JSON.stringify({ map, measure, sourceMap, sourceMeasure,
      packedArchiveSlots: count, baseWords: map.baseWords, retainedControl: [...metadata.subarray(0, 4)],
      scalarParity: bits[5], faceParity: bits[6] }, null, 2));
  } finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); }
}
