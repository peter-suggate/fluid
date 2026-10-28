import { attachSurfaceMapping } from "./surface-mapping";
import { decodeMeshVolume, type MeshVolume } from "./mesh-volume";

/** Hosts may serve .gz as a compressed file or as HTTP Content-Encoding.
 * Fetch removes HTTP encoding itself, so inspect the body, not the suffix or
 * response header, before applying file-level decompression. */
export async function fetchMeshVolume(url: string, request: typeof fetch = fetch): Promise<MeshVolume> {
  try {
    const response = await request(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    let bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    }
    const volume = decodeMeshVolume(bytes);
    if (volume.header.surfaceMapping) {
      const companion = await request(volume.header.surfaceMapping.url);
      if (!companion.ok) throw new Error(`Surface mapping HTTP ${companion.status}`);
      await attachSurfaceMapping(volume, bytes, new Uint8Array(await companion.arrayBuffer()));
    }
    return volume;
  } catch (cause) {
    throw new Error(`Loading mesh asset ${url}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
  }
}
