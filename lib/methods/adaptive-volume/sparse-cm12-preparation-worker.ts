/// <reference lib="webworker" />
import { WebGPUSparseCM12Resident } from "./webgpu-sparse-cm12-resident";
import { SolidWorldDirectory } from "../../core/solid-world";

self.onmessage = async (event: MessageEvent<Parameters<typeof WebGPUSparseCM12Resident.recordPreparedGeneration>[0]>) => {
  try {
    const input = event.data;
    // Structured cloning preserves directory data, not its lookup prototype.
    Object.setPrototypeOf(input.solidWorld.directory, SolidWorldDirectory.prototype);
    const recipe = await WebGPUSparseCM12Resident.recordPreparedGeneration(input);
    const buffers = new Set<ArrayBuffer>();
    const seen = new Set<object>();
    const visit = (value: unknown) => {
      if (!value || typeof value !== "object" || seen.has(value)) return;
      seen.add(value);
      if (value instanceof ArrayBuffer) { buffers.add(value); return; }
      if (ArrayBuffer.isView(value)) { if (value.buffer instanceof ArrayBuffer) buffers.add(value.buffer); return; }
      if (value instanceof Map) { for (const [key, entry] of value) { visit(key); visit(entry); } return; }
      if (value instanceof Set) { for (const entry of value) visit(entry); return; }
      for (const entry of Object.values(value)) visit(entry);
    };
    visit(recipe);
    self.postMessage({ recipe }, { transfer: [...buffers] });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error) });
  }
};
