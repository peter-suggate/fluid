import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("transport clears receipts only for accepted cells", () => {
  assert.doesNotMatch(resident,
    /encoder\.clearBuffer\(this\.conditioning, 0,/,
    "reserved dynamic capacity must not be cleared every frame");
  assert.match(resident,
    /stage\("conservative-transport"[\s\S]*?dispatchAccepted\("clearSparseCM12TransportReceipts", "cell"\)[\s\S]*?dispatchTransportPacket\("traceGammaAndBeta"\)/);
  assert.match(shader,
    /fn clearSparseCM12TransportReceipts[\s\S]*?acceptedTemplateCellInvocation\(gid\.x\)[\s\S]*?plane<6u/);
});
