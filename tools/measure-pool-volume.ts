/** Post-run QA only. The production volume operator reads phi and fill textures,
 * resets its diagnostic shift to zero, and writes scratch plus budget telemetry.
 * It neither advances time nor writes phi/velocity/volume fields. Borrowed
 * scratch is no longer needed once the timed simulation has finished.
 */
import assert from "node:assert/strict";
import type {UniformMixedSurfaceVolume} from "../lib/methods/uniform/uniform-mixed-surface-volume";
export interface PoolVolumeFrame {
 surfaceVolume?:Pick<UniformMixedSurfaceVolume,"encode">;
 surfaceVolumeGroup?:Parameters<UniformMixedSurfaceVolume["encode"]>[1];
}
export function measureCurrentPoolPhiVolume(device:GPUDevice,frame:PoolVolumeFrame):void{
 assert.ok(frame.surfaceVolume&&frame.surfaceVolumeGroup,"Current phi volume QA needs the initialized mixed volume operator");
 const encoder=device.createCommandEncoder({label:"Post-run current phi volume QA (no time advance)"});
 frame.surfaceVolume.encode(encoder,frame.surfaceVolumeGroup,1,false);
 device.queue.submit([encoder.finish()]);
}
