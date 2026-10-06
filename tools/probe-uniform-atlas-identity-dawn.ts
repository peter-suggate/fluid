/** Control for the atlas addressing experiment: the table shader is unchanged,
 * but every page translation is zero. Separates shader adaptation from physical
 * placement. Run with --atlas=table and a distinct output filename. The main
 * probe's two-file experiment fingerprint excludes THIS launcher; its hash is
 * printed here and must accompany any identity-control claim.
 */
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFileSync} from "node:fs";
import {UniformAtlasAddressExperiment} from "./uniform-atlas-address-experiment";
assert.ok(process.argv.includes("--atlas=table"));
console.log(JSON.stringify({control:"zero page translations",launcherFingerprint:createHash("sha256").update(readFileSync(new URL(import.meta.url))).digest("hex")}));
const install=UniformAtlasAddressExperiment.prototype.install;
UniformAtlasAddressExperiment.prototype.install=function(device:GPUDevice){
 assert.equal(this.mode,"table");this.offsets.fill(0);return install.call(this,device);
};
await import("./probe-uniform-stage-scaling-dawn");
