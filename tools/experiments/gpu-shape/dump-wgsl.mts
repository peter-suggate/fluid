// Writes the sweep's kernel variants as .wgsl files for the register-cap check (fluid-scratch/brick-tune/regcap.sh).
//   [X=1] node --import tsx tools/experiments/gpu-shape/dump-wgsl.mts <dir> [R,...] [WGB,...]   (X=1: xor-add registers, X=p: pinned registers, X=q: recurrence registers)
import { mkdirSync, writeFileSync } from "node:fs";
import { chaseWGSL, kernelKey } from "./kernels.mjs";
const dir = process.argv[2]!; mkdirSync(dir, { recursive: true });
const Rs = (process.argv[3] ?? "0,16,32,48,64,96,128").split(",").map(Number);
const Ws = (process.argv[4] ?? "0,1024,8192,16384,24576").split(",").map(Number);
if (process.env.X) for (const M of [1, 4]) for (const R of Rs) { const k = { L: 256, M, R, X: process.env.X === "1", P: process.env.X === "p", Q: process.env.X === "q" }; writeFileSync(`${dir}/${kernelKey(k)}.wgsl`, chaseWGSL(k)); }
else for (const M of [1, 4]) for (const R of Rs) { const k = { L: 256, M, R }; writeFileSync(`${dir}/${kernelKey(k)}.wgsl`, chaseWGSL(k)); }
if (!process.env.X) for (const W of Ws) { const k = { L: 256, M: 1, WGB: W }; writeFileSync(`${dir}/${kernelKey(k)}.wgsl`, chaseWGSL(k)); }
