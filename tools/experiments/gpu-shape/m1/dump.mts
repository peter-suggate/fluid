// Write each M1 variant of a kcache kernel as its own WGSL file (for regcap.sh): node dump.mts <base.wgsl> <kernel> <outdir>
import { readFileSync, writeFileSync } from "node:fs";
import { variants } from "./variants.mjs";
const [file, kernel, out] = process.argv.slice(2) as [string, string, string];
const code = readFileSync(file, "utf8");
writeFileSync(`${out}/${kernel}-base.wgsl`, code);
for (const v of variants.filter(v => v.kernel === kernel)) { writeFileSync(`${out}/${kernel}-${v.name.replace(/\+/g, "_")}.wgsl`, v.apply(code)); console.log(`${kernel}/${v.name} written`); }
