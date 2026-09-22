/**
 * TEMPORARY measurement switch for
 * docs/research/uniform-geometric-gpu-utilization-2026-09-21.
 *
 * `FLUID_UNIFORM_AB_OFF=a,b` restores the pre-change arm of each named
 * optimization so one working tree can run interleaved A/B lanes. Unset (the
 * app, every gate) means every optimization is on. Delete with the program.
 */
const off = new Set((typeof process !== "undefined" ? process.env?.FLUID_UNIFORM_AB_OFF ?? "" : "")
  .split(",").map((name) => name.trim()).filter(Boolean));
export type UniformAbFeature = "opentest" | "facetest" | "openlocal" | "philoops" | "batch" | "facecache" | "solidheader" | "measurelean" | "mgstaticid" | "staticid" | "inplace" | "fusevisit" | "liquidmask" | "rowsweep" | "deadgroups" | "donortiles" | "authoritystatic" | "lazystats" | "pressuretiles" | "surfacewindow" | "targetcache" | "pressureauthority" | "tileseed" | "maskfirst" | "balancetree" | "shelllist" | "frontreuse" | "solidfreetrace" | "philean" | "phicensuswindow" | "tilereach" | "cycletiles";
export const uniformAbOn = (feature: UniformAbFeature): boolean => !off.has(feature) && !off.has("all");
