/**
 * The scene contact's barrier stiffness (clothContactStiffness): kc = 2 F_max / r, F_max the membrane's tent bound
 * at the materials' strain-limit traction, so the design contact load sits in the barrier's quadratic stage (d >= r / 2).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { clothContactStiffness } from "../lib/cloth/scene-contact";
import { bridson2002RotatingSphere } from "../lib/cloth/scenes/bridson-2002";
import { herringBoneSheet } from "../lib/cloth/scenes/bridson-2002/mesh";
import { clothAnalyticFreeFall } from "../lib/cloth/scenes/analytic/scenes";
import type { ClothDocument, ClothMaterial } from "../lib/cloth/types";

const material: ClothMaterial = { id: "m", arealDensity_kg_m2: 1, stretchWarp_N_m: 200, stretchWeft_N_m: 200, shear_N_m: 100, poisson: 0,
  compressionRatio: 1, bending_N_m: 1e-4, damping_s: 0, thickness_m: 1e-3, strainLimit: 0.1 };
const sheet = (h: number, m: ClothMaterial = material): ClothDocument => ({ schemaVersion: "2.0.0", id: "sheet", materials: [m],
  geometry: { kind: "inline", data: herringBoneSheet({ columns: 6, rows: 6, width_m: 5 * h, depth_m: 5 * h, origin_m: [0, 0, 0] }) } });

test("contact stiffness: the tent bound at the strain limit's StVK traction, kc = 2 F / r, so F_max rests at r / 2", () => {
  const h = 0.05, { kc_N_m, load_N } = clothContactStiffness(sheet(h));
  // Traction at the 10 % limit: (1 + eps) k E, E = ((1 + eps)^2 - 1) / 2 = 0.105, so 23.1 N/m; a herring-bone sheet's
  // interior vertex has valence 6, its link four rest edges h and two diagonals h sqrt 2: half its perimeter (2 + sqrt 2) h.
  const T = 1.1 * 200 * 0.105, F = T * (2 + Math.SQRT2) * h;
  assert.ok(Math.abs(load_N - F) < 1e-9 * F, `F_max ${load_N} vs ${F}`);
  assert.ok(Math.abs(kc_N_m - 2 * F / 1e-3) < 1e-6 * kc_N_m, `kc ${kc_N_m}`);
  // The barrier's quadratic stage carries kc (r - d): at d = r / 2 it is F_max, so every load up to it rests at d >= r / 2.
  assert.ok(Math.abs(kc_N_m * 0.5e-3 - load_N) < 1e-9 * load_N);
  // The load scales with the mesh (the vertex's tributary width) and with the traction; Poisson coupling adds |c12|.
  assert.ok(Math.abs(clothContactStiffness(sheet(2 * h)).kc_N_m - 2 * kc_N_m) < 1e-6 * kc_N_m);
  const coupled = clothContactStiffness(sheet(h, { ...material, poisson: 0.3 })).load_N, D = 1 - 0.09;
  assert.ok(Math.abs(coupled - 1.1 * (200 / D + 0.3 * 200 / D) * 0.105 * (2 + Math.SQRT2) * h) < 1e-9 * coupled, `coupled ${coupled}`);
});

test("contact stiffness: the scenes' values (Bridson sphere 32^2 9.7e3 N/m, the analytic sheets 1.6e3 N/m), and a material with no strain limit fails explicitly", () => {
  const sphere = clothContactStiffness(bridson2002RotatingSphere({ nodesAcross: 32 }).cloth);
  assert.ok(Math.abs(sphere.kc_N_m - 9668) < 1, `sphere 32^2 kc ${sphere.kc_N_m}`);
  // C1's t25 fold loads (0.5-1.4 N) are inside the bound and rest at r - F / kc, 0.86-0.95 mm of the 1 mm.
  assert.ok(sphere.load_N > 1.4 && 1e-3 - 1.4 / sphere.kc_N_m > 0.5e-3);
  for (const d of clothAnalyticFreeFall().cloth) assert.ok(Math.abs(clothContactStiffness(d).kc_N_m - 1577) < 1, `free fall kc ${clothContactStiffness(d).kc_N_m}`);
  const { strainLimit: _, ...unlimited } = material;
  assert.throws(() => clothContactStiffness(sheet(0.05, unlimited)), /strain limit/);
});
