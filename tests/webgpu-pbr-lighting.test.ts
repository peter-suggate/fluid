import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluateUnifiedLighting,
  ggxNormalDistribution,
  schlickFresnel,
  smithGgxVisibility,
  unifiedLightingShaderLibrary,
  type UnifiedLightingMaterial,
} from "../lib/webgpu-lighting";

function material(overrides: Partial<UnifiedLightingMaterial> = {}): UnifiedLightingMaterial {
  return {
    baseColorLinear: [0.72, 0.72, 0.72],
    emissiveLinear: [0, 0, 0],
    roughness: 0.42,
    metallic: 0,
    specularF0Linear: [0.04, 0.04, 0.04],
    specularWeight: 1,
    ambientDiffuse: 0,
    rimColorLinear: [0, 0, 0],
    rimWeight: 0,
    ...overrides,
  };
}

test("Schlick, GGX, and Smith CPU mirrors preserve finite physical endpoints", () => {
  assert.deepEqual(schlickFresnel(1, [0.04, 0.2, 0.8]), [0.04, 0.2, 0.8]);
  assert.deepEqual(schlickFresnel(0, [0.04, 0.2, 0.8]), [1, 1, 1]);
  assert.deepEqual(schlickFresnel(Number.NaN, [0.04, 0.2, 0.8]), [1, 1, 1]);

  const sharp = ggxNormalDistribution(1, 0.1);
  const medium = ggxNormalDistribution(1, 0.5);
  const rough = ggxNormalDistribution(1, 1);
  assert.ok(Number.isFinite(sharp) && Number.isFinite(medium) && Number.isFinite(rough));
  assert.ok(sharp > medium && medium > rough, "roughness broadens and lowers the normal-incidence GGX peak");
  assert.equal(smithGgxVisibility(0, 1, 0.4), 0);
  assert.equal(smithGgxVisibility(1, 0, 0.4), 0);
  assert.ok(Number.isFinite(smithGgxVisibility(1e-5, 1e-5, 0.04)));
});

test("roughness changes highlight concentration while metallic uses colored F0 without diffuse", () => {
  const input = {
    shadingNormal: [0, 1, 0] as const,
    towardViewer: [0, 1, 0] as const,
    towardLight: [0, 1, 0] as const,
    lightColorLinear: [1, 1, 1] as const,
  };
  const sharp = evaluateUnifiedLighting(material({ baseColorLinear: [0, 0, 0], roughness: 0.12 }), input);
  const rough = evaluateUnifiedLighting(material({ baseColorLinear: [0, 0, 0], roughness: 0.8 }), input);
  assert.ok(sharp[0] > rough[0] * 100, "the same specular energy is concentrated into a sharper peak");

  const metal = evaluateUnifiedLighting(material({
    baseColorLinear: [0.9, 0.2, 0.05], metallic: 1, roughness: 0.45, specularF0Linear: [0.04, 0.04, 0.04],
  }), input);
  assert.ok(metal[0] > metal[1] && metal[1] > metal[2], "metal base color becomes conductor Fresnel color");
});

test("geometric normals reject back-side light and degenerate shading inputs stay finite", () => {
  const backLit = evaluateUnifiedLighting(material(), {
    shadingNormal: [0.999, 0.045, 0],
    geometricNormal: [0, 1, 0],
    towardViewer: [0, 1, 0],
    towardLight: [0, -1, 0],
    lightColorLinear: [100, 100, 100],
  });
  assert.deepEqual(backLit, [0, 0, 0], "a perturbed shading normal cannot leak light through the geometric back face");

  const degenerate = evaluateUnifiedLighting(material({
    baseColorLinear: [Number.NaN, Number.POSITIVE_INFINITY, -1],
    emissiveLinear: [0.1, 0.2, 0.3],
  }), {
    shadingNormal: [0, 0, 0],
    geometricNormal: [0, 0, 0],
    towardViewer: [0, 0, 0],
    towardLight: [0, 0, 0],
    lightColorLinear: [Number.NaN, -1, Number.POSITIVE_INFINITY],
  });
  assert.ok(degenerate.every(Number.isFinite));
  assert.deepEqual(degenerate, [0.1, 0.2, 0.3], "emissive remains independent of malformed incident lighting");
});

test("deterministic white-furnace integration does not create reflected energy", () => {
  const rings = 72;
  const sectors = 144;
  const solidAngle = 2 * Math.PI / (rings * sectors);
  const furnaceMaterials = [
    material({ baseColorLinear: [1, 1, 1], roughness: 0.12 }),
    material({ baseColorLinear: [1, 1, 1], roughness: 0.5 }),
    material({ baseColorLinear: [1, 1, 1], roughness: 1 }),
    material({ baseColorLinear: [0.92, 0.62, 0.18], metallic: 1, roughness: 0.35 }),
  ];
  for (const closure of furnaceMaterials) {
    const reflected = [0, 0, 0];
    for (let ring = 0; ring < rings; ring += 1) {
      const y = (ring + 0.5) / rings;
      const radius = Math.sqrt(Math.max(0, 1 - y * y));
      for (let sector = 0; sector < sectors; sector += 1) {
        const azimuth = 2 * Math.PI * (sector + 0.5) / sectors;
        const sample = evaluateUnifiedLighting(closure, {
          shadingNormal: [0, 1, 0],
          geometricNormal: [0, 1, 0],
          towardViewer: [0, 1, 0],
          towardLight: [radius * Math.cos(azimuth), y, radius * Math.sin(azimuth)],
          lightColorLinear: [1, 1, 1],
        });
        for (let channel = 0; channel < 3; channel += 1) reflected[channel] += sample[channel] * solidAngle;
      }
    }
    for (const channel of reflected) {
      assert.ok(Number.isFinite(channel) && channel >= 0 && channel <= 1.015, `white furnace reflectance ${channel} must remain bounded`);
    }
  }
});

test("WGSL retains legacy constructors while exposing the shared PBR and normal contract", () => {
  assert.match(unifiedLightingShaderLibrary, /fn unifiedMaterial\([\s\S]*\) -> UnifiedLightingMaterial/);
  assert.match(unifiedLightingShaderLibrary, /return unifiedPbrMaterial\(baseColor, 0\.0, roughness/,
    "legacy raster/SVO call sites remain dielectric by default");
  assert.match(unifiedLightingShaderLibrary, /fn unifiedPbrMaterial\(/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedGgxDistribution\(/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedSmithGgxVisibility\(/);
  assert.match(unifiedLightingShaderLibrary, /diffuseColor \* \(vec3f\(1\.0\) - fresnel\) \/ UNIFIED_PI/);
  assert.match(unifiedLightingShaderLibrary, /let f0 = mix\(dielectricF0, material\.baseColor, material\.metallic\)/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedLightingInputWithGeometry\(/);
  assert.match(unifiedLightingShaderLibrary, /geometricDotL/);
  assert.match(unifiedLightingShaderLibrary, /fn unifiedSafeNormal\(/);
  assert.doesNotMatch(unifiedLightingShaderLibrary, /@group|@binding/);
});
