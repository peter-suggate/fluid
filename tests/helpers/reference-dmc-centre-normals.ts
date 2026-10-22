import assert from "node:assert/strict";

/** Restore the prior centre-normal route without changing production options. */
export function referenceDmcCentreNormals(device: GPUDevice): GPUDevice {
  return new Proxy(device, {
    get(target, property) {
      if (property === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        let code = descriptor.code;
        if (code.includes("fn rebuildDirtyBrickPayload(")) {
          const primitiveGuard = "bestPrimitive != INVALID_INDEX && primitiveFraction<=0.0";
          assert.equal(code.split(primitiveGuard).length - 1, 1,
            "reference DMC shader must restore exactly one occupied primitive centre-normal guard");
          code = code.replace(primitiveGuard, "bestPrimitive != INVALID_INDEX");
          if (code.includes("var<private> swSkipCentreNormal:bool;")) {
            const pageGuard = "swSkipCentreNormal=far==0;";
            assert.equal(code.split(pageGuard).length - 1, 1,
              "reference DMC page shader must restore exactly one far-terrain normal guard");
            code = code.replace(pageGuard, "swSkipCentreNormal=false;");
          }
        }
        return target.createShaderModule({ ...descriptor, code });
      };
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}
