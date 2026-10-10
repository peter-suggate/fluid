/** Replace scene geometry baked into mixed shaders with the ownership uniform.
 * The first sixteen bytes remain the GPU-updated work counts. Geometry follows
 * them and survives adoption of a new ownership generation. */
export function uniformRuntimeConstantsWGSL(code: string, roots: Readonly<Record<string, { type: string; value: string }>>): string {
  const definitions = new Map<string, { type: string; value: string; source: string }>();
  for (const match of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*([^=;]+))?\s*=\s*([^;]+);/g)) {
    definitions.set(match[1]!, { type: match[2]?.trim() ?? "", value: match[3]!.trim(), source: match[0] });
  }
  const replacements = new Map(Object.entries(roots).filter(([key]) => definitions.has(key)));
  for (;;) {
    const size = replacements.size;
    for (const [key, definition] of definitions) {
      if (!replacements.has(key) && [...replacements.keys()].some(dependency => new RegExp(`\\b${dependency}\\b`).test(definition.value))) {
        replacements.set(key, { type: definition.type || "u32", value: definition.value });
      }
    }
    if (size === replacements.size) break;
  }
  if (!replacements.size) return code;
  const replace = (text: string) => text.replace(new RegExp(`\\b(${[...replacements.keys()].join("|")})\\b`, "g"), "$1()");
  for (const key of replacements.keys()) code = code.replace(definitions.get(key)!.source, "");
  return replace(code) + [...replacements].map(([key, definition]) => `\nfn ${key}()->${definition.type}{return ${replace(definition.value)};}\n`).join("");
}

export function uniformSceneConstantsWGSL(code: string): string {
  const counts = [...code.matchAll(/var<uniform>\s+(\w*umCounts)\s*:\s*vec4u\s*;/g)];
  for (const [, name] of counts) {
    const prefix = name!.slice(0, -"umCounts".length);
    const definitions = new Map<string, { type: string; value: string; source: string }>();
    for (const match of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*([^=;]+))?\s*=\s*([^;]+);/g)) {
      definitions.set(match[1]!, { type: match[2]?.trim() ?? "", value: match[3]!.trim(), source: match[0] });
    }
    const replacements = new Map<string, { type: string; value: string }>();
    if (definitions.has(`${prefix}UM_D`)) replacements.set(`${prefix}UM_D`, { type: "vec3u", value: `${name}.dimensions.xyz` });
    if (definitions.has(`${prefix}UM_TILES`)) replacements.set(`${prefix}UM_TILES`, { type: "u32", value: `${name}.dimensions.w` });
    if (definitions.has(`${prefix}UM_H`)) {
      const scalar = definitions.get(`${prefix}UM_H`)!.type === "f32";
      replacements.set(`${prefix}UM_H`, { type: scalar ? "f32" : "vec3f", value: `${name}.spacing.${scalar ? "w" : "xyz"}` });
    }
    if (definitions.has(`${prefix}UM_HMIN`)) replacements.set(`${prefix}UM_HMIN`, { type: "f32", value: `${name}.spacing.w` });
    if (!prefix && definitions.has("H")) replacements.set("H", { type: "vec3f", value: `${name}.spacing.xyz` });
    if (!prefix && definitions.has("MAX_H")) replacements.set("MAX_H", { type: "f32", value: `max(${name}.spacing.x,max(${name}.spacing.y,${name}.spacing.z))` });
    if (!replacements.size) continue;
    // Derived offsets and dimensions follow the same geometry. Infer vector
    // type only for the tile dimensions; all other implicit declarations here
    // are unsigned offsets. Explicit declarations retain their type.
    for (;;) {
      const size = replacements.size;
      for (const [key, definition] of definitions) {
        if (replacements.has(key)) continue;
        if ([...replacements.keys()].some(dependency => new RegExp(`\\b${dependency}\\b`).test(definition.value))) {
          replacements.set(key, { type: definition.type || (key === `${prefix}UM_T` ? "vec3u" : "u32"), value: definition.value });
        }
      }
      if (size === replacements.size) break;
    }
    // Fixed workgroup/private array extents still require compile-time values.
    // Leave those modules intact until their tables have a padded layout.
    if ([...replacements.keys()].some(key => new RegExp(`(?:array<[^;{}]*,|@workgroup_size\\()[^;{}]*\\b${key}\\b`).test(code))) continue;
    for (const key of replacements.keys()) code = code.replace(definitions.get(key)!.source, "");
    let helpers = "";
    for (const [key, replacement] of replacements) helpers += `\nfn ${key}()->${replacement.type}{return ${replacement.value};}\n`;
    const replace = (text: string) => text.replace(new RegExp(`\\b(${[...replacements.keys()].join("|")})\\b`, "g"), "$1()");
    code = replace(code);
    // Replace dependencies in bodies without changing the functions' names.
    helpers = helpers.replace(/\{return ([^;]+);\}/g, (_, value: string) => `{return ${replace(value)};}`);
    const declaration = `struct ${prefix}UMSceneCounts{work:vec4u,dimensions:vec4u,spacing:vec4f}\n`;
    code = code.replace(new RegExp(`\\b${name}\\b(?!\\s*:)`, "g"), `${name}.work`);
    code = code.replace(new RegExp(`(var<uniform>\\s+${name}\\s*:)\\s*vec4u`), `$1 ${prefix}UMSceneCounts`);
    // WGSL directives must precede every declaration.
    const directives = /^(?:\s*(?:enable|requires|diagnostic)\b[^;]*;)*\s*/.exec(code)![0];
    code = directives + declaration + code.slice(directives.length) + helpers + `\nfn ${prefix}umSceneSpacing()->vec3f{return ${name}.spacing.xyz;}\nfn ${prefix}umSceneMinimumSpacing()->f32{return ${name}.spacing.w;}\nfn ${prefix}umSceneMaximumSpacing()->f32{return max(${name}.spacing.x,max(${name}.spacing.y,${name}.spacing.z));}\n`;
  }
  return code;
}
