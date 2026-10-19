/** Keep physical spacing, work counts and paged pool capacity in the
 * ownership uniform. Boxed grid dimensions remain specialized: constant division and addressing are
 * substantially cheaper than runtime integer division in the mixed kernels.
 * The first sixteen bytes are GPU-updated counts; geometry survives adoption
 * of a new ownership generation. */
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

export function uniformSceneConstantsWGSL(code: string, specializeWorldGeometry = false): string {
  const counts = [...code.matchAll(/var<uniform>\s+(\w*umCounts)\s*:\s*vec4u\s*;/g)];
  for (const [, name] of counts) {
    const prefix = name!.slice(0, -"umCounts".length);
    const definitions = new Map<string, { type: string; value: string; source: string }>();
    for (const match of code.matchAll(/\bconst\s+(\w+)\s*(?::\s*([^=;]+))?\s*=\s*([^;]+);/g)) {
      definitions.set(match[1]!, { type: match[2]?.trim() ?? "", value: match[3]!.trim(), source: match[0] });
    }
    const replacements = new Map<string, { type: string; value: string }>();
    if (definitions.has(`${prefix}UM_D`)) replacements.set(`${prefix}UM_D`, { type: "vec3u", value: definitions.get(`${prefix}UM_D`)!.value });
    // Paged kernels divide by fixed page edges, not by the pool capacity.
    // The existing ownership header carries capacity in dimensions.w.
    if (definitions.has(`${prefix}UM_TILES`)) replacements.set(`${prefix}UM_TILES`, { type: "u32", value: definitions.has(`${prefix}UM_D`) ? definitions.get(`${prefix}UM_TILES`)!.value : `${name}.dimensions.w` });
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
  return uniformWindowCountsWGSL(code, specializeWorldGeometry);
}

/** Window metadata shares each owning topology's existing uniform.
 * Generation targets which delegate their addresses have no matching reads,
 * and retain their original 48-byte binding. Other buffers retain storage reads. */
function uniformWindowCountsWGSL(code:string,specializeWorldGeometry:boolean):string {
 const counts=[...code.matchAll(/var<uniform>\s+(\w*umCounts)\s*:\s*(\w+)\s*;/g)];
 for(const [,name,type] of counts){
  const prefix=name!.slice(0,-"umCounts".length);
  const pattern=new RegExp(`\\b${prefix}umTopology\\[${prefix}UM_PAGE_TABLE(?:\\(\\))?\\+(2621(?:4[4-9]|[56][0-9]|7[0-5]))u\\]`,"g");
  if(!pattern.test(code))continue;pattern.lastIndex=0;
  if(type==="vec4u"){
   const struct=`${prefix}UMWindowCounts`;
   code=code.replace(new RegExp(`\\b${name}\\b(?!\\s*:)`,"g"),`${name}.work`);
   code=code.replace(new RegExp(`(var<uniform>\\s+${name}\\s*:)\\s*vec4u`),`$1 ${struct}`);
   const directives=/^(?:\s*(?:enable|requires|diagnostic)\b[^;]*;)*\s*/.exec(code)![0];
   code=directives+`struct ${struct}{work:vec4u,dimensions:vec4u,spacing:vec4f,window:array<vec4u,8>}\n`+code.slice(directives.length);
  }else{
   const declaration=new RegExp(`struct\\s+${type}\\s*\\{([^}]*)\\}`);
   const match=declaration.exec(code);if(!match)throw new Error(`Window counts missing struct ${type}`);
   code=code.replace(declaration,`struct ${type}{${match[1]},window:array<vec4u,8>}`);
  }
  code=code.replace(pattern,(_read,index:string)=>{const word=Number(index)-262144;return `${name}.window[${word>>2}u][${word&3}u]`;});
  if(specializeWorldGeometry)continue;
  // Keep the fixed-world promise (SolidIsBox), but read each scene's bounds
  // and frame from its uniform. Coordinates must not split pipeline keys.
  const windowPrefix=`${prefix}um`;
  code=code.replace(new RegExp(`select\\((${name}\\.window\\[\\d+u\\]\\[\\d+u\\]),${windowPrefix}WindowWord\\d+,${windowPrefix}(?:FixedWorldGeometry|FixedFrame)\\)`,"g"),"$1");
  code=code.replace(new RegExp(`override ${windowPrefix}WindowWord\\d+:u32=0u;`,"g"),"");
  code=code.replace(`override ${windowPrefix}FixedFrame:bool=false;`,"");
 }
 return code;
}
