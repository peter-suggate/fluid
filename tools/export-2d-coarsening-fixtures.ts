/** Export authored inputs only. This tool never constructs or advances a solver. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";

const directory = fileURLToPath(new URL("../rust/core/testdata/coarsening-2d/", import.meta.url));
const ids = ["sparse-cm12-long-dam-break", "cm12-figure-2",
  "hydrostatic-power-large-offset", "coarse-first-pool-impact-half-slab"];
if (!process.argv.includes("--check")) mkdirSync(directory, { recursive: true });
for (const id of ids) {
  const definition = findSceneDefinition(id);
  if (!definition) throw new Error(`Missing authored scene ${id}`);
  const text = JSON.stringify(sceneDocument(definition), null, 2) + "\n";
  const path = `${directory}/${id}.json`;
  if (process.argv.includes("--check")) {
    if (readFileSync(path, "utf8") !== text) throw new Error(`Fixture drift: ${id}; review before regenerating`);
  } else writeFileSync(path, text);
}
