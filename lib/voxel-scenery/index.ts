export * from "./builder";

/**
 * The primitive vocabulary scenery is expanded into.
 *
 * This directory used to hold one hand-written module per environment, each
 * calling `b.cylinder(...)` forty times. Geometry that only exists after a
 * function runs has nothing to select and nothing to edit, so the description
 * moved into the scene document: see lib/scenery-graph.ts for the schema,
 * lib/scenery-presets.ts for what each environment seeds, and
 * lib/scenery-expand.ts for the expansion that replaced the modules.
 *
 * What stayed behind is what a description cannot be: the builder that assigns
 * owner indices and derives bounds, and the procedural tree, whose seed
 * genuinely generates its ~30 primitives rather than naming them.
 */
