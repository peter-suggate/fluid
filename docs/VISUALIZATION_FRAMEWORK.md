# Visualization framework

A pass owns its picture. The frame assembles them.

Status: **all three phases landed.** Decorations, the registry-driven UI and the
binding resolver are in, with 39 framework tests and the full unit suite at or
below its pre-existing failure count. Not confirmed visually in the running app.

## What it replaced

Five unrelated mechanisms drew diagnostics and shared nothing. The first four
rows below are what the framework now spans; the fifth is deliberately left
alone.

| Mechanism | Module | Size | Producers |
| --- | --- | --- | --- |
| Technique field overlay | `webgpu-octree-technique-overlay.ts` | 535 | 20 modes |
| Technique audit overlay | `webgpu-octree-technique-audit-overlay.ts` | 283 | same mode code |
| Generic field overlay | `webgpu-grid-overlay.ts` | 683 | ~8 modes |
| Instanced line work | `webgpu-svo-pixel-trace-overlay.ts` | 368 | 1 (the SVO trace) |
| Selection probes | `webgpu-svo-pixel-trace.ts`, `webgpu-fluid-cell-trace.ts` | 747 + 435 | 2 |

### The cost of adding one visualization

Adding `flood-provenance` — one view, no new data — took six edits across four
files: a string in `OCTREE_TECHNIQUE_OVERLAY_MODES`, an integer in
`OCTREE_TECHNIQUE_OVERLAY_CODES`, a WGSL branch inside the fine-lifecycle
fragment shader, a branch in `encode()`'s pipeline selector, a `PaperView` card
in `PerformancePanel.tsx`, and a count in `performance-observatory-ui.test.ts`.

The mode integer is a hand-managed global namespace: codes run 12–32 with a hole
at 24. `PAPER_VIEWS` is a hand-maintained parallel list of the same set. Neither
the shader nor the panel can be told a mode exists; they each have to be edited
to agree.

`blast-radius` was worse: it needed its own pipeline, so mode 32 is a special
case delegated out of `encode()` before the pipeline selector runs.

## The proposal

One registry. A visualization is declared next to the pass that produces its
data, and the frame assembles whatever is enabled.

```ts
export interface Visualization<Subject> {
  readonly id: string;              // "pressure-solve/stencil"
  readonly pass: string;            // groups the toggles, names the source
  readonly label: string;
  readonly description: string;
  readonly legend: readonly { swatch: `#${string}`; label: string }[];
  accepts(subject: unknown): subject is Subject;
  readonly kind: VisualizationKind;
}
```

### Two kinds, not one

This is the load-bearing finding, and forcing it to one kind is the way this
framework goes wrong.

**`decoration`** — host-built world-space line work.

```ts
build(subject: Subject, into: DecorationBuilder): readonly VisualizationNote[];
```

A decoration is segments: two points, a colour, a width in device pixels, an
optional dash, an optional arrowhead. Boxes, stencils, cones, rings and crosses
are all expressible, so *every* decoration in a frame lands in one buffer and
draws in **one instanced call**, whatever produced it. N of them compose freely.

**`field`** — a per-pixel WGSL fragment.

```ts
readonly wgsl: string;              // contributes `sample(point) -> vec4f`
readonly bindings: readonly VisualizationBinding[];
```

A field samples GPU buffers per pixel inside a volume or slice raymarch. It
cannot be host-built — that is the whole reason those 20 modes live in a shader.
Fields are volume-composited, so **one at a time**, which is already true of the
mode code today.

The framework's job is to make both declarable in the same place and assembled
centrally. It is not to pretend they are the same thing.

### Assembly

```ts
// One buffer, one draw, any number of contributors.
assembleDecorations({ definitions, subject, space, enabled }): {
  geometry: DecorationGeometry;
  entries: readonly { id; pass; label; segmentCount; notes }[];
  key: string;                       // upload only when the picture changed
}

// One pipeline, one mode.
resolveField({ definitions, id }): { pipeline; bindGroup } | undefined
```

`DecorationSpace` is the single place a pass's lattice becomes world metres, so
a decorator that reasons in fine cells and one that reasons in level-3 cells
both land correctly without either knowing the container convention.

### Registration is static, not a singleton

Each pass exports its own definitions; `lib/visualization-catalog.ts` imports
and lists them. A mutable global registry would be simpler to write and worse to
live with: order would depend on import timing, hot reload would double-register,
and a test could not construct a catalog without the whole app. A static list
keeps colocation — the definition lives with the shader it explains — while
staying deterministic, tree-shakeable and testable.

### What stops being hand-maintained

The `PaperView` cards, the `VisualPanel` toggles and the trace HUD's layer
buttons all become reads of the registry. Legends live with the shader that
chooses the colours, which is the only place they can be kept honest.

## What landed

### Phase 1 — decorations

- `lib/visualization-decorations.ts` — the vocabulary. `DecorationSpace` is the
  single place a pass's lattice becomes world metres; `DecorationBuilder`
  accumulates segments, boxes, arrows, rings and crosses into one buffer.
- `lib/visualization-registry.ts` — the declaration types and
  `assembleDecorations`, which merges every enabled contributor that accepts any
  of the frame's selections into one geometry, one key and one set of notes.
- `lib/visualization-catalog.ts` — the static list, plus the group lookup the UI
  speaks through.
- `lib/webgpu-decoration-overlay.ts` — the instanced pipeline, renamed from
  `webgpu-svo-pixel-trace-overlay.ts` because it is no longer the ray probe's.

Contributors, each beside its own pass:

| Declaration | Lives in | Draws |
| --- | --- | --- |
| `pressure-solve/cell` | `fluid-cell-visualizations.ts` | owning leaf + the finest cell under the pointer |
| `pressure-solve/stencil` | `fluid-cell-visualizations.ts` | same-size power neighbours, arrow + box |
| `pressure-solve/transition` | `fluid-cell-visualizations.ts` | neighbours at a different leaf size |
| `pressure-solve/cone` | `fluid-blast-radius.ts` | nested dependency shells, dashed |
| `fine-band/flood-reach` | `fine-flood-provenance.ts` | the flood's reach envelope, dashed |
| `svo-traversal/*` (9) | `svo-pixel-trace.ts` | the ray probe, wrapped not rewritten |

The picker is the payoff: hovering draws the whole diagnostic, cone included,
and pinning changes nothing about the picture — it freezes the *aim* so the
camera can move around it. `emphasis` is prominence, never content: a hovered
cell that showed less than a pinned one made committing the price of learning
anything. `assembleDecorations` keys on the drawn facts rather than a revision
counter, so orbiting a frozen cell or sweeping within one leaf uploads nothing —
which is what makes hover parity affordable, given the gather re-runs every
frame.

### Phase 2 — registry-driven UI

`PAPER_VIEWS` is now `VISUALIZATION_FIELDS.filter((field) => !field.hidden)`, and
the cell-trace HUD's legend is `visualizationsForGroups(...)` — one list that is
both the key and the switchboard, its glyphs read from each view's declared
`mark` so an arrow means a coupling and a dash means derived. Every view keeps a
row whether it is on or has anything to draw, dimmed rather than dropped: the row
is the switch, so hiding it would take the way back with it. The 20
technique modes are declared in `octree-technique-debug.ts` with their labels,
descriptions, legends and sources — beside the publication they read. Seven modes
the overlay can render but no picker offered are declared `hidden`, so the
harness has the complete set without new cards appearing as a side effect.

### Phase 3 — bindings and dispatch

`lib/visualization-bindings.ts` turns a program's bindings into both its WGSL
preamble and its host bind-group entries, from one declaration. Slot numbers now
exist in exactly one place, so the shader cannot read a different buffer than the
frame bound. `visualizationProgramProblems` makes the storage ceiling arithmetic:
an eleventh storage buffer fails the build instead of the driver.

`encode()`'s `modeCode === 18 || modeCode === 25 || …` chain is gone; a mode
declares its program and the overlay looks it up.

**What phase 3 deliberately did not do.** The five fragment bodies were not split
into 20 per-mode WGSL fragments. Modes within a program share nearly all their
code — `rowSample`, `faceSample`, `fineState` each serve four modes — so
splitting them would be churn against working, unverifiable-by-test shader
output for no structural gain. A *program* is a binding set plus a shader; a
*field* is a mode of a program. The coupling the plan named — hand-numbered
slots, a hand-written dispatch chain, three parallel UI lists — is paid off.

### Measured budgets

| Program | Storage buffers | Uniforms | Textures | Headroom |
| --- | --- | --- | --- | --- |
| topology | 5 | 1 | 1 | 5 |
| face | 4 | 1 | 1 | 6 |
| structured | 6 | 2 | 1 | 4 |
| lifecycle | 1 | 2 | 0 | 9 |
| fine | 7 | 3 | 0 | 3 |

The fine-band program is the tightest. Three slots are what a new fine-band view
has to fit inside before it needs a program of its own.

## Feasibility — what has been confirmed

- **The overlay pipeline is already pass-agnostic.** It consumes only
  `{segments, segmentCount}`. Widening `setGeometry` to a `DecorationGeometry`
  interface typechecks against the existing SVO producer with no other change.
- **The 16-float instance layout is sufficient.** It already carries everything
  the SVO trace draws — boxes, rings, screen-space arrowheads, dashes,
  near-plane clipping, depth ghosting. A shared `DecorationBuilder` covers both
  existing producers with no format change.
- **The SVO pixel trace migrates by wrapping, not rewriting.**
  `buildSvoPixelTraceGeometry` already emits the exact layout, so it joins via
  `DecorationBuilder.append()` and draws in the same call as everything else.
  Its 1,082-line geometry builder is untouched.
- **It builds and runs the tests.** `npx tsc --noEmit` clean except a pre-existing
  error in `webgpu-octree-coarse-summary.ts`; 57 shaders validate, including the
  five overlay programs now assembled from declarations; 39 framework tests pass;
  the full unit suite is at 28 failures against a 29-failure baseline, and every
  remaining failure is a strict subset of the pre-existing list.

### The obstacle, stated plainly

Fields are **not** cheap to migrate, and the reason is a hardware limit rather
than a code-shape problem. The 20 modes are packed into five bind-group layouts
because Apple silicon caps a fragment stage at **10 storage buffers**. Letting
each field declare its own bindings needs a binding-budget resolver that packs
declarations into layouts and fails loudly when a field asks for more than a
stage can hold. That resolver is the real work in phase 3, and until it exists,
"each pass declares its bindings" is a promise the hardware will not keep.

## Migration phases — as executed

**Phase 1 — decorations (done).**
`DecorationBuilder`, the registry, `assembleDecorations`, one shared overlay in
the renderer. Contributors: the fluid cell picker (leaf, cell, 18-neighbour
stencil, resolution transitions), the blast-radius cone (from
`fluid-blast-radius.ts`), the flood reach shell (from
`fine-flood-provenance.ts`), and the SVO pixel trace by wrapping. Deletes
`lib/fluid-cell-trace-geometry.ts`, which was the one-off this replaces.
Nothing existing changes behaviour.

**Phase 2 — registry-driven UI (done).** `PAPER_VIEWS`, the panel toggles
and the HUD layer buttons read the catalog. Removes three parallel lists. The
mode integers stay exactly as they are.

**Phase 3 — bindings and dispatch (done, scoped).** See "What landed" for what
was and was not migrated, and why the shared fragment bodies stayed whole.

**Not proposed.** Folding `webgpu-grid-overlay.ts` in. It renders dense texture
fields with a different sampling model, its own modes are not paper techniques,
and it has no producer that wants to colocate with it.

## Convention the framework carries

Solid line work is **measured** — state a frame published and read back. Dashed
line work is **derived** — a bound, a schedule, or anything a residual gate may
have cut short. Every decorator's notes declare `gathered` or `scheduled` for
the same reason, so a picture cannot claim more than its source knows. See
`FLUID_WORK_INSPECTION.md` for why a cell trace has to make that distinction at
all.
