# Shared UI plugin handoff — one capability, two hosts

Peter, 2026-09-17: *"you are clearly not sharing code via a plugin based, colocated
approach"*, *"regions is just one example of UI"*, *"visuals such as the fraction view
should be defined in one place and work for both the 2d and 3d"*, *"resizing regions in
3d is too smooth. it should snap to tiles/blocks by default"*, and: the 2-D region
strip's cell-size dropdown should exist in 3-D too.

Today's session put the 2-D advance lab on the studio's **mechanisms** — a lab
`PaneSession` (`advance-lab/AdvanceLab.tsx:633`), `RadialMenu` with a `perform` prop
(`:2360`), `ViewportModeToggle` (`:1631`), `EditorModeChip` (`:2034`),
`components/toolstrip.tsx` primitives, `components/armed-stroke.ts`, and the `host` arm
on `EditorActionEffect` (`lib/core/editor-action.ts:188`). It re-implemented every
**capability** lab-side. This document is the design that makes each capability a
single colocated plugin module rendered by both hosts.

**The verdict in five lines.**
1. The mechanism layer is already host-agnostic (`ComposedFeatureSlot`, `RadialRing`,
   `useArmedStroke`, the toolstrip primitives, `PaneSession`).
2. Every capability module is coupled to exactly one thing that is not: the module-level
   `simulation` singleton (`lib/core/simulation/controller.ts:1880`).
3. So the contract is one small `EditorHost`, reached from the session, holding the
   ~9 operations UI actually calls — plus making `EditorEntity`'s `apply`/`remove`
   generic over the host's document.
4. Regions become one record in **finest cells** plus a `RegionSpace` adapter per host;
   the snap step becomes the **brick** (8 finest cells), which is what the solver
   actually binds on.
5. The lab's toolstrip stops being hand-written rows and becomes `ComposedFeatureSlot`s
   over an `advanceSliceFeature` — the same slots the studio renders.

---

# PART 1 — Capability audit

`S` = 3-D studio, `L` = 2-D advance lab. "Dup" = re-implemented rather than shared.

| Capability | S definition / policy | S ui / ring / key | L definition | L ui | Dup? | S controller dependency |
|---|---|---|---|---|---|---|
| Edit-mode toggle (Tab) | `lib/core/editor-viewport-mode.ts:21,42,57`; policy `lib/core/stores/ui-store.ts:446-456` | `components/ViewportModeToggle.tsx:21-41`; `components/EditorModeChip.tsx:26-61`; key `lib/core/use-editor-shortcuts.ts:76-80` | shared | `advance-lab/use-slice-shortcuts.ts:88-93` | **key handler only** | none (ui store) |
| Ring composition per target | `lib/core/editor-probe-catalog.ts:529-545` `targetActionsAt`; `lib/core/editor-entity-catalog.ts:234-262` `entityActionsAt`, `:301-316` `sceneActionsAt` | `components/RadialMenu.tsx:21-36`; ring `components/RadialRing.tsx:231-413`; right-click `components/WebGPUViewport.tsx:1695-1727` | `advance-lab/slice-actions.ts:95-247` (`SliceRingContext`, `sliceActionsAt`, `sliceRingTitle`) | `AdvanceLab.tsx:1638-1678` | **yes — whole composer** | `performEditorAction` → `simulation` |
| Armable gestures | `lib/core/editor-gesture-catalog.ts:44-60,106-205,216,225` | rows `components/MakeRows.tsx:47,78,149`; keys `use-editor-shortcuts.ts:221-237`; `components/armed-stroke.ts:26-45` | reuses catalog; `SLICE_GESTURES` filter `use-slice-shortcuts.ts:43-44` | `SliceToolstrip.tsx:213,263` | no (shared) | none |
| Region draw | `lib/core/editor-refinement-region.ts:165-207` `refinementRegionFromDrag` + `:132-154` `snapRefinementRegionBox` | `components/WebGPUViewport.tsx:2109-2124,2895-2899` | `advance-lab/slice-regions.ts:147-154` `draftRegionBox` | `AdvanceLab.tsx:1386-1399` `drawRegion` | **yes** | `beginDraft`/`commitDraft` (`:1602`,`:1615`) |
| Region select | `lib/core/editor-refinement-region.ts:50-60` prefix, `:498-511` `pick` | click + ring `select` | `slice-regions.ts:41-51` prefix **mirrored**, `:91-101` `regionAt` | `AdvanceLab.tsx:1797-1821` | **yes** | none |
| Region move / resize | `:79-89` `refinementRegionResizePolicy`; `editor-entity.ts:838` `resizeBox`, `:765` `moveBoxWithinLimits` | generic handle drag `WebGPUViewport.tsx:1896-1899,2823-2842` | `slice-regions.ts:166-176`, `:186-193` | `SliceRegions.tsx:91-205` | **yes — policy mirrored** | `beginEdit`/`beginDraft`/`commitDraft` |
| Region delete | `editor-refinement-region.ts:475` `remove` | `EntityOptions.tsx:499-512`; key `use-editor-shortcuts.ts:136-143`; ring `entity-catalog.ts:250-260` | — | `AdvanceLab.tsx:1464-1467`; key `use-slice-shortcuts.ts:102-107` | **yes** | `simulation.removeEntity` (`:1731`) |
| Region options rows (MIN/MAX/Means/Remove) | `editor-refinement-region.ts:341-429` `refinementRegionChoices` (generic `EditorChoiceGroup`) | `components/EntityOptions.tsx:310-440` `EntityOptionRows` | ladder `ADVANCE_RUNGS` (`lib/physics-wasm/advance-view.ts:5`); rules `slice-regions.ts:208-232` | **hand-written** `advance-lab/SliceRegions.tsx:217-287` | **yes — the headline** | `beginEdit`+`commitEdit` (`EntityOptions.tsx:331-332,337-338`) |
| Pre-draw region draft (cell size / rule / hold-at-one-tier) | **absent in S** | — | React state `AdvanceLab.tsx:706-708` (`enforceCells`, `holdAtOneTier`; default `2` at `:304`) | `SliceToolstrip.tsx:209-259` `SliceRegionRow` | **L-only; S lacks it** | — |
| Region capacity | `lib/core/refinement-regions.ts:36` = 8; `editor-refinement-region.ts:210-212` | notice `WebGPUViewport.tsx:2111` | `ENFORCEMENT_CAPACITY = 8` `AdvanceLab.tsx:303` | `SliceToolstrip.tsx:222` | **yes (constant)** | — |
| Water drop / ball | gesture `editor-gesture-catalog.ts:112-120`; tools `lib/core/voxel-editor/registry.ts:15`; groups `lib/core/editor-voxel-tool-actions.ts:71-118` | ring `editor-fluid-body.ts:650-726`; row `MakeRows.tsx:72-125` | `slice-actions.ts:171-199` | `SliceToolstrip.tsx:262-272` + `AdvanceLab.tsx:1347-1360` `commitDrop` | **yes (ring + row)** | `beginDraft`/`commitDraft`/`cancelDraft`; `voxel-tool` → `ui.setVoxelTool` |
| Inspect cell / pin probe | `lib/core/editor-probe-actions.ts:54-63`; probe `editor-probe-catalog.ts:297-331`; effect `editor-action.ts:124` | ring `editor-fluid-body.ts:788`; keys `c`,`[`,`]`,`i` `use-editor-shortcuts.ts:178-211` | `slice-actions.ts:156-166` (`host` verb `pin-cell`) | `AdvanceLab.tsx:1506` | **yes** | none (ui store) |
| Ray probe | `editor-probe-actions.ts:34-43`; appended `editor-probe-catalog.ts:566-586` | key `r` `use-editor-shortcuts.ts:190-194` | — | — | n/a | none |
| Field visuals + overlays | `lib/core/visualization-registry.ts:102-115` `VisualizationScalar`, `:189-234` `FieldVisualization`; catalog `lib/core/grid-overlay-visualizations.ts:48-127`; pick `lib/core/field-overlay-pick.ts:61-79` | `components/FieldQuickBar.tsx:118-261` `FieldViewRows`; `components/FluidFieldFlyout.tsx:254-417` | lenses `advance-lab/lenses.ts:269-281,1264-1287`; overlays `:780-799` | `SliceToolstrip.tsx:118-194` | **yes (rows)** — colour policy **already shared** via `lib/core/fluid-fraction-view.ts` (`lenses.ts:26-30`) | none |
| Solver / method controls | `lib/framework/composition.ts:37-47`; placements in `lib/features/*/definition.ts` | `lib/features/ui/FeatureSlot.tsx:33-47`; slots rendered: `scene.visibility`, `scene.surface`, `scene.physics`, `scene.adaptivity`, `scene.simulation` (`components/SceneToolstrip.tsx:251-258`), `fluid.inspection` (`FluidFieldFlyout.tsx:368`) | `lib/methods/adaptive-volume/features/advance-slice/definition.ts` — **declarations only, no `FeatureDefinition`** | hand-written SURFACE `SliceToolstrip.tsx:294-316`, SOLVE `:317-337`, transport `<select>` `AdvanceLab.tsx:1594-1613` | **yes** | `setMethodParam`/`setQuality`/`setMethod` |
| Topology freeze | `lib/features/topology-freeze/definition.ts:9-14` (slot `sim.topology`) | `lib/features/topology-freeze/ui.tsx:6-23`; mounted via `MakeRows.tsx:55` | — | — | n/a | none (runtime store) |
| Scene document verbs | `lib/core/editor-scene-document.ts:31-105` | `SceneToolstrip.tsx:159-206`; ring `entity-catalog.ts:333-342` | — (scene chip) | `AdvanceLab.tsx:1533-1551` | partial | `newScene`/`saveNamedScene`/`importScene`/`setFluidSystem` |
| Camera / view framing | `lib/core/editor-camera-framing.ts:12-39` (keys 0-3) | `use-editor-shortcuts.ts:215-220` | `advance-lab/view-transform.ts:54-198` (2-D zoom/pan) | key `0` `use-slice-shortcuts.ts:108-112` | different geometry — **not shared, correctly** | none |
| Undo / redo | `controller.ts:1338-1339` | `use-editor-shortcuts.ts:49-58` | **none** (history store created `session.ts:49` and unused) | — | L missing | `simulation.undo/redo` |
| Entity toolstrips | `editor-entity.ts:133-235` `EditorEntity` (`fields:167`, `choices:172`, `groups:184`, `remove:220`) | `SceneToolstrip.tsx:226-307` `ContainerToolstrip`, `:318-365` `EntityToolstrip` | — | `SliceToolstrip.tsx:274-351`, `SliceRegions.tsx:217-287` | **yes** | `EntityOptions.tsx` × 9 call sites |

## 1.1 The controller dependency, exactly

Every UI file that cannot be rendered by a second host is one that imports the
module-level `simulation` singleton. Inside the *editor capability* surface that is a
short list:

| File | import | calls |
|---|---|---|
| `components/EntityOptions.tsx` | `:5` | `:85 setFluidSystem`, `:106 rebuildSceneAtLattice`, `:193-194 / :210-211 / :331-332 / :337-338 beginEdit+commitEdit`, `:510 removeEntity`, `:539 dropBody`, `:547 resetBody` |
| `components/SceneToolstrip.tsx` | `:12` | `:65-66 beginEdit+commitEdit`, `:140 setMethod` |
| `components/FluidFieldFlyout.tsx` | `:10` | `:114/:131/:457 setMethodParam`, `:130 resetMethodParam`, `:446 setQuality` |
| `lib/core/editor-action-runtime.ts` | `:13` | `:37/:65 beginEdit`, `:39/:72 commitEdit`, `:52 addBodyAt`, `:60 addScenery`, `:85 dropBody`, `:132 newScene`, `:138 saveNamedScene`, `:166 importScene`, `:181 setFluidSystem` |
| `lib/core/use-editor-shortcuts.ts` | `:14` | `:51/:56 undo/redo`, `:140 removeEntity` |
| `lib/features/gravity/ui.tsx` | `:6` | `:17-18/:48-49/:64-65 beginEdit+commitEdit` |
| `lib/features/pressure-inspection/ui.tsx` | `:6` | `:40 setMethodParam` |

Files that already prove the pattern works without it: `components/MakeRows.tsx`,
`components/SculptRows.tsx`, `components/FieldQuickBar.tsx`, `components/armed-stroke.ts`
(see its own note at `:14-15`), `components/RadialMenu.tsx`,
`lib/features/topology-freeze/ui.tsx`, `lib/features/surface-display/ui.tsx`.

## 1.2 Two audit findings worth acting on

- **Declared slots nobody renders.** `sim.topology` (`topology-freeze/definition.ts:13`),
  `sim.inspection` (`pressure-inspection/definition.ts:6`), `sim.adaptivity`
  (`adaptive-volume/features/adaptivity/definition.ts:202`), `frame.surface`,
  `fluid.material` have placements but no `<FeatureSlot>` render site. Conversely
  `scene.simulation` is rendered (`SceneToolstrip.tsx:258`) with no placement declaring
  it. The `sim.*` family is free real estate: it is exactly where the lab's
  instrument-side rows belong.
- **The prefix is mirrored by hand.** `slice-regions.ts:41` restates
  `REFINEMENT_REGION_SELECTION_PREFIX` (`editor-refinement-region.ts:50`) and pins the
  two with a test. That is the whole duplication story in one constant.

---

# PART 2 — The host contract

## 2.1 What is already host-agnostic (do not re-invent)

- `ComposedFeatureSlot({composition, views, slot})` — `lib/framework/ui/slot.tsx:13-31`.
  Takes its composition and its view bindings as **props**. `FeatureSlot`
  (`lib/features/ui/FeatureSlot.tsx:33-47`) is only the studio's binding of it.
- `RadialRing` — `components/RadialRing.tsx:231-413`; `RadialMenu` already takes
  `perform` as a prop (`components/RadialMenu.tsx:21-23`).
- `PaneSession` and `SessionProvider` — `lib/core/session/session.ts:26-52`,
  `session-context.tsx:15-25`. The lab already builds one.
- `useArmedStroke` / `strokeHint` — `components/armed-stroke.ts:26-45`.
- `EditorChoiceGroup` / `EditorField` / `EditorControlGroup` —
  `lib/core/editor-entity.ts:247-338`. Already a data protocol; only the *return type*
  of `apply` is studio-specific.

## 2.2 The seam: `EditorHost`

One interface, reached from the session, holding what the UI actually calls. Capability
groups rather than one god-method, because the lab has no document to patch and a
document-shaped `commitEdit` cannot be honoured there.

```ts
// lib/core/editor-host.ts   (zone: core)
export interface EditorCommitOptions {
  /** The edit moved the lattice; the run restarts from a defined t = 0. */
  readonly reseed?: boolean;
  readonly announceRebuild?: string;
}

/**
 * What a capability module may ask of the world it is editing.
 *
 * Everything but `id`, `commit`, `select`, `arm` and `notice` is optional: a host
 * declares what it can honour and a row needing a missing capability does not render —
 * the bargain `SimulationMethod.capabilities` makes, one level up.
 */
export interface EditorHost<Doc = unknown, Patch = Doc> {
  readonly id: string;
  /** A whole document, or a merge patch, under one history label. */
  commit(label: string, next: Doc, options?: EditorCommitOptions): void;
  commitPatch?(label: string, patch: Patch, options?: EditorCommitOptions): void;
  /** An open gesture's transient document. */
  draft?: { begin(subject: string, label: string): void;
            commit(options?: EditorCommitOptions): void; cancel(): void };
  history?: { undo(): boolean; redo(): boolean };
  /** Placing liquid, in the host's own frame — the lab has no scene to patch. */
  liquid?: { dropAt(centre: readonly number[], radius: number): void };
  /** Feature-control values, keyed by `FeatureControl.setting`. */
  params?: { get(key: string): number | string | boolean | undefined;
             set(key: string, value: number | string | boolean): void };
  select(selection: EditorSelection | undefined, openControls?: boolean): void;
  arm(gesture: EditorGestureId | undefined): void;
  notice(text: string, tone?: "info" | "warn"): void;
}
```

Reached the way the session is:

```ts
// lib/core/session/host-context.tsx  (zone: core)
export const EditorHostContext = createContext<EditorHost | undefined>(undefined);
export function EditorHostProvider({ value, children }: {...}): JSX.Element;
export function useEditorHost<D = SceneDescription, P = Partial<SceneDescription>>(): EditorHost<D, P>;
```

The studio mounts `<EditorHostProvider value={studioEditorHost(session)}>` beside its
existing `<SessionProvider>`; the lab mounts its own. A missing provider throws — there
is no default, because a silent fall-back to pane A is exactly the bug this removes.

### 2.3 Both fulfilments

```ts
// lib/core/simulation/editor-host.ts   (zone: core; the only new importer of `simulation`)
export function studioEditorHost(session: PaneSession): EditorHost<SceneDescription, Partial<SceneDescription>> {
  const ui = () => session.ui.getState();
  return {
    id: session.id,
    // beginEdit :1154 → setScene → commitEdit :1172, in that order. That triple is
    // the whole of what `{kind:"scene"}` does today (editor-action-runtime.ts:37-39).
    commit: (label, next, o) => { simulation.beginEdit(label, session.id);
      session.scene.getState().setScene(next);
      simulation.commitEdit(undefined, { reseed: o?.reseed }, session.id); },
    commitPatch: (label, patch, o) => { simulation.beginEdit(label, session.id);
      simulation.commitEdit(patch, { reseed: o?.reseed ?? true }, session.id); },
    draft: {                                                        // controller.ts:1602/1615/1626
      begin: (s, label) => simulation.beginDraft(s as SceneDraftSubject, label, session.id),
      commit: (o) => simulation.commitDraft({ reseed: o?.reseed, announceRebuild: o?.announceRebuild }, session.id),
      cancel: () => simulation.cancelDraft(session.id) },
    history: { undo: () => simulation.undo(session.id), redo: () => simulation.redo(session.id) }, // :1338-1339
    params: { get: (k) => resolvedMethodValues(session.method.getState())[k],
      set: (k, v) => simulation.setMethodParam(session.method.getState().methodId, k, v, session.id) }, // :1102
    select: (s, open) => { ui().select(s); if (open) ui().setSelectionControlsOpen(true); },
    arm: (g) => ui().setArmedGesture(g),
    notice: (t, tone = "info") => session.runtime.getState().setNotice(t, tone),
  };
}

// advance-lab/lab-host.ts — the lab's "document" is its region list. No patch form:
// there is no merge to express, and `setRefinementRegions` is a whole-list command.
export function labEditorHost(input: { session: PaneSession; controller: AdvanceLabController;
  setRegions: (next: readonly AdvanceRefinementRegion[]) => void; ny: number;
}): EditorHost<LabDocument, LabDocument> {
  const { session, controller, setRegions, ny } = input;
  const ui = () => session.ui.getState();
  return {
    id: session.id,
    commit: (_label, next) => { setRegions(next.regions);
      void controller.setRefinementRegions(next.regions); },                    // :112
    liquid: { dropAt: ([x, y], r) => void controller.injectLiquid([x, ny - y], r) }, // :108
    params: { get: labParam, set: labSetParam },      // solve budget :96, time step :92
    select: (s, open) => { ui().select(s); if (open) ui().setSelectionControlsOpen(true); },
    arm: (g) => ui().setArmedGesture(g),
    notice: (t) => console.info(t),                   // the lab has no notice surface yet
  };
}
```

`history`, `draft` and `commitPatch` are absent on the lab host, which is the honest
statement: the lab has no undo (`session.history` is built at `session.ts:49` and never
read) and no draft document.

## 2.4 `EditorEntity` becomes document-generic

The entity protocol is already data. Only two return types name `SceneDescription`:
`EditorField.apply` (`editor-entity.ts:267`) and `EditorChoice.apply` (`:292`), plus
`EditorEntity.remove` (`:220`). Parameterise, with defaults so **every existing call site
compiles unchanged**:

```ts
// lib/core/editor-entity.ts — every member unchanged except the three `apply`/`remove`
// return types; defaults keep all existing call sites source-compatible.
export interface EditorChoice<Patch = Partial<SceneDescription>> { /* id,label,hint,enabled */
  readonly apply: () => Patch }
export interface EditorChoiceGroup<Patch = Partial<SceneDescription>> { /* id,label,tag,value */
  readonly options: readonly EditorChoice<Patch>[] }
export interface EditorField<Patch = Partial<SceneDescription>> extends ControlMetadata {
  readonly apply: (value: number) => Patch }
export interface EditorEntity<Patch = Partial<SceneDescription>, Doc = SceneDescription> {
  readonly choices?: readonly EditorChoiceGroup<Patch>[];
  readonly fields?: readonly EditorField<Patch>[];
  readonly remove?: () => Doc }
export interface EditorEntityDefinition<Patch = Partial<SceneDescription>, Doc = SceneDescription> {}

// components/EntityOptions.tsx — the host replaces the singleton; markup untouched.
export function EntityOptionRows<P, D>({ entity }: { entity: EditorEntity<P, D> }) {
  const host = useEditorHost<D, P>();
  const commitChoice = (g: EditorChoiceGroup<P>, o: EditorChoice<P>) =>
    (host.commitPatch ?? commitWhole)(`Set ${entity.label} ${g.label}`, o.apply(), { reseed: true });
  /* …rows exactly as today, EntityOptions.tsx:341-440 */
}
```

That single change makes the lab's `SliceRegionToolstrip` (`SliceRegions.tsx:217-287`)
deletable: MIN/MAX/Remove are already `EditorChoiceGroup`s under another name.

## 2.5 One region record, one `RegionSpace`

```ts
// lib/features/refinement-region/definition.ts    (zone: feature)
/** A refinement box, in FINEST CELLS, N-dimensional. The one record both hosts store. */
export interface RefinementRegionRecord {
  readonly id: string;
  /** Length 2 in the lab, 3 in the studio. Axis-major, y-up, origin at the lattice corner. */
  readonly min_cells: readonly number[];
  readonly max_cells: readonly number[];
  readonly rule: RefinementRegionRule;            // lib/core/refinement-regions.ts:62
  readonly minimumCellSize_cells: number;          // power of two
  readonly maximumCellSize_cells?: number;         // absent = AUTO
}

export interface RegionLattice { readonly dimensions: readonly number[] }

/** How a host stores, names and writes back the records. One adapter per host. */
export interface RegionSpace<Doc, Patch = Doc> {
  readonly axes: 2 | 3;
  /** Finest cells across the domain, per axis. */
  lattice(doc: Doc): RegionLattice;
  list(doc: Doc): readonly RefinementRegionRecord[];
  /** The document with one record replaced, added, or — with `undefined` — dropped. */
  write(doc: Doc, id: string, next: RefinementRegionRecord | undefined): Patch;
  nextId(doc: Doc): string;
  readonly capacity: number;                        // both are 8
  /** The ladder this host offers. S: [1,2,4,8,16,32]; L: [1,2,4,8]. */
  readonly cellSizes: readonly number[];
  /** Brick width in finest cells. Both are 8 — see §2.6. */
  readonly brick_cells: number;
}
```

Adapters:

- **3-D** `lib/core/editor-refinement-region.ts` becomes the adapter. `list` converts
  `min_m/max_m` → cells via `refinementRegionCellBounds`
  (`lib/core/refinement-regions.ts:122-135`); `write` converts back with
  `refinementRegionLattice` (`:111-119`) and calls the existing `withRefinementRegion`
  (`editor-refinement-region.ts:92-110`). `cellSizes` =
  `OCTREE_REFINEMENT_REGION_CELL_SIZES` (`refinement-regions.ts:46`); `capacity` =
  `OCTREE_REFINEMENT_REGION_CAPACITY` (`:36`); `brick_cells` = the new core constant
  `BRICK_FINE_CELLS` (§2.11 — `feature` may not import a method zone, so the 8 that
  `sparse-brick-atlas.ts:50` and `advance-view.ts:4` hold moves into core).
- **2-D** `advance-lab/lab-region-space.ts`. `AdvanceRefinementRegion`
  (`lib/physics-wasm/advance-controller.ts:22-28`) is already the record modulo names
  (`minimumFine`/`maximumFine`/`minimumCellWidth`/`maximumCellWidth`). `cellSizes` =
  `ADVANCE_RUNGS` (`advance-view.ts:5`); `brick_cells` = `ADVANCE_BRICK_FINE`
  (`advance-view.ts:4`).

Shared policy, N-dimensional, no metres and no `SceneDescription`:

```ts
// lib/features/refinement-region/policy.ts
export function regionSnapStep_cells(record: Pick<RefinementRegionRecord,"minimumCellSize_cells">, brick_cells: number): number;
export function snapRegionBox(min: readonly number[], max: readonly number[], step: number, lattice: RegionLattice): { min: number[]; max: number[] };  // outward
export function resizeRegionBox(box, side: { axis: number; end: "min"|"max" }, to: number, step, lattice): { min: number[]; max: number[] };
export function moveRegionBox(box, delta: readonly number[], step, lattice): { min: number[]; max: number[] };
export function regionWithFloor(record: RefinementRegionRecord, cells: number): RefinementRegionRecord;    // hold-at-one-tier follows the floor
export function regionWithCeiling(record: RefinementRegionRecord, cells: number | undefined): RefinementRegionRecord;
export function regionCaption(record: RefinementRegionRecord, cellEdge_mm?: number): string;
export function regionChoices<Doc, Patch>(space: RegionSpace<Doc, Patch>, doc: Doc, record: RefinementRegionRecord): EditorChoiceGroup<Patch>[];
```

`regionChoices` is `refinementRegionChoices` (`editor-refinement-region.ts:341-429`)
with the metres arithmetic lifted into the adapter — the three groups (`rule`/"Means",
`minimumCellSize`/"Min", `maximumCellSize`/"Max") and their re-snap-on-change behaviour
stay verbatim.

## 2.6 The snap step is the **brick**, not the region's floor cell

Today's 3-D step is the region's own smallest allowed cell
(`editor-refinement-region.ts:79-90`, `:83`). With `MIN = 1` that is one finest cell,
which is why resizing feels smooth. The solver does not bind at that granularity:

- `lib/methods/adaptive-volume/sparse-cm12-refinement-regions.ts:68-93`
  `cellSizeBoundsForBrick` raises the **floor** for every brick that *intersects*
  (`:82-84`) and lowers the **ceiling** only for bricks *fully contained* (`:85-91`).
- `rust/crates/fluid-core/src/resolution.rs:410-438` `region_bounds` is the same test,
  brick by brick, with `B = 8`.

So a region edge inside a brick changes nothing for the ceiling and over-applies the
floor. The honest step is the brick, and because both the brick width and the ladder are
powers of two, one rule serves both constraints:

```ts
export function regionSnapStep_cells(record, brick_cells) {
  return Math.max(brick_cells, clampRegionCellSize(record.minimumCellSize_cells));
}
```

`brick_cells = 8` on both sides (`sparse-brick-atlas.ts:50`, `advance-view.ts:4`).
Consequences: with the studio default `MIN = 8`
(`lib/core/refinement-regions.ts:49`) the step is unchanged, so **default studio
behaviour is byte-identical**. `MIN` of 1/2/4 now snaps to 8 instead of 1/2/4 — the fix
Peter asked for. `MIN` of 16/32 keeps its own coarser step. In the lab, whose default is
`2` (`AdvanceLab.tsx:304`), the step becomes 8 — a visible change, and the right one.

## 2.7 The pre-draw draft is shared ui-store state, and `RegionRow` is one component

The lab's `enforceCells` / `holdAtOneTier` (`AdvanceLab.tsx:706-708`) are the dropdown
Peter liked. Move them into the ui store so both hosts' draw commit reads the same value:

```ts
// lib/core/stores/ui-store.ts  (beside placementShape/placementDimensions :413 ff.)
regionDraft: { cellSize_cells: number; rule: RefinementRegionRule; holdAtOneTier: boolean };
setRegionDraft: (patch: Partial<UIState["regionDraft"]>) => void;
```

Then one component set in `lib/features/refinement-region/ui.tsx`, plus the shared draw:

```tsx
export function RegionRow({ space, doc }: { space: RegionSpace<any,any>; doc: unknown })  // arms `region-draw`
export function RegionDraftMenu({ space }: { space: RegionSpace<any,any> })               // the cell-size chooser
export function RegionOptionRows({ entity })                                             // = EntityOptionRows(regionChoices)
export function regionFromDraw<Doc, Patch>(space: RegionSpace<Doc, Patch>, doc: Doc,
  anchor_cells: readonly number[], at_cells: readonly number[], draft: UIState["regionDraft"]): RefinementRegionRecord;
```

`RegionRow` = today's `MakeRows.tsx:46-57` plus the lab's draft menu
(`SliceToolstrip.tsx:227-257`): mark, `strokeHint("region-draw", armed)`,
`after={<><RegionDraftMenu/><TopologyFreezeButton/></>}`, capacity in the hint
(`refinementRegionCapacityRemaining`). `components/MakeRows.tsx` re-exports it so
`MakeRows({fluid})` keeps its signature and `SceneToolstrip.tsx:263` does not move —
that is how 3-D gains the dropdown. 3-D keeps its screen→world→cells conversion and its
height seeding (`editor-refinement-region.ts:165-197`) in the adapter; the snap and the
record are shared.

## 2.8 Visuals: one `FieldViewRows`, injected state

`lib/core/fluid-fraction-view.ts` is **already** shared (`advance-lab/lenses.ts:26-30`,
`lib/core/grid-overlay-visualizations.ts:14`, `lib/core/webgpu-grid-overlay.ts:11`).
The gap is the row, not the colour policy. `FieldViewRows`
(`components/FieldQuickBar.tsx:118-261`) reads six ui-store members directly
(`:121-126`) and `getMethod(methodId)` (`:80-83`). Split it:

```tsx
// lib/features/field-view/ui.tsx
export interface FieldViewState {
  readonly mode: string; readonly axis: GridOverlayConfig["axis"]; readonly slice: number;
  readonly setMode: (m: string) => void; readonly setAxis: (a: GridOverlayConfig["axis"]) => void;
  readonly setSlice: (s: number) => void;
}
export function FieldViewRows({ views, state, volumeCapable }: {
  readonly views: readonly FieldVisualization[];   // already the registry's own shape
  readonly state: FieldViewState;
  readonly volumeCapable: boolean;
}): JSX.Element | null;
```

The studio passes `VISUALIZATION_QUICK_FIELDS` narrowed by
`getMethod(methodId).supportedFieldModes` and the ui store's six members — markup
unchanged. The lab passes `ADVANCE_LENSES` (`lenses.ts:1264-1287`) and `SLICE_OVERLAYS`
(`:799`) re-declared as `fieldVisualization(...)` entries carrying an `icon` and, where
they draw a shared quantity, the `scalar` (`visualization-registry.ts:102-115`) taken
from `fluid-fraction-view`. `advance-lab/SliceToolstrip.tsx:118-194` (`LensRow`,
`OverlayRow`) then deletes.

## 2.9 The lab's toolstrip is `FeatureSlot`s — and the lab is **not** a `SimulationMethod`

`SimulationMethod` (`lib/core/method-contract.ts:429-492`) is a GPU contract:
`GPUSolverInstance`, `resolveComposition`, `presetFor(quality)`, overlay pipelines,
~80 optional render sources. The Rust/Wasm lab world satisfies none of it and would
have to stub all of it. **Do not register it.** What the lab needs is the *composition*
half, which `composeFeatures` gives without any registry:

```ts
// lib/methods/adaptive-volume/features/advance-slice/definition.ts  (add to the existing file)
export const advanceSliceFeature = {
  id: "simulation.advance-slice",
  controls: [
    { id: "lens",      label: "Stage lens",  kind: "choice", setting: "advanceLens",     update: "live", options: /* ADVANCE_STAGE_ORDER */ },
    { id: "surface",   label: "Surface",     kind: "choice", setting: "advanceSurface",  update: "live", options: /* ADVANCE_SURFACE_VIEWS :180 */ },
    { id: "budget",    label: "Solve",       kind: "number", setting: "pressureIterations", min: 4, max: 256, step: 4, update: "live" },
    { id: "transport", label: "Transport",   kind: "choice", setting: "transportExperiment", update: "reset", options: /* ADVANCE_TRANSPORT_EXPERIMENTS :228 */ },
    { id: "overlays",  label: "Overlays",    kind: "choice", setting: "advanceOverlays", update: "live" },
  ],
  placements: [
    { slot: "scene.surface",    control: "surface",   priority: "high" },
    { slot: "scene.visibility", control: "lens",      priority: "high" },
    { slot: "scene.visibility", control: "overlays" },
    { slot: "sim.solve",        control: "budget",    presentation: "expanded" },
    { slot: "sim.transport",    control: "transport" },
  ],
} as const satisfies FeatureDefinition;
```

and lab-side:

```tsx
// advance-lab/LabFeatureSlot.tsx
const labViews: FeatureControlViews = { "simulation.advance-slice/lens": LensRow, /* … */ };
const composition = composeFeatures({ features: [advanceSliceFeature, topologyFreezeFeature] });
export const LabFeatureSlot = ({ slot }: { slot: string }) =>
  <ComposedFeatureSlot composition={composition} views={labViews} slot={slot} />;
```

Each control's row reads and writes through `host.params` (§2.2), so the *same* row
component can be bound in `applicationViews` (`lib/features/ui/FeatureSlot.tsx:18-31`)
when the studio ever wants it. `SliceToolstrip.tsx` collapses to the `<Toolstrip>`
frame plus five `<LabFeatureSlot slot="…"/>` calls plus `<RegionRow/>` and `<WaterRow/>`.

## 2.10 Module layout

```
lib/core/
  editor-host.ts                       EditorHost, EditorCommitOptions            (new)
  session/host-context.tsx             EditorHostProvider, useEditorHost          (new)
  simulation/editor-host.ts            studioEditorHost(session)                  (new; only new `simulation` importer)
  editor-entity.ts                     + <Patch, Doc> type params, defaults kept
  editor-refinement-region.ts          → thin 3-D RegionSpace adapter (~120 lines, from 512)
  stores/ui-store.ts                   + regionDraft / setRegionDraft

lib/features/refinement-region/        (new package; zone "feature")
  definition.ts   RefinementRegionRecord, RegionSpace, RegionLattice, ladder re-export
  policy.ts       snap / resize / move / floor / ceiling / caption / regionChoices / regionFromDraw
  ui.tsx          RegionRow, RegionDraftMenu, RegionOptionRows
  ring.ts         regionWedges(space, doc, target) → EditorAction[]
  verification/   policy.test.ts, rows.test.tsx, snap.test.ts

lib/features/liquid-drop/              (new)
  definition.ts   LiquidDropSpec { centre, radius } in host cells
  ui.tsx          WaterRow  (today's components/MakeRows.tsx:72-125, host-injected shapes)
  ring.ts         liquidWedges(...)  (today's editor-fluid-body.ts:650-692 "Water" wedge)

lib/features/inspect/                  (new)
  ring.ts         cellProbeWedge / rayProbeWedge  (from lib/core/editor-probe-actions.ts)

lib/features/field-view/               (new)
  ui.tsx          FieldViewRows({views, state, volumeCapable})

components/
  EntityOptions.tsx   uses useEditorHost; markup unchanged
  MakeRows.tsx        re-exports RegionRow / WaterRow; MakeRows() unchanged
  FieldQuickBar.tsx   thin studio binding of lib/features/field-view

advance-lab/
  lab-host.ts           labEditorHost(...)                     (new)
  lab-region-space.ts   RegionSpace over AdvanceRefinementRegion (new)
  LabFeatureSlot.tsx    composition + views                     (new)
  AdvanceLab.tsx        mounts EditorHostProvider; keeps drawing/pointer/playback
  SliceRegions.tsx      keeps the SVG painting; SliceRegionToolstrip DELETED (:217-287)
  slice-regions.ts      DELETED (policy moves to lib/features/refinement-region/policy.ts)
  slice-actions.ts      DELETED (ring composed from feature ring.ts modules)
  SliceToolstrip.tsx    ~60 lines: frame + LabFeatureSlot ×5 + RegionRow + WaterRow
  lenses.ts             stays (the picture); + fieldVisualization() declarations
  view-transform.ts     stays (2-D camera; genuinely not shared)
```

## 2.11 Module boundaries

Read `tools/check-module-boundaries.ts:50-70` (zones) and `:72-145` (`ALLOWED`) before
touching anything. Facts:

- `advance-lab/` is **not in `roots`** (`:203`), so nothing in it is checked, and the one
  edge that is — `app/advance-lab/page.tsx [ui] → advance-lab/AdvanceLab.tsx [tooling]`
  — is already a reported violation (6 at HEAD).
- `feature` → `["framework","core","feature","ui"]` (`:74`). So
  `lib/features/refinement-region/*` may import `lib/core` and `components/` — good —
  but **not** `lib/physics-wasm` (`lib-other`) and **not** `lib/methods/*`.
- `method-adaptive-volume` → `["core","method-adaptive-volume","sparse-world"]` (`:85-87`),
  which is why `advance-lab/lenses.ts` could not move under the method: it needs
  `lib/physics-wasm` types.

Two changes, both small:

1. **Add an `advance-lab` zone.** `zoneOf`: `if (relPath.startsWith("advance-lab/")) return "advance-lab";`
   `ALLOWED["advance-lab"] = new Set(["core","ui","feature","composition","framework","lib-other","method-adaptive-volume","tooling"])`,
   and add `"advance-lab"` to `ALLOWED.ui` and to `ALLOWED.tooling`. Add `"advance-lab"`
   to `roots` (`:203`). Net effect: the lab is policed for the first time, and the
   existing `ui → tooling` violation is retired.
2. **`BRICK_FINE_RESOLUTION` must be reachable from `feature`.** Do **not** widen
   `feature → method-adaptive-volume`. Instead re-export the constant from core:
   `lib/core/sparse-brick-geometry.ts` holding `export const BRICK_FINE_CELLS = 8;`,
   with `sparse-brick-atlas.ts:50` and `advance-view.ts:4` importing it (core is
   reachable from every zone). A test pins the three values equal.

No other edge changes. `lib/features/refinement-region` reaching `components/toolstrip`
is already allowed (`:74`), and the existing `featureView` escape at `:232` is not needed.

## 2.12 Doctrine check

- **No persistent panels** — `RegionRow` is a toolstrip row, `RegionOptionRows` appear
  only on a selection, the draft chooser is a `ToolstripMenuButton` on an existing row.
- **Verbs on the ring / selection** — `regionWedges`, `liquidWedges` and the probe wedges
  are ring contributions; `EditorHost` never exposes a verb to a shelf.
- **Instruments, not verbs** — lens/overlay/surface/solve are `FeatureControl`s with
  placements, not `EditorAction`s.
- **Prominence declared** — `FeaturePlacement.priority` (`lib/framework/composition.ts:18-24`),
  sorted by `ComposedFeatureSlot` (`lib/framework/ui/slot.tsx:18-23`).
- **Two-level ring** unchanged (`editor-action.ts:265-270`). **Tests pin invariants**:
  rendered rows, numeric snap results, catalog equality — never source text.

---

# PART 3 — Migration plan

Nine packages. One agent each. `G` = gates: `npx tsc --noEmit`, `npm run test:unit`,
`node --import tsx tools/check-module-boundaries.ts` (violation count must not rise
above the 6 at HEAD, and WP1 lowers it to 5).

### WP0 — `EditorHost` seam (no behaviour change)
- **Owns** `lib/core/editor-host.ts`, `lib/core/session/host-context.tsx`,
  `lib/core/simulation/editor-host.ts`, `components/AppShell.tsx` (mount the provider),
  `advance-lab/AdvanceLab.tsx` (mount a stub host).
- **Invariant** nothing renders differently; `studioEditorHost` is not yet consumed.
- **Tests** `lib/core/verification/editor-host.test.ts`: `studioEditorHost(session).commit`
  issues `beginEdit` then `setScene` then `commitEdit` in that order (fake controller);
  `useEditorHost` throws without a provider.
- **G** all three. **Browser-only** none.

### WP1 — module-boundary zone + shared brick constant
- **Owns** `tools/check-module-boundaries.ts`, `lib/core/sparse-brick-geometry.ts` (new),
  `lib/methods/adaptive-volume/sparse-brick-atlas.ts:50`,
  `lib/physics-wasm/advance-view.ts:4`.
- **Invariant** both constants keep the value 8; no import cycle.
- **Tests** `tests/brick-fine-resolution.test.ts` asserts
  `BRICK_FINE_CELLS === BRICK_FINE_RESOLUTION && BRICK_FINE_CELLS === ADVANCE_BRICK_FINE`.
- **G** boundary checker reports **5** violations (the `ui → tooling` lab-route one is gone).
- Independent of WP0; may run in parallel.

### WP2 — `EditorEntity` generics + `EntityOptionRows` on the host
- **Owns** `lib/core/editor-entity.ts` (type params only),
  `components/EntityOptions.tsx` (replace nine `simulation.*` sites with `host.*`).
- **Invariant** studio markup **byte-identical**: same `testId`s
  (`entity-option-<id>`), same row order (choices → fields → groups,
  `EntityOptions.tsx:341-440`), same commit labels (`Set <label> <group>`), same
  `{reseed:true}`.
- **Tests** `components/verification/entity-options.test.tsx` renders a fabricated
  `EditorEntity` against a recording host and asserts the emitted
  `(label, patch, options)` triples; a second case renders the *same* entity under a
  lab-shaped host and asserts an identical row tree.
- **Depends on** WP0. **G** all three.

### WP3 — `lib/features/refinement-region` package (policy + record)
- **Owns** the whole new package except `ui.tsx`;
  `lib/core/editor-refinement-region.ts` becomes the 3-D `RegionSpace` adapter.
- **Invariant** `snapRefinementRegionBox` behaviour is preserved **for `MIN ≥ 8`**; the
  URL round-trip (`editor-refinement-region.ts:249-337`) is unchanged; the hold-at-one-tier
  rule (`:381-390` and `slice-regions.ts:208-218`) has one implementation.
- **Tests** `lib/features/refinement-region/verification/policy.test.ts`:
  (a) **the snap default** — `regionSnapStep_cells({minimumCellSize_cells: n}, 8)` is
  `8,8,8,8,16,32` for `n = 1,2,4,8,16,32`;
  (b) a drawn box snaps **outward** and is idempotent to 1e-6 after a metres round trip;
  (c) a resize never inverts and never leaves the lattice;
  (d) 2-D and 3-D `RegionSpace` adapters round-trip the same record.
- **Depends on** WP1 (brick constant). **G** all three.
- **Browser-only** that a 3-D region drag now *feels* stepped.

### WP4 — `RegionRow`, draft state, `RegionOptionRows`
- **Owns** `lib/features/refinement-region/ui.tsx`, `lib/core/stores/ui-store.ts`
  (`regionDraft`), `components/MakeRows.tsx` (re-export).
- **Invariant** `MakeRows({fluid})` signature and `scene-region-row` testId unchanged;
  `TopologyFreezeButton` stays in `after` (`MakeRows.tsx:55`).
- **Tests** `lib/features/refinement-region/verification/rows.test.tsx`: **both hosts
  render the same rows for the same definition** — mount `RegionOptionRows` over the
  studio `RegionSpace` and over the lab one and assert equal `testId`/`tag`/option-id
  sequences; the draft menu writes `ui.regionDraft` and the draw commit reads it.
- **Depends on** WP2, WP3. **G** all three.
- **Browser-only** the 3-D cell-size dropdown Peter asked for.

### WP5 — lab adopts the region package
- **Owns** `advance-lab/lab-region-space.ts`, `advance-lab/lab-host.ts`,
  `advance-lab/SliceRegions.tsx` (keep the SVG, delete `:217-287`),
  delete `advance-lab/slice-regions.ts` + its test, rewire `AdvanceLab.tsx:1386-1489`.
- **Invariant** lab testIds `slice-region-min|max|remove` are replaced by the shared
  `entity-option-*`; that is intended and must be stated in the commit.
- **Tests** the region tests from WP3/WP4 now cover the lab; add
  `advance-lab/lab-region-space.test.ts` pinning the y-flip
  (`slice-regions.ts:66-81`) through the adapter.
- **Depends on** WP4. **G** all three.

### WP6 — rings become feature modules
- **Owns** `lib/features/liquid-drop/ring.ts`, `lib/features/inspect/ring.ts`,
  `lib/features/refinement-region/ring.ts`, `lib/core/editor-fluid-body.ts:650-726`
  (re-point), delete `advance-lab/slice-actions.ts` + its test, rewire
  `AdvanceLab.tsx:1500-1509,1638-1678`.
- **Invariant** the studio's ring is unchanged wedge-for-wedge: `entityActionsAt`
  (`editor-entity-catalog.ts:234-262`) still appends `select` then `delete`, `delete`
  still last and `danger`.
- **Tests** `tests/editor-scene-ring.test.ts` (existing) must still hold that
  **no studio catalog emits a `host` effect** — extend it to walk every wedge from
  `sceneActionsAt` and `entityActionsAt` for every entity kind and assert
  `effect.kind !== "host"`; add the mirror for the lab (every lab wedge that is not
  `arm`/`select` **is** a `host` effect).
- **Depends on** WP0. Disjoint from WP3-WP5 except `AdvanceLab.tsx` — sequence after WP5.

### WP7 — `advanceSliceFeature` + `LabFeatureSlot`; the lab toolstrip is placements
- **Owns** `lib/methods/adaptive-volume/features/advance-slice/definition.ts` (append the
  `FeatureDefinition`), `advance-lab/LabFeatureSlot.tsx`,
  `advance-lab/SliceToolstrip.tsx` (collapse to the frame + slots).
- **Invariant** every control the lab has today is still reachable, in the same order:
  lens, overlays, surface, solve, region, water.
- **Tests** `advance-lab/lab-feature-slot.test.tsx`: **the lab's toolstrip is composed
  from placements, not hand-written rows** — assert the rendered row order equals
  `composeFeatures(...).placements.filter(slot)` order, and that removing a placement
  removes exactly one row.
- **Depends on** WP0, WP6. **G** all three.

### WP8 — one `FieldViewRows`
- **Owns** `lib/features/field-view/ui.tsx`, `components/FieldQuickBar.tsx` (thin
  binding), `advance-lab/lenses.ts` (`fieldVisualization` declarations), delete
  `SliceToolstrip.tsx:118-194`.
- **Invariant** studio `field-quick-bar` markup byte-identical; `pickFieldOverlay`
  (`lib/core/field-overlay-pick.ts:61-79`) untouched.
- **Tests** `lib/features/field-view/verification/rows.test.tsx` renders the same
  `FieldVisualization[]` under a studio state object and a lab one and asserts equal row
  trees; a case asserts the `scalar` bands drawn in the legend come from
  `fluid-fraction-view` rather than from a literal.
- **Depends on** WP7. **G** all three.
- **Browser-only** that the lab's lens list still reads correctly at 16 entries.

### Order

```
WP0 ──┬── WP2 ── WP4 ── WP5 ── WP6 ── WP7 ── WP8
WP1 ──┴── WP3 ──┘
```

WP0 and WP1 in parallel; WP2 and WP3 in parallel after them; then the chain.
WP6 and WP7 both touch `AdvanceLab.tsx`, so they are sequential.

### What only the browser can confirm

1. That a 3-D region resize now *steps* by a brick and reads as deliberate rather than
   sticky at `MIN = 1`.
2. That the ring opened on a lab region and on a studio region shows the same wedges in
   the same places.
3. That `RegionRow`'s new dropdown does not push the container strip past its column
   width at the narrowest viewport.
4. That collapsing `SliceToolstrip` to slots does not reorder rows under the lab's own
   CSS module (`AdvanceLab.module.css`).

---

# The three riskiest decisions

**1. Capability-scoped `EditorHost` vs. a single document-shaped `commitEdit`.**
A single `commitEdit(patch)` is smaller, but the lab has no document: its world is a
Rust `World` behind `AdvanceLabController.setRefinementRegions` / `injectLiquid`
(`advance-controller.ts:108-116`), both async and both whole-list. Forcing a patch shape
there means inventing a fake document and a reducer for it.
**Recommend the capability-scoped host**, with `commit`/`commitPatch` as the only
document members and everything else (`liquid`, `params`, `draft`, `history`) optional,
so a host declares what it can honour and a row that needs a missing capability does not
render.

**2. Registering the lab world as a `SimulationMethod`.**
Tempting, because it would make `FeatureSlot` work unchanged. But
`SimulationMethod` (`lib/core/method-contract.ts:429-492`) is a GPU contract —
`GPUSolverInstance`, `presetFor(quality)`, overlay pipeline factories, ~80 render
sources — and a Wasm 2-D slice satisfies none of it. Stubs would be load-bearing lies
the method registry then hands to the renderer.
**Recommend not registering it.** `composeFeatures` + `ComposedFeatureSlot` are already
free-standing (`lib/framework/ui/slot.tsx:13`), so the lab gets identical slot rendering
for the price of a 12-line `LabFeatureSlot.tsx`. Revisit only if the lab ever needs a
`quality` preset.

**3. Changing the region snap step to the brick.**
This changes shipped 3-D behaviour for `MIN < 8` and changes the lab's default
(`MIN = 2`) visibly. The alternative — keeping the floor-cell step and adding a
"snap to bricks" toggle — is a second control for something that has one correct answer:
the solver binds per brick on *intersection* for the floor and on *containment* for the
ceiling (`sparse-cm12-refinement-regions.ts:82-91`, `resolution.rs:415-430`), so an edge
inside a brick is meaningless.
**Recommend the change**, as `max(brick, floor)` so the studio default (`MIN = 8`,
`refinement-regions.ts:49`) is byte-identical and only the smooth cases move. If Peter
wants the finer step back for a specific experiment, it is one line in
`regionSnapStep_cells` and a `RegionSpace.brick_cells` of 1 — not a UI toggle.

---

# Deviations (WP0–WP6, as landed)

Recorded per the brief: each is a signature or a placement that had to change to
be implementable. Nothing below widens the contract; all of them narrow, add an
optional parameter, or move a file the doc placed somewhere the runner cannot
see.

**WP0.**
1. *The provider mounts beside each `SessionProvider`, not in `AppShell`.*
   `components/AppShell.tsx` has no session — `components/FluidLab.tsx:65` mounts
   pane A's and `components/CompareHost.tsx:173` mounts pane B's. One host per
   realm is the whole point of the seam, so the provider went beside each of
   those two, memoised on its session. Mounting one host in `AppShell` would have
   given pane B pane A's host, which is exactly the bug `useEditorHost`'s missing
   default exists to prevent.
2. *`studioEditorHost(session, controller?)`.* The controller arrives as an
   optional second parameter defaulting to the `simulation` singleton, typed as
   `StudioEditorHostController` — the eight methods the host actually calls. The
   order of `beginEdit → setScene → commitEdit` **is** the contract (a `setScene`
   first would snapshot the post-edit document and silently make the edit
   un-undoable) and that order is only observable to a test that can watch the
   calls. Every existing call site still reads `studioEditorHost(session)`.
3. *`EditorHost.draft.commit` returns `void`.* `commitDraft` returns a boolean on
   the controller; the interface in §2.2 declares `void`, so the host swallows it.

**WP1.**
4. *`advance-lab/AdvanceLab.tsx` joins `COMPOSITION_ROOTS`.* Adding `advance-lab`
   to `roots` polices the lab for the first time, which retires the
   `ui → tooling` violation as §2.11 predicted — but it also exposes
   `import "../lib/methods"` at `AdvanceLab.tsx:113` as a method-catalog edge.
   The lab is a genuine module-graph entry point (its route's page is a server
   component, so nothing the studio's shell installs is evaluated for it), which
   is the same rule the other four roots are listed under. Without this the count
   would have stayed at 6 rather than falling to 5.

**WP2.**
5. *The test lives at `tests/entity-options-host.test.tsx`, not
   `components/verification/`.* `tools/run-feature-tests.ts:24` discovers `tests`,
   `lib` and `advance-lab` only, so a test under `components/` would never run.
6. *`EditorControlGroup` is parameterised too.* §2.4 lists three types; the
   entity's `groups` carry `choices` and `fields` of their own, so `EditorEntity`
   could not be generic without it.
7. *`EntityMoreRow`, `EntitySceneTab` and `EntityObjectTab` take
   `EditorEntity<unknown, unknown>`.* They read an entity and commit nothing, so
   they are indifferent to its document types; `unknown` is the covariant widening
   that accepts every instantiation.
8. *Two exports added to `components/EntityOptions.tsx`:* `patchCommitter` (how a
   patch lands on a host with no patch form) and `entityCommitLabel` (the
   `Set <entity> <control>` history label, now written once). `renderToStaticMarkup`
   cannot press a control, so the commit triple is unreachable through markup;
   these are the two functions that *are* the routing and the label, and the test
   drives them directly rather than asserting on source text.
9. *`EntityDeleteRow` spells out `simulation.removeEntity` through the seam,* in
   its order: `select(undefined)`, then `commit(label, remove(), {reseed:true})`,
   then `notice(label)`. The clear must precede the commit because a re-seeding
   `commitEdit` captures and restores whatever is selected
   (`lib/core/simulation/controller.ts`, `commitEdit`). `documentSnapshot` reads only the scene and the
   preset id, so moving the clear ahead of `beginEdit` changes nothing else.

**WP3.**
10. *`regionWithFloor(record, cells, lattice?, brick_cells?)` and
    `regionWithCeiling(record, cells, lattice?, brick_cells?)`.* Changing the floor
    can change `regionSnapStep_cells`, and a box left on the old lattice loses a
    shell of cells to partial containment. Given a lattice these re-snap; the
    doc's two-argument call is still valid and returns the bounds change alone.
11. *`RegionSpace` gained `cellEdge_mm?(doc)`.* §2.5's `regionChoices(space, doc,
    record)` takes no millimetre argument, but the 3-D option hints read
    `8³ finest cells · 40 mm edge`. The adapter is the only thing that knows a
    metre scale, so it supplies one; the lab omits it and the same rows render
    without the clause rather than with a fabricated number.
12. *`regionCaption` serves both phrasings.* With a `cellEdge_mm` it is the
    studio's millimetre sentence, without one the lab's `held at N` / `≥ N cells`.
    The branching — no ceiling, a ceiling equal to the floor, a range — is the
    part that was written twice.
13. *The hold-at-one-tier rule is the lab's, and it is a studio behaviour change.*
    A ceiling equal to the floor now follows the floor **down** as well as up
    (`advance-lab/slice-regions.ts:201-218`); the studio only ever lifted it
    (`Math.max(cells, …)` at `editor-refinement-region.ts:440` in HEAD). One implementation was the
    instruction, and the lab's is the one that was actually stated.
14. *`resizeRegionBox` / `moveRegionBox` are not yet on the 3-D path.* The studio
    resizes through `boxResizeDrag` with a `BoxResizePolicy`, and only that
    policy's *step* changed (to the brick). The package's own N-dimensional
    resize and move are pinned by `verification/policy.test.ts` and are what WP5
    binds lab-side; re-routing the 3-D gizmo through them is a behavioural change
    WP3 was told to avoid.
15. *`lib/features/refinement-region/ui.tsx` is not in this change.* It is WP4's,
    per PART 3. `ring.ts` is here and is not yet consumed — WP6 re-points
    `editor-fluid-body.ts:717-723` at it.

**WP4.**
16. *`TopologyFreezeButton` rides on an `after` prop rather than inside the shared
    row.* §3.1 has the shared `RegionRow` keep the freeze toggle. It cannot: the
    freeze is a *solver-wide* verb (`lib/features/topology-freeze/ui.tsx` reads the
    studio's `simulation` singleton and has no meaning in the lab, which freezes
    nothing). So `RegionRow({space, doc, after?})` renders arm switch, count and
    flyout, and `components/MakeRows.tsx` passes `after={<TopologyFreezeButton />}`
    — the button sits exactly where it sat, following the chooser, and the lab
    simply passes nothing.
17. *`regionEntity(space, doc, record)` is the entity **minus the gizmo**.* It
    carries identity `WORLD_FRAME` and no handles, because a 2-D canvas has no
    3-D box gizmo and the rows never read one. `lib/core/editor-refinement-region.ts`
    spreads it and augments it with the studio's frame, handles and resize policy,
    so the two agree on `groups`/`remove` by construction rather than by review.
18. *`RegionOptionRows` / `RegionDeleteRow` are thin wrappers over the shared
    `EntityOptionRows` / `EntityDeleteRow`,* not new row components. §3.1 named the
    shared components directly; a caller would otherwise have to build the entity
    itself, which is the duplication this package exists to end.
19. *`RegionDraft` lives in `lib/core/refinement-regions.ts`, not in the feature.*
    The draft is UI-store state (`lib/core/stores/ui-store.ts`), and putting its
    type in `lib/features/…` would have pulled the feature package into the store's
    import graph in both hosts. `DEFAULT_REGION_DRAFT` is frozen there beside it.

**WP5.**
20. *The lab's snap step is now the brick, and a press with no drag commits a box.*
    `regionSnapStep_cells` is `max(dyadicFloor(brick), dyadicFloor(floor))` per the
    recommendation above, so the lab moved off "the region's own floor cell", and
    `snapRegionBox` always yields at least one step of thickness — where
    `regionBoxIsDrawn` used to decline a zero-area press, a click now lands one
    brick. Both are behaviour changes, both are the shared policy's, and the
    alternative was the lab keeping its own snapping.
21. *One rounding, in the solver's frame.* The lab snapped in canvas cells and then
    flipped, which lands off the leaf lattice whenever `ny` is not a multiple of the
    step. `labSolverCell(at, ny)` (new, in `lab-region-space.ts`) flips the two drag
    corners *first*; `regionFromDraw` runs once in the solver's frame; the rubber
    band is that record flipped back by `labRegionCanvasBox`. The band therefore
    **is** the record that commits.
22. *`EditorHostProvider` moved from `AdvanceLab()` into `AdvanceSlice`.* The host
    closes over this render's controller handle, regions and lattice, so it is built
    where those are — one component inward.
23. *`LabEditorHostInput.params` is optional.* The lab builds its host before the
    parameter table exists on some paths; `params` is spread in only when supplied,
    which matches `EditorHost.params` already being optional in §2.2.

**WP6.**
24. *`advance-lab/slice-actions.ts` is replaced by a smaller `advance-lab/lab-ring.ts`,
    not deleted outright.* The lab still owns verbs no studio module can express —
    `pin-cell`, `surface-view`, `overlay`, `drop-ball`, `remove-region` — so what
    remains is the `LabHostVerb` union, its `host`-effect codec and the performer
    that answers them against an `EditorHost`. Every *wedge* in it now comes from
    `lib/features/*/ring.ts` or `lib/core/editor-entity-wedges.ts`.
25. *New core file `lib/core/editor-entity-wedges.ts`.* §3.3 names three feature
    rings; the select/delete pair belongs to no feature — it is what *any* entity
    offers — so it sits in core beside `editor-entity-catalog.ts`, which now composes
    from it. This is what makes the cross-host ring pin in
    `tests/editor-scene-ring.test.ts` a structural fact rather than a coincidence.
26. *`fluidPlayActions` gained a leading `scene` parameter.* The region wedge is
    `regionDrawWedge(studioRegionSpace, scene)`, which reads the document to disable
    itself at capacity — the studio's old inline wedge never did, so it now dims at
    eight regions the way the lab's always has.
27. *Lab wedge ids are the shared ones.* `slice-water` → `water`, `slice-drop-ball`
    → `ball`, `slice-draw-region` → `region`, `slice-inspect-cell` → `inspect-cell`,
    `slice-region-select` → `select`, `slice-region-remove` → `delete` (and "Remove"
    → "Delete"). `slice-visuals` / `slice-surface-*` / `slice-overlay-*` are still the
    lab's own and keep their names. The removed testIds are `slice-region-row`,
    `slice-region-draft*` and `slice-region-min|max|remove`; both hosts now render
    `scene-region-row`, `region-draft*`, `entity-option-*` and `entity-delete`.
28. *`advance-lab/slice-regions.ts` and `advance-lab/slice-actions.ts` took their
    tests with them.* `slice-regions.test.ts` and `slice-actions.test.ts` are
    deleted and replaced at the level the code now lives at:
    `advance-lab/lab-region-space.test.ts` (the space and the flip),
    `advance-lab/lab-ring.test.ts` (the ring and the performer),
    `lib/features/refinement-region/verification/rows.test.tsx` (both hosts, one row).
29. *One `react-hooks/refs` disable in `AdvanceLab.tsx`.* The host's members close
    over `controller.current`; the compiler-backed rule sees a ref reaching a call
    made during render. Nothing invokes them there — they are stored and called from
    pointer handlers, ring wedges and row `onChange`. The disable carries that
    sentence.

**WP7 (`advanceSliceFeature` + `LabFeatureSlot`).**
30. *The row components live in `advance-lab/lab-instruments.tsx`, not inside
    `LabFeatureSlot.tsx`.* §2.9 sketched one file; the slot file is the twelve lines
    the doc asked for (a composition, a binding table, a component) and the five
    views are beside it, so the binding table reads as a table. `LAB_FEATURE_VIEWS`
    and `LAB_FEATURE_COMPOSITION` are exported for the test that asks what was
    placed where.
31. *The lens and overlay controls declare no `options`.* Their roster is the
    `FieldVisualization` declarations (`ADVANCE_LENS_VIEWS`, `SLICE_OVERLAY_VIEWS`),
    which is what WP8 makes shared; restating sixteen lenses as `FeatureControl`
    options would have been the second roster the exercise exists to remove — and it
    would have made `definition.ts` import the lab's drawing code, which the module
    boundary forbids. `surface`, `budget` and `transport` do declare theirs: those
    rosters are already beside the method.
32. *`sim.transport` is rendered by the lab **header**, not the toolstrip.* PART 3's
    WP7 invariant lists the strip as "lens, overlays, surface, solve, region, water"
    and §2.9 places transport in `sim.transport`; both are honoured by mounting that
    one slot where the `<select>` already stood. The control's own `update: "reset"`
    is the reason — choosing an arm starts a new run, so it is not an instrument on
    the water in front of you.
33. *`advanceOverlays` is a comma-separated list.* `EditorHost.params` values are
    scalars (`number | string | boolean`), and the annotations are a *set*: the row
    flips one membership and hands the whole list back. Whether each can be drawn is
    derived rather than carried — "the surface is imposed" is exactly "the chosen
    `ADVANCE_SURFACE_VIEWS` entry declares itself unselectable" — so no availability
    flag travels beside the value.
34. *The lab's transport row is a label **beside** its `<select>`, not around it.*
    The two header classes moved to the wrapper `AdvanceLab.tsx` mounts the slot in.
    `.iters select` is a descendant rule, so the control is styled exactly as before;
    what this buys is that no `lib`-side or slot-side file imports one route's CSS
    module, which is what makes `advance-lab/lab-feature-slot.test.tsx` runnable at
    all (the unit runner has no CSS loader).
35. *Lab testIds moved to the shared ones.* `slice-lens-row` / `slice-overlay-row`
    are gone; the lab now renders `field-quick-*` and `field-overlay-*` (the studio's
    marks) and `scene-water-row`. The lab's lens row lost its `tag="LENS"` and its
    loop-step readout, because the shared row has neither.

**WP8 (one `FieldViewRows`).**
36. *`FieldViewState` gained `dismissable`, `adjustable` and `defaultAxis`, and
    `FieldViewRows` gained `catalog`.* The first two are the only genuine difference
    between the hosts, declared rather than branched on: a 3-D overlay is drawn
    *over* the water so clicking the lit mark puts it away and a plane and a scrub
    sit beside it, while a 2-D lens **is** the picture. `defaultAxis` is the studio's
    `DEFAULT_GRID_OVERLAY_AXIS` passed in rather than imported, and `catalog` is the
    unnarrowed registry the chevron's card lists (the studio narrows `views` to the
    running method's `supportedFieldModes`).
37. *`FieldOverlayRows` is a second export not named in §2.8.* The overlay toggles
    are switches over `FieldVisualization`s, not a choice among them, so they are a
    second component over the same declarations rather than a mode of the first.
38. *`VisualizationCommon.swatch` widened to `` `#${string}` | `var(--${string})` ``.*
    The lab resolves its band tones against the page theme, so its declarations carry
    a custom property rather than a literal. Nothing reads a swatch other than as a
    CSS colour.
39. *`ICONS` and `glyphOf` moved out of `components/FieldQuickBar.tsx`* into
    `lib/features/field-view/ui.tsx` (`fieldViewGlyph`, exported). `FieldQuickBar.tsx`
    is 261 → 76 lines and is now only the studio's two arguments.
40. *`lib/features/liquid-drop/ui.tsx` `LiquidDropRow` takes all-optional props.*
    The shape roster is the one thing neither host can fake for the other — the studio
    resolves an installed voxel tool against a scene, the Rust 2-D world has one shape
    — so the roster is injected and a host that passes none gets the ball. The chevron
    appears only for a roster of more than one.

**FIX (a press with no drag commits nothing).**
41. *`regionFromDraw` returns `RefinementRegionRecord | undefined`, and
    `refinementRegionFromDrag` returns `FluidRefinementRegion | undefined`.* The new
    `regionDrawIsDegenerate(anchor_cells, at_cells)` sits beside the minimum thickness
    that made the bug: `snapRegionBox` guarantees a step on every axis, which is right
    for a box somebody drew and wrong for a box nobody drew. Every axis, not any — a
    drag along one axis is a wide thin region, which is a real instruction. The studio
    asks the question on the raw drag corners *before* it seeds the box's height,
    because the seed would hide it.
42. *The studio press no longer previews.* `beginRegionDraw` opened the draft and
    immediately wrote a zero-footprint box into it; it now only opens the draft, and
    the `region-draw` pointer record carries `drawn`, set by the first pointer sample
    that names an area. A release without it cancels the draft and **stays armed** —
    the reader is about to make the drag they fumbled — and makes no selection, since
    a click while a stroke is armed belongs to the stroke on both hosts.

---

# WP9 — plugin-defined URL state on both hosts

Peter, 2026-09-17: *"2d should store state (regions, etc) in the query param just the
same as 3d. again...defined via plugin"*.

The lab writes two keys by hand (`AdvanceLab.tsx:519-533` `publishRunSelection`, read at
`:485-489`, `:511-514`) and mirrors nothing else. The studio mirrors everything through
one loop (`url-state.ts:837` `startQueryStateSync`, mounted only from
`components/FluidLab.tsx:37`, gated on `/scene`). The loop is host-agnostic; only the
parse and serialize it calls are the studio's. So: extract the loop, and make every key
a codec its own feature declares.

## 9.1 The loop, extracted verbatim

```ts
// lib/core/query-state-sync.ts   (zone: core; no scene, no preset, no method registry)
export interface HostQueryStateSync {
  /** The route this address describes. `/x` and `/x/…` are both this host. */
  readonly path: string;
  /** The canonical query for the current store snapshot. */
  serialize(search: string): string;
  /** Put a whole query onto the stores. Runs inside the write suppression. */
  hydrate(search: string): void;
  /** Every store the address must follow; each returns its unsubscribe. */
  readonly sources: readonly ((onChange: () => void) => () => void)[];
  /** False when a client navigation already retained the stores. */
  readonly hydrateFromUrl?: boolean;
}
export function startHostQueryStateSync(sync: HostQueryStateSync): () => void;
/** The one `history.replaceState`, shared so both hosts write the same way. */
export function replaceLocationSearch(search: string): void;
```

The body is today's `url-state.ts:837-914` with four studio nouns removed: the
`active`/`queued`/`applyingUrl` flags, the `queueMicrotask` coalescer (the memory rule
that the mirror is a microtask behind stays true of both hosts), the path gate, the
`popstate` re-hydrate and the teardown. `startQueryStateSync` becomes a ~22-line binding
keeping its exported signature: `path: "/scene"`, its existing `hydrate` body moved in
unchanged, and its four subscriptions as `sources` — the `runtimeFeaturesChanged` and
shell `view`/`compare` guards staying in the closures they already live in. **Studio
behaviour is byte-identical**: nothing in `parseQueryState`/`serializeQueryState` moves,
and `replaceQueryStateUrl` (`:806`, used by `SceneOverridesChip.tsx:56`) keeps its
signature by delegating to `replaceLocationSearch`.

## 9.2 Regions as a package-owned codec — percent of the **lattice**

```ts
// lib/features/refinement-region/persistence.ts   (zone: feature)
export const REGIONS_QUERY_KEY = "regions";
/** Records as a query value: 2·axes percentages, then the dyadic bounds. */
export function regionsToQuery<Doc>(
  space: Pick<RegionSpace<Doc, unknown>, "axes" | "lattice" | "list">, doc: Doc): string;
/** The records a value describes, resolved and re-snapped against THIS lattice. */
export function regionsFromQuery<Doc>(
  space: Pick<RegionSpace<Doc, unknown>, "axes" | "lattice" | "capacity" | "brick_cells">,
  doc: Doc, raw: string): RefinementRegionRecord[];
/** The document carrying exactly those regions. */
export function withRegionsFromQuery<Doc, Patch>(
  space: RegionSpace<Doc, Patch>, doc: Doc, raw: string): Doc;
```

**The format is the studio's, generalised over axes, and existing links parse
unchanged.** Wire form stays `<min…>_<max…>_<minCells>[_<maxCells>][_<rule>]*<next>`,
same `_`/`*` separators (`editor-refinement-region.ts:351-353`), same 4-dp
`percentText`, same 0..100 clamp, same per-record drop, same capacity break, same
re-snap on read. Only the field *count* is `2·axes + 1..3`: 7/8/9 in the studio exactly
as today, 5/6/7 in the lab. A codec is built with a known `space.axes`, so there is no
ambiguity within a host, and the two hosts never share an address.

Percent-of-container and percent-of-lattice are the **same number**, which is what makes
this free: `refinementRegionLattice.origin_m` is the container minimum
(`lib/core/refinement-regions.ts:153`) and `sceneCellSizes_m` is `extent / n`
(`lib/core/scene-lattice-dimensions.ts:33-40`), so `100·cells/n ≡ 100·(m − min)/span`
identically, the `maximumDimension` clamp included. Finest cells on the wire were the
alternative and are **rejected**: they would make a link lattice-specific — the same box
naming a different part of a re-authored domain — and would break every shipped
`?regions=` link.

`editor-refinement-region.ts:359` and `:437` become three-line metre adapters over the
package via `studioRegionSpace` (`:234`); `url-state.ts:261` stops declaring
`REGIONS_QUERY_KEY` and imports it.

## 9.3 The lab's keys, each owned by the feature that defines it

| key | owner module (already holds the values) | default |
|---|---|---|
| `scene` | `advance-lab/lab-scenes.ts` (new; the `SCENE_CATALOG` filter at `AdvanceLab.tsx:459-460`) | `water-box-dam-break` |
| `transport`, `solve` | `…/features/advance-slice/definition.ts:228,251,255,259` | arm; **arm-relative** budget |
| `regions` | `lib/features/refinement-region/persistence.ts` (§9.2) | the scene's authored regions |
| `freezeTopology` | `lib/features/topology-freeze/definition.ts:4` — **already**, free the moment the lab renders `RegionRow` (`MakeRows.tsx:67`) | false |
| `gridMode` (lens) | `advance-lab/lenses.ts:1264` `ADVANCE_LENSES`, through WP8's `fieldViewQuery` | `conservative-transport` |
| `overlays` | `advance-lab/lenses.ts:799` `SLICE_OVERLAYS`, `_`-joined ids | empty |
| `surface` | `…/advance-slice/definition.ts:180` `ADVANCE_SURFACE_VIEWS` | `shared-rdf` |
| `view.zoom`, `view.x`, `view.y` | `advance-lab/view-transform.ts:54,66,79` | `1`, `0.5`, `0.5` |
| — not mirrored — | playback (`playing`, `step`), hover/pinned probe, `folds`, `openSeam`, `metric`, `stepMs` | |

Two need more than a `queryRecord`:

- **`solve` is arm-relative**, like the studio's camera: `defaultPressureBudget` is a
  property of the arm (`definition.ts:228-247`), so `transport` and `solve` are **one
  hand-written `QueryCodec<{transportExperiment; pressureBudget}>`** in that same file,
  writing `solve` only when it differs from the named arm's default. A `numberQuery`
  with a fixed initial cannot say this.
- **the 2-D view is stored as fractions of the slice, not fine cells.** `SliceView.panFine`
  is `[nx/2, ny/2]` at fit (`view-transform.ts:79-81`), so a fraction is `0.5` on any
  lattice and a scene change never strands the pan. **Yes, the studio camera is
  mirrored** — read at `url-state.ts:652-667` against the preset's camera, written at
  `:750-755` only when it differs — and this is that rule with the fit view in the
  preset's place.

**Mirrored state must leave React.** The loop subscribes to stores, so the nine
`useState`s at `AdvanceLab.tsx:666-681,736-737` backing these keys move into one new
`advance-lab/lab-store.ts` `createLabStore()`, built beside the lab's
`createPaneSession("a")` (`:642`) and read by `labEditorHost.params`
(`lab-host.ts:60`), whose `commit` already writes the region list. `ui-store.ts` is
**not** touched: §2.7 put `regionDraft` there because both hosts have it, and none of
these nine is a studio concept. The loop's `sources` are then exactly three —
`session.ui`, `session.runtime`, the lab store — and `publishRunSelection`,
`requestedSceneId`, `requestedTransportExperiment` (`AdvanceLab.tsx:485-533`) delete.

## 9.4 The allow-list, derived

`isManagedKey` (`url-state.ts:695-700`) is a literal ORed with two codec key sets and
three prefixes. Make the shape declarable once:

```ts
// lib/framework/persistence.ts
export interface QueryKeyOwnership {
  readonly keys?: readonly string[];
  readonly prefixes?: readonly string[];
  readonly matches?: readonly ((key: string) => boolean)[];
}
export function managedQueryKey(...owners: readonly QueryKeyOwnership[]): (key: string) => boolean;
```

The studio composes `uiFeatureQuery.keys`, `runtimeFeatureQuery.keys`, the five
scene-side key constants, one exported `STUDIO_QUERY_KEYS`, one exported
`RETIRED_QUERY_KEYS` (`panel`, `panelWidth`, `sceneConfig` — the tolerance `:688-694`
documents), the three prefixes and `isCompareQueryKey`. **Which keys the studio accepts
does not change**: the literal is split by owner, not shortened, and §9.5(d) pins that.
The lab composes `labQuery.keys` and `runtimeFeatureQuery.keys` and nothing else — no
prefixes, so `?scene.container.width_m=` pasted onto `/advance-lab` is left alone.

## 9.5 Tests

- **(a)** `lib/features/refinement-region/verification/persistence.test.ts` — per space,
  `regionsFromQuery ∘ regionsToQuery` is identity on records, and a re-serialise after a
  hydrate equals the string hydrated from (idempotence, the invariant a mirroring loop
  actually rests on).
- **(b) one record, one string.** A 2-D record and the 3-D record that is it with `z`
  spanning the depth must produce **the same x and y percentage fields** under
  `labRegionSpace` and `studioRegionSpace` for matching lattices — the claim "both hosts
  encode the same record" reduces to this, and it fails loudly if either adapter's axis
  order or y-flip (`lab-region-space.ts:90-124`) drifts.
- **(c) the shipped link still parses.** `"0_0_0_100_100_100_2"` — the literal already in
  `tests/compare-model.test.ts:548` — read through the package gives the same cell
  bounds, floor and rule it gives at HEAD. Pin the numbers, not the source.
- **(d)** `tests/managed-query-keys.test.ts` — the derived predicate's verdict on every
  key a maximal `serializeQueryState` emits, plus the three retired names, plus a
  foreign key, equals the literal's. Enumerated from a fabricated state, never restated.
- **(e)** `lib/core/verification/query-state-sync.test.ts` — on a fake `window`: one
  hydrate before any write; N source changes in a tick produce **one** write; a write
  while `pathname` is another route is suppressed; `popstate` re-hydrates; `stop()`
  unsubscribes every source and the listener.
- **(f)** `advance-lab/lab-url-state.test.ts` — a link carrying every key in §9.3
  hydrates then re-serialises to itself; an unknown `?scene=` falls back to
  `water-box-dam-break` exactly as `:485-489` does; the lab writes nothing on `/scene`.

## 9.6 Ordering, gates, browser-only

**After WP8**, one agent, last in the chain: three lab keys (`gridMode`, `overlays`,
`surface`) are the field-view declarations WP8 creates, and declaring them twice to land
early is the duplication this document exists to remove. It **must not start while
WP4-WP6 are open** — §9.3 rewrites the same `AdvanceLab.tsx` state block they are
rewiring.

**G** all three, and the boundary count stays at 5:
`lib/features/refinement-region/persistence.ts` is `feature → core/framework`
(`check-module-boundaries.ts:74`), `lib/core/query-state-sync.ts` adds no core edge, and
the lab reaches `core`/`feature`/`framework`, all in the `advance-lab` set at `:110-122`.

**Browser-only.** (1) That a lab link reopens the same water — scene, arm, boxes,
framing. (2) That panning does not thrash the address bar: the coalescer is the
studio's, but the lab's pan is a pointer-rate source the studio has no equivalent of.
(3) That Back still leaves the lab rather than stepping through pans. (4) That a
`/advance-lab` link in a fresh tab does not flash the fit view before the hydrated one.

## 9.7 The riskiest decision

**Giving the lab's lens the studio's `gridMode` key.** It is what makes the field view
one plugin declaration instead of two, but `parseGridOverlayMode` (`url-state.ts:404-411`)
is a union of five *open-ended* predicates, not a list, so a shared codec deriving
acceptance from a registered view list would silently narrow which modes a studio link
can name — one per solver stage, none written down there. **Recommend the shared codec
take the host's own `accepts(raw): boolean`**: the studio passes its five-predicate
union unchanged, the lab passes `id in ADVANCE_LENSES`. One declaration point, zero
change to what the studio parses. The fallback, if that still reads as too much
coupling, is a lab-private `lens` key; all that is lost is the two hosts agreeing on
what "which field is drawn" is called.

The other load-bearing claim is §9.2's — that percent-of-container and percent-of-lattice
are the same number. It is the first thing to re-verify if `refinementRegionLattice`
ever stops taking its origin from the container.
