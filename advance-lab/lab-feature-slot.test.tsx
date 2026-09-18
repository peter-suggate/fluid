import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EditorHostProvider } from "../lib/core/session/host-context";
import { createPaneSession } from "../lib/core/session/session";
import { SessionProvider } from "../lib/core/session/session-context";
import { composeFeatures } from "../lib/framework/composition";
import { ComposedFeatureSlot } from "../lib/framework/ui/slot";
import {
  ADVANCE_SLICE_SETTINGS, ADVANCE_SURFACE_VIEWS, ADVANCE_TRANSPORT_EXPERIMENT_ORDER,
  advanceSliceFeature,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import {
  LAB_FEATURE_COMPOSITION, LAB_FEATURE_VIEWS, LabFeatureSlot,
} from "./LabFeatureSlot";
import { labEditorHost } from "./lab-host";
import type { LabRegionDocument } from "./lab-region-space";
import { ADVANCE_LENS_VIEWS, REPRESENT_LENS_MODE } from "./lenses";

/**
 * The lab's strip is its declaration, not a hand-written column.
 *
 * Before WP7 every row on `SliceToolstrip.tsx` was written there: a sixteen-
 * entry lens menu, two overlay switches, a surface chooser, a budget spinner
 * and a transport `<select>`, each with its own tag, its own hint and its own
 * page state wired in. The rows a reader sees are now `advanceSliceFeature`'s
 * `FeaturePlacement`s rendered by `ComposedFeatureSlot` — the *same* component
 * `components/SceneToolstrip.tsx` renders the studio's declarations with — so
 * these pins are all of the form "what is on screen is what was declared":
 *
 *   1. a slot's rows are its placements, in the order and prominence declared;
 *   2. a slot is *exactly* its placements — drop one and exactly that row goes,
 *      with the rest byte-identical;
 *   3. every placement the feature declares has a view bound, so a control
 *      added beside the method fails here rather than at the reader's pointer
 *      (`ComposedFeatureSlot` throws on an unbound placement).
 */

/** What each declared control puts on screen, as the row's own test id. */
const MARK: Readonly<Record<string, string>> = {
  // The lens picker and the overlay switches are the *shared* field-view rows
  // (`lib/features/field-view/ui.tsx`), so their marks are the studio's marks.
  lens: 'data-testid="field-quick-',
  overlays: 'data-testid="field-overlay-',
  surface: 'data-testid="slice-surface-row"',
  budget: 'data-testid="slice-budget-row"',
  transport: 'data-testid="advance-transport"',
  "adaptive-sdf": 'data-testid="advance-adaptive-sdf"',
};

/** A page's worth of answers, so every row has something to draw. */
function labHost(session: ReturnType<typeof createPaneSession>) {
  const values: Record<string, number | string | boolean> = {
    [ADVANCE_SLICE_SETTINGS.lens]: REPRESENT_LENS_MODE,
    [ADVANCE_SLICE_SETTINGS.overlays]: "fraction",
    [ADVANCE_SLICE_SETTINGS.surface]: ADVANCE_SURFACE_VIEWS[0]!.id,
    [ADVANCE_SLICE_SETTINGS.budget]: 64,
    [ADVANCE_SLICE_SETTINGS.transport]: "level-set-volume",
    [ADVANCE_SLICE_SETTINGS.adaptiveSdf]: true,
  };
  return labEditorHost({
    session,
    commitRegions: () => {},
    dropAt: () => {},
    params: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, {
      value, set: (next: number | string | boolean) => { values[key] = next; },
    }])),
  });
}

function draw(node: ReactNode): string {
  const session = createPaneSession("a");
  return renderToStaticMarkup(
    <SessionProvider value={session}>
      <EditorHostProvider<LabRegionDocument, LabRegionDocument> value={labHost(session)}>
        {node}
      </EditorHostProvider>
    </SessionProvider>);
}

/** The placements of one slot, ranked as the slot renderer ranks them. */
function placed(slot: string): readonly string[] {
  return LAB_FEATURE_COMPOSITION.placements
    .filter((placement) => placement.slot === slot)
    .map((placement, index) => ({ placement, index }))
    .sort((a, b) => (a.placement.priority === "high" ? 0 : 1)
      - (b.placement.priority === "high" ? 0 : 1) || a.index - b.index)
    .map(({ placement }) => placement.control);
}

const SLOTS = [...new Set(LAB_FEATURE_COMPOSITION.placements.map((p) => p.slot))];

test("every slot's rows are its placements, in the declared order", () => {
  assert.ok(SLOTS.length >= 4, "the lab's instruments are spread over named slots");
  for (const slot of SLOTS) {
    const markup = draw(<LabFeatureSlot slot={slot} />);
    const controls = placed(slot);
    const at = controls.map((control) => {
      const mark = MARK[control];
      assert.ok(mark !== undefined, `no mark known for ${control}`);
      const index = markup.indexOf(mark!);
      assert.ok(index >= 0, `${slot} did not render ${control}`);
      return index;
    });
    for (let row = 1; row < at.length; row += 1) {
      assert.ok(at[row]! > at[row - 1]!,
        `${slot}: ${controls[row]} should follow ${controls[row - 1]}`);
    }
  }
});

test("a slot is exactly its placements: drop one and exactly that row goes", () => {
  const slot = "scene.visibility";
  const controls = placed(slot);
  assert.deepEqual([...controls], ["lens", "overlays"],
    "the lens is the prominent instrument, the annotations follow it");

  // One placement at a time, through the same renderer and the same bindings.
  const only = (control: string) => draw(<ComposedFeatureSlot
    composition={composeFeatures({
      features: [{
        ...advanceSliceFeature,
        placements: (advanceSliceFeature.placements ?? [])
          .filter((placement) => placement.control === control),
      }],
    })}
    views={LAB_FEATURE_VIEWS}
    slot={slot}
  />);

  // The whole slot is its rows concatenated: nothing is written between them,
  // so removing a placement can only remove its own row.
  assert.equal(draw(<LabFeatureSlot slot={slot} />), only("lens") + only("overlays"));
  assert.ok(!only("lens").includes(MARK.overlays!), "the lens row draws no annotations");
  // And what the row draws is the declaration, not a second list: the mark is
  // the lens the host says is up, keyed by the registry entry's own mode, and
  // its name is that entry's declared label. (The other fifteen are in the
  // chevron's card, which is closed until a reader opens it.)
  const lens = ADVANCE_LENS_VIEWS[0]!;
  assert.equal(lens.mode, REPRESENT_LENS_MODE);
  assert.ok(only("lens").includes(`field-quick-${lens.mode}`), "the lit mark is the lens up");
  assert.ok(only("lens").includes(lens.label), "and its name is the declared label");
});

test("every placement the feature declares has a view bound", () => {
  for (const placement of LAB_FEATURE_COMPOSITION.placements) {
    const key = `${placement.feature}/${placement.control}`;
    assert.ok(LAB_FEATURE_VIEWS[key] !== undefined, `unbound placement ${key}`);
  }
});
