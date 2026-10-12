import type { SceneDescription } from "../model";
import type { PaneSession } from "../session/session";
import { voxelTools } from "./registry";
import { sessionToolHost } from "./session-host";
import { publishSolidPatches } from "./transaction";
import { sceneLatticeRegion, voxelRegionVerbContext, voxelRegionVerbTarget, voxelVerbVariant, voxelVerbView,
  type VoxelRegionVerbPlugin } from "./verb";

/**
 * Run a selection verb on a pane, and commit what it returns.
 *
 * The verb only says what should change. Everything a stroke's release does
 * happens here, through the same host and the same publish: the proposed
 * document goes to `validate` (the renderer's live solid-edit preflight) before
 * it is written, one history entry is recorded under the verb's caption, and
 * `voxelStrokePending` keeps undo, import and a second edit out meanwhile. So a
 * verb can be refused for capacity exactly as a stroke can, and is never the
 * re-seeding scene replacement it used to be.
 *
 * `variantId` names one of the verb's declared ways of running — the axis a
 * wedge was for, the direction a key meant. The pane's camera is read here and
 * handed on as lattice axes, so a verb that moves things "left" never sees a
 * camera. A verb marked `standalone` runs with nothing selected, pointed at
 * the whole lattice.
 *
 * Rejects with the reason, as a sentence, when the verb cannot run or the edit
 * is refused; the caller owns how that is said.
 */
export async function runVoxelRegionVerb(verb: VoxelRegionVerbPlugin, session: PaneSession,
  validate: ((scene: SceneDescription, base: SceneDescription) => Promise<void>) | undefined,
  variantId?: string): Promise<void> {
  const ui = session.ui.getState();
  const variant = voxelVerbVariant(verb, variantId);
  const selected = ui.selection?.kind === "voxel-region" ? ui.voxelRegion : undefined;
  if (!selected && !verb.ui.standalone) throw new Error("Select some voxels first.");
  if (ui.voxelStrokePending) throw new Error("Finish the voxel stroke first.");
  const base = session.scene.getState().scene;
  const target = voxelRegionVerbTarget({ scene: base, methodId: session.method.getState().methodId,
    region: selected ?? sceneLatticeRegion(base), clipboard: ui.voxelClipboard });
  const unavailable = verb.unavailable(target);
  if (unavailable) throw new Error(unavailable);
  const result = verb.run(voxelRegionVerbContext(verb, target, ui.voxelToolValues,
    { variant, view: voxelVerbView(ui.camera.azimuth_rad) }));
  // Named before anything is written: a verb that arms a tool nobody registered must not half-apply.
  if (result.armTool !== undefined && !voxelTools.get(result.armTool)) {
    throw new Error(`Voxel verb ${verb.id} arms an unregistered tool: ${result.armTool}`);
  }
  if (result.patches.length > 0) {
    if (!validate) throw new Error("Wait for the scene to be ready.");
    const host = sessionToolHost(session, validate);
    host.begin(result.caption);
    // Superseded by another document: nothing was written, so nothing else the verb asked for applies.
    if (!await publishSolidPatches(host, base, result.patches)) return;
  }
  const after = session.ui.getState();
  if (result.clipboard) after.setVoxelClipboard(result.clipboard);
  // The selection is the verb's to move only while it is still the one it ran on.
  const standing = after.selection?.kind === "voxel-region" ? after.voxelRegion : undefined;
  if (result.region !== undefined && standing === selected) after.selectVoxelRegion(result.region ?? undefined);
  // Last: the tool is armed over the document and the selection the verb left.
  if (result.armTool !== undefined) after.setVoxelTool(result.armTool);
  if (result.caption) session.runtime.getState().setNotice(result.caption);
}
