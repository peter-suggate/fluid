import type { SceneDescription } from "../../core/model";
import type { PaneId } from "../../core/session/session";
import type { InteractionView } from "../../interaction";
import type { InteractionPane } from "../interactive/pane";

/** What a provider sees of one pane. Providers write through their own domain's commands, never through the host. */
export interface WidgetContext {
  readonly paneId: PaneId;
  readonly scene: SceneDescription;
  readonly interaction?: { readonly pane: InteractionPane; readonly view: InteractionView; readonly busy: boolean };
}
