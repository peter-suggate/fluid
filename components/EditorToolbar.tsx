"use client";

import { DEFAULT_EDITOR_TOOL, EDITOR_TOOLS } from "@/lib/editor-tools";
import type { RigidShape, ScenePropShape } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useEditorHistoryStore } from "@/lib/stores/history-store";
import { useUIStore } from "@/lib/stores/ui-store";

const PLACEMENT_SHAPES: ReadonlyArray<{ shape: RigidShape; label: string }> = [
  { shape: "sphere", label: "Sphere" },
  { shape: "box", label: "Box" },
  { shape: "capsule", label: "Capsule" },
  { shape: "cylinder", label: "Cylinder" },
];

const PROP_SHAPES: ReadonlyArray<{ shape: ScenePropShape; label: string }> = [
  { shape: "box", label: "Box" },
  { shape: "cylinder", label: "Post" },
  { shape: "ellipsoid", label: "Blob" },
];

/**
 * Left-edge tool strip, mirroring the right-edge utility tabs.
 *
 * Only tools that actually do something are listed. Unimplemented ones used to
 * be rendered disabled so "the chassis is visible", but a permanently dead
 * button is a row of noise in a strip a user has to scan to find the mode they
 * want — they stay declared in `EDITOR_TOOLS` with their plan phase, and appear
 * here the moment their status flips to active.
 *
 * Each tool's hint lives in its tooltip rather than in a paragraph pinned under
 * the strip: the armed tool announces itself in the viewport when it has
 * something to say.
 *
 * Clicking the armed tool disarms it, back to the default. A mode you can enter
 * but not leave by the same control you entered it with is a trap, and the
 * keyboard agrees: the tool's own shortcut toggles it too.
 */
export function EditorToolbar() {
  const activeTool = useUIStore((state) => state.activeTool);
  const setActiveTool = useUIStore((state) => state.setActiveTool);
  const placementShape = useUIStore((state) => state.placementShape);
  const setPlacementShape = useUIStore((state) => state.setPlacementShape);
  const propShape = useUIStore((state) => state.propShape);
  const setPropShape = useUIStore((state) => state.setPropShape);
  const canUndo = useEditorHistoryStore((state) => state.past.length > 0);
  const canRedo = useEditorHistoryStore((state) => state.future.length > 0);

  return (
    <div className="editor-toolbar" data-active-tool={activeTool}>
      <nav className="editor-tool-strip" aria-label="Editor tools">
        {EDITOR_TOOLS.filter((tool) => tool.status === "active").map((tool) => (
          <button
            key={tool.id}
            type="button"
            className={activeTool === tool.id ? "active" : ""}
            aria-pressed={activeTool === tool.id}
            data-testid={`editor-tool-${tool.id}`}
            title={`${tool.label} (${tool.shortcut.toUpperCase()}) · ${tool.hint}`}
            onClick={() => setActiveTool(activeTool === tool.id ? DEFAULT_EDITOR_TOOL : tool.id)}
          >
            <strong>{tool.label}</strong><small>{tool.shortcut.toUpperCase()}</small>
          </button>
        ))}
      </nav>
      <div className="editor-history" role="group" aria-label="Edit history">
        <button type="button" disabled={!canUndo} title="Undo (⌘Z)" onClick={() => simulation.undo()}>UNDO</button>
        <button type="button" disabled={!canRedo} title="Redo (⇧⌘Z)" onClick={() => simulation.redo()}>REDO</button>
      </div>
      {activeTool === "prop-place" && (
        <div className="editor-tool-options" role="group" aria-label="Prop shape">
          {PROP_SHAPES.map(({ shape, label }) => (
            <button
              key={shape}
              type="button"
              className={propShape === shape ? "active" : ""}
              aria-pressed={propShape === shape}
              title={`Place a ${label.toLowerCase()} prop`}
              onClick={() => setPropShape(shape)}
            >
              <i className={`body-shape-icon shape-${shape === "ellipsoid" ? "sphere" : shape}`} aria-hidden="true" /><span>{label}</span>
            </button>
          ))}
        </div>
      )}
      {activeTool === "body-place" && (
        <div className="editor-tool-options" role="group" aria-label="Placement shape">
          {PLACEMENT_SHAPES.map(({ shape, label }) => (
            <button
              key={shape}
              type="button"
              className={placementShape === shape ? "active" : ""}
              aria-pressed={placementShape === shape}
              title={`Place a ${label.toLowerCase()}`}
              onClick={() => setPlacementShape(shape)}
            >
              <i className={`body-shape-icon shape-${shape}`} aria-hidden="true" /><span>{label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
