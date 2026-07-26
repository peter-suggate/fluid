"use client";

import { EDITOR_TOOLS, getEditorTool } from "@/lib/editor-tools";
import type { RigidShape } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useEditorHistoryStore } from "@/lib/stores/history-store";
import { useUIStore } from "@/lib/stores/ui-store";

const PLACEMENT_SHAPES: ReadonlyArray<{ shape: RigidShape; label: string }> = [
  { shape: "sphere", label: "Sphere" },
  { shape: "box", label: "Box" },
  { shape: "capsule", label: "Capsule" },
  { shape: "cylinder", label: "Cylinder" },
];

/**
 * Left-edge tool strip, mirroring the right-edge utility tabs.
 *
 * Tools whose pointer behaviour has not landed are rendered disabled with the
 * plan phase in their tooltip rather than hidden: the chassis is visible, and
 * a click cannot silently do nothing.
 */
export function EditorToolbar() {
  const activeTool = useUIStore((state) => state.activeTool);
  const setActiveTool = useUIStore((state) => state.setActiveTool);
  const placementShape = useUIStore((state) => state.placementShape);
  const setPlacementShape = useUIStore((state) => state.setPlacementShape);
  const canUndo = useEditorHistoryStore((state) => state.past.length > 0);
  const canRedo = useEditorHistoryStore((state) => state.future.length > 0);
  const spec = getEditorTool(activeTool);

  return (
    <div className="editor-toolbar" data-active-tool={activeTool}>
      <nav className="editor-tool-strip" aria-label="Editor tools">
        {EDITOR_TOOLS.map((tool) => (
          <button
            key={tool.id}
            type="button"
            className={activeTool === tool.id ? "active" : ""}
            disabled={tool.status !== "active"}
            aria-pressed={activeTool === tool.id}
            data-testid={`editor-tool-${tool.id}`}
            title={tool.status === "active"
              ? `${tool.label} (${tool.shortcut.toUpperCase()}) · ${tool.hint}`
              : `${tool.label} — not yet implemented · ${tool.phase}`}
            onClick={() => setActiveTool(tool.id)}
          >
            <strong>{tool.label}</strong><small>{tool.shortcut.toUpperCase()}</small>
          </button>
        ))}
      </nav>
      <div className="editor-history" role="group" aria-label="Edit history">
        <button type="button" disabled={!canUndo} title="Undo (⌘Z)" onClick={() => simulation.undo()}>UNDO</button>
        <button type="button" disabled={!canRedo} title="Redo (⇧⌘Z)" onClick={() => simulation.redo()}>REDO</button>
      </div>
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
      <p className="editor-tool-hint" data-testid="editor-tool-hint">{spec.hint}</p>
    </div>
  );
}
