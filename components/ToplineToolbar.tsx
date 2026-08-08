"use client";

import { DEFAULT_EDITOR_TOOL, editorToolsPlacedAt } from "@/lib/editor-tools";
import { useUIStore } from "@/lib/stores/ui-store";

/**
 * The topline tool cluster, top-right of the viewport.
 *
 * Distinct from the left strip on purpose: those tools build the *set* — bodies,
 * props, water, a hose — while these annotate the *solve*. Which side of the
 * viewport a tool appears on is declared with the tool itself
 * (`EditorToolSpec.placement`), so this component does not know that regions
 * exist and neither does the strip.
 *
 * Clicking the armed tool disarms it, back to SELECT, exactly as the strip
 * does; the tool's own shortcut toggles it too.
 */
export function ToplineToolbar() {
  const activeTool = useUIStore((state) => state.activeTool);
  const setActiveTool = useUIStore((state) => state.setActiveTool);
  const tools = editorToolsPlacedAt("topline");
  if (tools.length === 0) return null;

  return (
    <nav className="topline-toolbar" aria-label="Solve annotation tools">
      {tools.map((tool) => (
        <button
          key={tool.id}
          type="button"
          className={activeTool === tool.id ? "active" : ""}
          aria-pressed={activeTool === tool.id}
          data-testid={`editor-tool-${tool.id}`}
          title={`${tool.label} (${tool.shortcut.toUpperCase()}) · ${tool.hint}`}
          onClick={() => setActiveTool(activeTool === tool.id ? DEFAULT_EDITOR_TOOL : tool.id)}
        >
          <i className="topline-tool-icon" aria-hidden="true" />
          <strong>{tool.label}</strong>
          <small>{tool.shortcut.toUpperCase()}</small>
        </button>
      ))}
    </nav>
  );
}
