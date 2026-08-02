import assert from "node:assert/strict";
import { findEntity } from "../../lib/editor-entity-catalog";
import type { BoxSides, EditorEntity, EditorEntityContext, EditorHandle } from "../../lib/editor-entity";
import type { EditorSelection } from "../../lib/editor-tools";
import type { SceneDescription } from "../../lib/model";

/**
 * Box-handle ids are sign triples — `+00` is the +x face, `+-+` a corner — so
 * a test can name the handle it means and get the sides that handle owns.
 */
export function sides(id: string): BoxSides {
  const axes = ["x", "y", "z"] as const;
  const result: Record<string, "min" | "max"> = {};
  id.split("").forEach((character, index) => {
    if (character !== "0") result[axes[index]!] = character === "-" ? "min" : "max";
  });
  return result as BoxSides;
}

export function context(scene: SceneDescription, bodies: EditorEntityContext["bodies"] = []): EditorEntityContext {
  return { scene, bodies };
}

export function entityFor(
  scene: SceneDescription,
  selection: EditorSelection,
  bodies: EditorEntityContext["bodies"] = [],
): EditorEntity {
  const entity = findEntity(context(scene, bodies), selection);
  assert.ok(entity, `no entity for ${selection.kind}:${selection.id}`);
  return entity;
}

export function handleById(
  scene: SceneDescription,
  selection: EditorSelection,
  handleId: string,
  bodies: EditorEntityContext["bodies"] = [],
): EditorHandle {
  const entity = entityFor(scene, selection, bodies);
  const handle = entity.handles.find((candidate) => candidate.id === handleId);
  assert.ok(handle, `no handle ${handleId} on ${selection.kind}`);
  return handle;
}
