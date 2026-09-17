"use client";

import { createContext, useContext, type ReactNode } from "react";
import type { EditorHost } from "../editor-host";
import type { SceneDescription } from "../model";

/**
 * The world a subtree of chrome commits into.
 *
 * A sibling of `SessionContext` rather than a member of `PaneSession`, because
 * a session is eight stores and a host is a set of *operations over* them: the
 * advance lab builds a real `PaneSession` and has no `SimulationController` at
 * all, so a session that carried its own host would have to carry a null one.
 *
 * Unlike the session there is **no default value**, and `useEditorHost` throws
 * without a provider. A silent fall-back to pane A is exactly the bug this
 * removes: a row rendered in the lab that quietly committed into the studio's
 * document would look like it worked.
 */
const EditorHostContext = createContext<EditorHost<never, never> | undefined>(undefined);

export function EditorHostProvider<Doc, Patch>({ value, children }: {
  value: EditorHost<Doc, Patch>;
  children: ReactNode;
}) {
  return <EditorHostContext.Provider value={value as unknown as EditorHost<never, never>}>
    {children}
  </EditorHostContext.Provider>;
}

/**
 * The host this component commits through.
 *
 * The type parameters default to the studio's document shapes, so every
 * existing studio call site reads `useEditorHost()` and gets what it had. A
 * capability module that is generic over its document passes its own.
 */
export function useEditorHost<
  Doc = SceneDescription,
  Patch = Partial<SceneDescription>,
>(): EditorHost<Doc, Patch> {
  const host = useContext(EditorHostContext);
  if (host === undefined) {
    throw new Error(
      "No EditorHostProvider is mounted. A capability module commits through its host; "
      + "mount <EditorHostProvider> beside the <SessionProvider> for this pane.");
  }
  return host as unknown as EditorHost<Doc, Patch>;
}
