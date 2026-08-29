"use client";

import { createContext, useContext, type ReactNode } from "react";
import { defaultSession, type PaneSession } from "./session";

/**
 * The realm a subtree of chrome is authoring and reading.
 *
 * The default value is pane A's session, so a tree with no provider mounted
 * behaves exactly as the app did before sessions existed — there is no
 * single-pane special case to keep in step.
 */
const SessionContext = createContext<PaneSession>(defaultSession);

export function SessionProvider({ value, children }: {
  value: PaneSession;
  children: ReactNode;
}) {
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

/** The pane this component belongs to. */
export function useSession(): PaneSession {
  return useContext(SessionContext);
}
