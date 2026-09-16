"use client";
import { Snowflake } from "lucide-react";
import { useSession } from "../../core/session/session-context";
import { ToolstripRow } from "../../../components/toolstrip";
import { topologyFreezeFeature } from "./definition";
export function TopologyFreezeButton() {
  const session = useSession();
  const frozen = session.runtime(state => state.topologyFrozen);
  const setFrozen = session.runtime(state => state.setTopologyFrozen);
  const control = topologyFreezeFeature.controls[0];
  return <div className="toolstrip-choice"><button type="button" className={frozen ? "active" : ""}
    style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
    aria-label={control.label} aria-pressed={frozen} data-testid="freeze-topology-toggle"
    title={`${control.hint}${frozen ? " Click to resume adaptivity." : ""}`}
    onClick={() => setFrozen(!frozen)}><Snowflake width={13} height={13} />{frozen ? "Frozen" : "Freeze"}</button></div>;
}

export function TopologyFreezeRow() {
  const control = topologyFreezeFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} icon={<Snowflake width={14} height={14} />}>
    <TopologyFreezeButton />
  </ToolstripRow>;
}
