"use client";
import { Snowflake } from "lucide-react";
import { useSession } from "../../core/session/session-context";
import { ToolstripRow } from "../../../components/toolstrip";
import { topologyFreezeFeature } from "./definition";
export function TopologyFreezeRow() {
  const session = useSession();
  const frozen = session.runtime(state => state.topologyFrozen);
  const setFrozen = session.runtime(state => state.setTopologyFrozen);
  const control = topologyFreezeFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} icon={<Snowflake width={14} height={14} />}>
    <div className="toolstrip-choice"><button type="button" className={frozen ? "active" : ""}
      aria-label={control.label} aria-pressed={frozen} data-testid="freeze-topology-toggle"
      onClick={() => setFrozen(!frozen)}>{frozen ? "Frozen" : "Freeze"}</button></div>
  </ToolstripRow>;
}
