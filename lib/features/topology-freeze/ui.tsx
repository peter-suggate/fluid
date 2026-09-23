"use client";
import { Snowflake } from "lucide-react";
import { useSession } from "../../core/session/session-context";
import { ToolstripRow } from "../../../components/toolstrip";
import { ToggleButton } from "../../../components/ui";
import { topologyFreezeFeature } from "./definition";
export function TopologyFreezeButton() {
  const session = useSession();
  const frozen = session.runtime(state => state.topologyFrozen);
  const setFrozen = session.runtime(state => state.setTopologyFrozen);
  const control = topologyFreezeFeature.controls[0];
  return <ToggleButton pressed={frozen} onChange={setFrozen} ariaLabel={control.label}
    hint={`${control.hint}${frozen ? " Click to resume adaptivity." : ""}`} testId="freeze-topology-toggle"
  ><Snowflake width={13} height={13} />{frozen ? "Frozen" : "Freeze"}</ToggleButton>;
}

export function TopologyFreezeRow() {
  const control = topologyFreezeFeature.controls[0];
  return <ToolstripRow name={control.label} hint={control.hint} icon={<Snowflake width={14} height={14} />}>
    <TopologyFreezeButton />
  </ToolstripRow>;
}
