import { installSimulationMethods } from "../core/method-registry";
import { pourMethod } from "./pour-method";
import { seepMethod } from "./seep-method";
import { uniformNarrowBandMethod } from "./uniform/uniform-narrow-band-method";

/** The composition root installs the solver; consumers use the plugin registry.
 * Seep · M1 Max and Pour · M1 Max are experimental and open only by URL (?method=seep-m1max, ?method=pour-m1max). */
installSimulationMethods({
  methods: [uniformNarrowBandMethod, seepMethod, pourMethod],
  defaultId: uniformNarrowBandMethod.id,
});
