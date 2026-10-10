import { installSimulationMethods } from "../core/method-registry";
import { uniformNarrowBandMethod } from "./uniform/uniform-narrow-band-method";

/** The composition root installs the solver; consumers use the plugin registry. */
installSimulationMethods({
  methods: [uniformNarrowBandMethod],
  defaultId: uniformNarrowBandMethod.id,
});
