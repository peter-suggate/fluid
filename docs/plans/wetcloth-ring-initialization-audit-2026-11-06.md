# Ring initialization and separate comparison inputs

The released `ring_test.xml` has effective vertex solid fraction 1 and zero initial pore capacity. This conclusion follows the pinned source initialization path, not just the XML parser. The original executable was not run for this audit. The released asset remains unchanged.

At upstream commit `b43993d775a0f9e3a9e32611f26936f2528fac37`:

1. [loadParticleSimulation](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneXMLParser.cpp#L690) loads liquid settings, particles, distance fields and materials before cloth topology and Gauss initialization.
2. `loadParticles` defaults missing `vf` to 1 and `setVolumeFraction` copies that value to both current and rest fractions. The uniform ring omits all vertex `vf` attributes; the nonuniform ring supplies all 40,000.
3. [loadClothes](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneXMLParser.cpp#L1472) assigns the material fraction only when `init_nonuniform_fraction` is false. Both ring inputs explicitly enable that flag, so neither is overwritten.
4. `initGaussSystem` angle-averages the existing fractions into face quadrature values. `updateRestPos` copies positions only. `updatePlasticity(0)` changes deformation matrices rather than packing fractions. `mapParticleSaturationPsiNodes` deposits the existing intrinsic fiber volumes; it does not change vertex fractions. `updateOptiVolume` only relabels free liquid particles.
5. [updateSolidVolumeFraction](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/TwoDScene.cpp#L5885) is guarded by `use_varying_fraction`. That option defaults false and is absent from both inputs. Both also use `solveSolid=0` and mark every cloth vertex fixed.

Direct integration of the original decimal XML coordinates and the source's area/3 vertex control volumes gives the following values. These use the corrected source thickness345µm (radius165µm plus independent default biradius180µm); earlier330µm calculations are superseded:

| Input | Vertex fraction range | Total control volume (m³) | Initial pore capacity (m³) |
|---|---:|---:|---:|
| Released uniform ring | 1–1 | 7.082559648e-6 | 0 |
| Released nonuniform ring | 0.0776470588–0.4847058824 | 7.082559648e-6 | 5.492421168e-6 |
| Exploratory uniform material packing | 0.4–0.4 | 7.082559648e-6 | 4.249535789e-6 |

The paper's ring discussion and supplemental parameter ranges do not establish a ring-specific uniform packing value. The exploratory fixture therefore chooses the released material's `restVolumeFraction=0.4` to resolve the XML flag/default inconsistency, and explicitly does not claim that value as a verified paper input. Its scene ID is `wetcloth-exploratory-ring-material-packing`. The source geometry asset and its checksum remain untouched.

A separate source nuance concerns dry mass: `loadClothes` uses material density times control volume times the material rest fraction, including when vertex fractions vary. Our conserved-fiber mass initialization uses vertex fraction. The two affected ring fixtures disable solid solving, but this distinction must be addressed before claiming a movable nonuniform-material source comparison.

Two paper-confirmed parameter variants are also available as unvalidated definitions: `wetcloth-paper-figure10b-25um` uses the 25 µm diameter printed in Figure 10b, and `wetcloth-paper-figure10j-90deg` restores the reference water surface tension while setting the exact 90° contact angle described in Figure 10j. Every variant records its original source ID, original source checksum, evidence links, and before/after numeric changes. The 36 released fixtures remain separate from these three comparison definitions.
