# Released source motion and shared-radius audit

Pinned source: `raymondyfei/libwetcloth` commit `b43993d775a0f9e3a9e32611f26936f2528fac37`. This is an execution-order audit, not a reproduction receipt.

## Script and coupled solve order

[`WetClothCore::stepSystem`](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/WetClothCore.cpp#L89) calls `stepScript(sub_dt, cur_time)` and `applyScript(sub_dt)` before emission, grid construction, particle transfers, force prediction and pressure projection.

[`TwoDScene::applyScript`](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/TwoDScene.cpp#L8067) updates fixed soft vertices' rest-position targets. It does not teleport their material positions or prescribe their translational velocities. Prescribed velocity is assigned only to sampled rigid surfels. Fixed twist targets receive the projected group-rotation increment. Distance fields advance in the same script application.

[`LinearizedImplicitEuler::constructNodeForce`](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/Core/LinearizedImplicitEuler.cpp#L548) computes the soft elastic impulse from `-dt * accumulateGradU` before mixture pressure projection. Consequently, the new attachment targets influence the same substep's force predictor while quadrature still samples the old material geometry. Pressure/drag and implicit solid integration follow, then particle positions advance.

At audit time the production NB-FLIP host advances source distance fields before fluid work, but advances cloth attachment targets and solves mechanics only after fluid completion. Using old material positions for fluid quadrature is consistent with the source. Omitting the same-step elastic/attachment predictor from fluid coupling is a separate Lie-split discrepancy: the liquid initially sees the previous material velocity. Moving the positions to the target would not fix this and would violate finite attachment physics. Root and mechanics owners were notified; this audit does not authorize or certify a scheduler change.

## Mixed shell and yarn radii

[`loadClothes`](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneXMLParser.cpp#L1455) sets zero vertex radii from shell parameters. [`loadHairs`](https://github.com/raymondyfei/libwetcloth/blob/b43993d775a0f9e3a9e32611f26936f2528fac37/libWetCloth/App/TwoDSceneXMLParser.cpp#L1693) preserves nonzero scene radii and copies each endpoint's radius into rod parameters. General inputs can therefore need varying endpoint radii.

The two released water scenes combining shells and rods were checked by replaying this initialization on their XML topology:

| Source | Shared shell/rod root uses | Shell and rod radius | Explicit particle-radius overrides | Differing endpoint radii |
|---|---:|---:|---:|---:|
| `fuzzy_cloth.xml` | 39,200 | 180 µm | 0 | 0 |
| `tighten_towel.xml` | 22,272 | 165 / 180 µm | 0 | 0 |

Both use material index zero for the shell and every rod. Every shared endpoint is the rod's start, never its final tip. The shared endpoints do not introduce taper because their shell and rod parameters agree. However, the material parser independently defaults biradius to180µm, so towel rods are elliptical and the shell thickness is345µm. The earlier circular-radius conclusion was incorrect; shared elliptical geometry is required.
