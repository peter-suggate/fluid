import type { AdvanceStageId } from "./advance-stage-contract";
import type { AdvanceSliceDeclaration } from "./definition";
/** Captions and stage descriptions for the adaptive 2-D lab. */
export const ADVANCE_STAGE_BANDS = {
    transport: "Transport velocity + geometric volume transport",
    momentum: "Momentum prediction",
    pressure: "Composite pressure projection + receipts",
    adaptivity: "Activity census + candidate topology",
    output: "Sparse presentation publication",
} as const;
interface AdvanceStageDeclaration {
    readonly label: string;
    readonly band: keyof typeof ADVANCE_STAGE_BANDS;
    readonly slice: AdvanceSliceDeclaration;
    readonly tip: {
        readonly summary: string;
        readonly reads?: string;
        readonly writes?: string;
        readonly feeds?: string;
    };
    readonly substages?: Readonly<Record<string, {
        readonly label: string;
    }>>;
}
export const ADVANCE_STAGE_DECLARATIONS = Object.freeze({
    "transport-velocity-extension": {
        label: "Velocity extension",
        band: "transport",
        substages: {
            "frame-control-authority": { label: "Frame-control authority + moving-solid activity" },
            "velocity-extension-mask-initialization": { label: "VEX2 cached schedule + packet-mask initialization" },
            "velocity-extension-sweeps": { label: "VEX2 eight scheduled packet sweeps + fused commit" },
            "transport-packet-authority": { label: "AEI packet-authority construction" }
        },
        slice: {
            caption: "Eight packet sweeps push face velocity out of the liquid into the empty band, so transport has a defined velocity everywhere it might sweep. A ghost row is one no liquid cell touches.",
            keys: [
                { id: "extended-ghost-row", tone: "transport", label: "extended ghost row",
                    note: "the sweeps wrote velocity here: no liquid cell touches this row" },
                { id: "carried-velocity", tone: "amber", label: "carried velocity",
                    note: "liquid here carries its own face velocity into the sweep" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Prepares frame controls, motion bounds and activity bookkeeping. Optionally stages extra sparse support needed by moving liquid. Caches accepted packet addresses by topology generation, initializes liquid velocities and clears air velocities, then optionally runs eight sweeps to extend velocity into air. Support lookahead and extension sweeps can be disabled independently.",
            reads: "accepted velocity, liquid level set, accepted topology and body/boundary inputs",
            writes: "sealed frame control, staged sparse support and transport velocity cache",
            feeds: "face preparation and geometric volume transport",
        }
    },
    "face-preparation": {
        label: "Face preparation",
        band: "transport",
        substages: {
            "face-support-publication": { label: "Resident face-velocity support clear + publication" },
            "accepted-face-row-preparation": { label: "Compact accepted face preparation" }
        },
        slice: {
            caption: "Every row is re-cut against the solids. The stored face velocity already folds in the aperture as u = a·u_fluid + (1−a)·u_wall — flux code must never multiply by a twice.",
            keys: [
                { id: "closed-row", tone: "solidEdge", label: "closed row · a = 0",
                    note: "a row of this cell is shut against the solid, so no flux crosses it" },
                { id: "partly-open-row", tone: "momentum", label: "partly open row",
                    note: "a row of this cell is part solid; the stored u already folds that aperture in" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Traces accepted faces with RK2 through extended velocity, then samples the source staggered face field on the finest incident lattice. Physical subface overlap selects mixed-resolution samples. Dry support, moving cut faces and uncertified exterior patches use the extended velocity field. Explicit coarse regions can enlarge trajectory sampling without coarsening the advected face field.",
            reads: "source face velocity, extended trajectory velocity, accepted cells and composite row topology",
            writes: "oriented face transport rows",
            feeds: "geometric volume transport",
        }
    },
    "conservative-transport": {
        label: "Adaptive level set and volume transport",
        band: "transport",
        substages: {
            "transport-coupling": { label: "Phi advection and geometric transport coupling" },
            "liquid-capacity-balancing": { label: "Liquid receiver capacity balancing" },
            "transport-gather": { label: "Conservative volume gather and commit" }
        },
        slice: {
            caption: "Volume moves as swept prisms cut from the PLIC polygon and handed across one shared subface. Each arrow is a paired debit and credit; a marked row is one the bounded limiter had to cut back.",
            keys: [
                { id: "swept-flux", tone: "transport", label: "swept flux",
                    note: "volume crossed a row of this cell as a swept prism this advance" },
                { id: "limiter-clipped", tone: "alarm", label: "limiter clipped",
                    note: "the bounded limiter had to cut a row of this cell back to keep V inside K" },
            ],
            loopStep: 3,
        },
        tip: {
            summary: "Advects adaptive phi, builds a sparse donor/receiver coupling, and balances overfull liquid receivers while preserving donor totals. Gathers extensive liquid volume once; any excess remaining after the bounded balancing stays explicit for the pressure source.",
            reads: "adaptive phi, extensive liquid volume, cell capacity, projected face velocity and compiled topology",
            writes: "advected adaptive phi, conservative liquid volume and transport receipts",
            feeds: "pressure, adaptivity and presentation publication",
        }
    },
    "tracer-advection": {
        label: "Marker advection",
        band: "transport",
        slice: {
            caption: "Markers ride the same published transport velocity the volume does. They carry no mass — they exist so a colour or an age can be read back out of the flow.",
            keys: [
                { id: "marker", tone: "adaptivity", label: "marker",
                    note: "a marker rides this cell; it carries no mass, only what was written on it" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "Presentation-only markers integrated through the extended velocity that supplies the geometric face fluxes. Encoded only while the marker view is on, so this reads zero on an ordinary frame.",
            reads: "extended transport velocity, accepted density",
            writes: "marker positions and their live flags",
            feeds: "the marker overlay, and nothing in the physics",
        }
    },
    "scalar-publication": {
        label: "Scalar publication",
        band: "transport",
        substages: {
            "final-scalar-mask-publication": { label: "FSM1 final-scalar packet-mask publication" }
        },
        slice: {
            caption: "What this advance actually changed. Only these cells enter the dirty worklists the adaptivity band walks — everything unlit is carried forward untouched.",
            keys: [
                { id: "volume-changed", tone: "output", label: "volume changed",
                    note: "this advance moved volume here, so the cell enters the dirty worklists" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "Publishes final-scalar packet masks and the completed scalar output without modifying density or gamma. This publication runs even when every correction is disabled.",
            reads: "completed scalar stage coverage",
            writes: "final-scalar packet masks and frame scalar output receipt",
            feeds: "body-force prediction",
        }
    },
    "body-forces": {
        label: "Body forces",
        band: "momentum",
        slice: {
            caption: "Gravity lands on the rows, not the cells — one add per row that touches liquid. Nothing else in the advance writes velocity without being projected afterwards.",
            keys: [
                { id: "row-taking-gravity", tone: "momentum", label: "row taking g·dt",
                    note: "gravity lands on this row once, and the projection answers for it" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Applies gravity and scene acceleration on the accepted face rows before projection.",
            reads: "transported face velocity, scene acceleration",
            writes: "predicted face velocity",
            feeds: "pressure RHS",
        }
    },
    "pressure-topology": {
        label: "Pressure topology",
        band: "pressure",
        substages: {
            "ptr-setup-brick-plan": { label: "PTR setup, seed and brick plan" },
            "pcm-cell-publication": { label: "PCM canonical cell publication" },
            "pcm-row-publication": { label: "Direct pressure-row membership publication" },
            "pca-fine-publication": { label: "PEI direct coefficients + PCA frontier" },
            "pca-coarse-repair": { label: "PCA brick + aggregate-edge repair" },
            "pca-hierarchy-and-freeze": { label: "PCA hierarchy repair + frozen coarse publication" },
            "pei-publication": { label: "PEI canonical pressure publication" },
            "ptr-commit-and-prepare-pressure": { label: "PTR commit, reopen + pressure preparation" }
        },
        slice: {
            caption: "The compact leaf set this solve runs on. The repair is incremental — seeded from the previous generation, walked over dirty worklists — but the classify pass is still a full accepted-cell scan.",
            keys: [
                { id: "pressure-cell", tone: "pressure", label: "pressure cell",
                    note: "over half full, so this cell is a leaf in the set the solve runs on" },
                { id: "two-to-one-port", tone: "adaptivity", label: "2:1 port",
                    note: "this brick meets a horizontal neighbour a rung away; the seam is a 2:1 port" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Incremental. Seeds the persistent pressure cache and the bounded topology repair from the prior accepted generation, classifies only changed cells and rows — bootstrap plus dirty worklists — with ghost-fluid theta at sparse air, repairs the brick, aggregate-edge and hierarchy caches, and assembles one symmetric GᵀWG operator across regular and 2:1 faces. Its timestamp is attributed to the topology generation accepted at the end of the prior advance; this advance's later commit is reported separately as next-frame input.",
            reads: "prior end-frame topology receipt, conditioned atlas, density-derived phi, temporal cell/row worklists and matched PCM generations",
            writes: "active rows, diagonal, nullspace components, repaired pressure cache",
            feeds: "this advance's RHS construction; the later topology commit feeds the next advance instead",
        }
    },
    "pressure-rhs": {
        label: "RHS + PCG initialization",
        band: "pressure",
        slice: {
            caption: "Divergence of the extended face field, one row per canonical incidence. Blue is compressing, red expanding; a converged solve drives every one of them to zero.",
            keys: [
                { id: "negative-divergence", tone: "pressure", label: "negative divergence",
                    note: "the cell is compressing; a converged solve drives this to zero" },
                { id: "positive-divergence", tone: "alarm", label: "positive divergence",
                    note: "the cell is expanding; a converged solve drives this to zero" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Builds the finite-volume divergence RHS from predicted face flux, geometric source volume and moving-solid capacity change, with enclosed components projected onto their compatible quotient space. It applies the brick-aggregate + hierarchy preconditioner once for the initial direction, reduces the initial true residual and primes the pipelined image the solve iterates on.",
            reads: "predicted face velocity, active pressure rows, pressure cache, source and solid-capacity rates",
            writes: "compatible RHS, initial direction, pipelined solver image",
            feeds: "sparse MGPCG",
        }
    },
    "pressure-solve": {
        label: "Pressure solve",
        band: "pressure",
        slice: {
            caption: "The solved pressure. One reduction per iteration, a single positive Jacobi diagonal as the preconditioner, and a true-residual guard every eighth iteration — the tail stays encoded whether or not it has converged.",
            keys: [
                { id: "high-pressure", tone: "pressure", label: "high pressure",
                    note: "solved pressure, above the free-surface value" },
                { id: "free-surface", tone: "ground", label: "free surface · p = 0",
                    note: "water with no solved pressure over it: the Dirichlet value the surface is pinned to" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Pipelined conjugate gradient with one uniform positive Jacobi preconditioner and one reduction per iteration. A guarded true-residual reduction after each fixed eight-iteration block gates later arithmetic or restarts the direction after curvature loss, and a final true residual closes the stage.",
            reads: "canonical incidence rows, diagonal, compatible RHS",
            writes: "compact leaf pressure, residual receipts",
            feeds: "velocity projection",
        }
    },
    "velocity-projection": {
        label: "Velocity projection",
        band: "pressure",
        substages: {
            "projection-faces": { label: "Composite pressure-gradient face projection + divergence" },
            "projected-frontier-commit": { label: "Projected transport frontier commit" },
            "projected-velocity-extension": { label: "Projected velocity extension" },
            "air-band-correction": { label: "Air-band velocity correction" },
            "momentum-snapshot": { label: "Pre-remesh momentum snapshot" },
            "projected-topology-rebuild": { label: "Projected commit compiled-topology rebuild" }
        },
        slice: {
            caption: "Grey is the field entering the projection, amber the divergence-free field leaving it. The difference is the pressure gradient, applied one row at a time.",
            keys: [
                { id: "before-projection", tone: "muted", label: "before projection",
                    note: "the grey arrow: the field entering the projection, gravity already in it" },
                { id: "after-projection", tone: "amber", label: "after projection",
                    note: "the amber arrow: the divergence-free field leaving it" },
            ],
            loopStep: 2,
        },
        tip: {
            summary: "Advances the incremental-activity clock, then projects the compiled dirty/pressure row masks directly through the same composite rows that built the divergence, conservative 2:1 ports and sparse-air boundaries included. Collocation publishes divergence maxima during its existing incidence traversal; rigid-body reaction and frame face output follow.",
            reads: "predicted face velocity, pressure, dirty bricks",
            writes: "projected face and collocated velocity, divergence receipts, frame face output",
            feeds: "activity measurement and the next frame's velocity extension",
        }
    },
    "activity-measurement": {
        label: "Activity census + frontier",
        band: "adaptivity",
        substages: {
            "dirty-brick-mask-publication": { label: "Incremental activity dirty-brick mask publication" },
            "brick-activity-measurement": { label: "Brick activity measurement and curvature" },
            "brick-activity-census-and-history": { label: "Brick activity census and history" },
            "sparse-world-frontier-allocation": { label: "Sparse-world frontier allocation" }
        },
        slice: {
            caption: "One score per brick, from interface presence and peak speed. This is the only number the resolution policy reads — geometry and motion, never an authored region.",
            keys: [
                { id: "high-activity", tone: "adaptivity", label: "high activity",
                    note: "interface presence and peak speed, scored; the only number the resolution policy reads" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "This interval is larger than its historical ‘activity measurement’ name implied. It publishes scalar/topology masks, measures and ages brick activity, seals the census and, when dynamic sparse-world growth is enabled, scans and synthesizes frontier pages.",
            reads: "conditioned density, momentum, previous records, dirty-brick worklist",
            writes: "activity masks, per-brick score/history and discovered sparse-world pages",
            feeds: "candidate topology planning",
        }
    },
    "resolution-planning": {
        label: "Candidate topology build",
        band: "adaptivity",
        substages: {
            "liquid-frontier-classification": { label: "Accepted-liquid frontier classification" },
            "refinement-policy-classification": { label: "Refinement-policy tile classification" },
            "initial-resolution-plan": { label: "Initial resolution plan" },
            "frontier-activation-and-retirement": { label: "Frontier activation and retirement" },
            "resolution-grading-and-validation": { label: "Repeated 2:1 grading and candidate validation" },
            "candidate-page-allocation-and-synthesis": { label: "Schedule backed candidate topology" }
        },
        slice: {
            caption: "The activity score becomes a target rung on the dyadic ladder — 1, 2, 4 or 8 cells per brick edge — then 2:1 grading pulls in any neighbour sitting more than one rung away.",
            keys: [
                { id: "target-rung", tone: "adaptivity", label: "target rung",
                    note: "the rung this brick is planned to carry, after 2:1 grading pulled its neighbours in" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "This is a candidate-topology construction interval, not just a policy decision. It scores and grades resolutions, consumes generation-stamped surface-output proofs, activates and retires pages, schedules the budget, allocates and synthesizes candidate cells, builds shadow row/leaf/structure worklists and publishes five indirect command copies for the following transaction.",
            reads: "transported density, momentum, policy history and accepted surface-output proofs",
            writes: "score/reason/proof history, urgent/ordinary queues, candidate levels, shadow worklists",
            feeds: "candidate transfer",
        }
    },
    "candidate-transfer": {
        label: "Candidate transfer",
        band: "adaptivity",
        substages: {
            "candidate-field-transfer": { label: "Candidate scalar + momentum field transfer" },
            "candidate-face-reconstruction": { label: "Candidate exterior-face reconstruction" },
            "candidate-face-validation": { label: "Candidate shadow-face validation" },
            "candidate-effects-preflight": { label: "Candidate effects census + preflight" },
            "candidate-ibo-construction": { label: "Candidate IBO delta construction" },
            "candidate-ibo-validation": { label: "Candidate independent IBO semantic validation" },
            "candidate-tei-compilation": { label: "Candidate TEI delta compilation" },
            "candidate-authorization": { label: "Candidate transaction authorization" },
            "candidate-ptr-publication": { label: "Candidate PTR effects publication" },
            "candidate-effects-seal": { label: "Candidate effects receipt seal" },
            "candidate-state-publication": { label: "Candidate fields + membership publication" },
            "candidate-image-replay": { label: "Candidate retired-image replay" }
        },
        slice: {
            caption: "Bricks whose rung moved this advance. The shadow topology is built beside the live one and committed as a single transaction at the frame tail — so this flip is the next advance's input, never this one's.",
            keys: [
                { id: "rung-changed", tone: "output", label: "rung changed",
                    note: "this brick's rung moved; the flip commits at the frame tail, so it is the next advance's input" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "One transaction over the topology delta resolution planning built: density, gamma, momentum and exterior faces transfer into double-buffered shadow slots; shadow faces are validated; the effects census and preflight run; the interned-boundary (IBO) delta is built and independently validated and the transport execution image (TEI) shadow compiled; the transaction is authorized; PTR effects publish; the receipt seals; fields and membership publish; retired images replay. That end-frame flip is input to the next advance's pressure topology, never this advance's.",
            reads: "shadow worklists, candidate levels, accepted cell and face state",
            writes: "shadow leaf/face storage, IBO + TEI deltas, pressure-row worklists, conservation receipts and accepted-generation metadata",
            feeds: "transport, pressure, projection, diagnostics and presentation through the next frame's indirect dispatches",
        }
    },
    "brick-retirement": {
        label: "Post-commit activity mask",
        band: "adaptivity",
        slice: {
            caption: "A brick holding no liquid and no source is released back to the atlas. The hatched bricks pay nothing this frame — the sparse set is the lit region plus its band, and no more.",
            keys: [
                { id: "retired-brick", tone: "muted", label: "retired brick",
                    note: "no liquid and no source: this brick is released back to the atlas" },
                { id: "resident-brick", tone: "liquid", label: "resident brick",
                    note: "still holding liquid, so this brick pays for the frame" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "Marks every brick the topology commit changed in the post-topology activity mask, so the next advance's direct face and activity transforms select exactly the bricks that moved. The decision to retire an unsupported empty brick is taken in resolution planning; this stage publishes the retired and reshaped brick bits.",
            reads: "committed topology, incremental-activity state",
            writes: "post-commit generation-stamped brick mask",
            feeds: "the next advance's face preparation and activity measurement",
        }
    },
    "presentation-publication": {
        label: "Presentation pages",
        band: "output",
        slice: {
            caption: "The surface the renderer receives: the PLIC segments of every interface cell, stitched across brick boundaries at whatever rung each brick happens to be carrying.",
            keys: [
                { id: "published-interface", tone: "output", label: "published interface",
                    note: "a PLIC chord in this cell is part of the surface the renderer receives" },
            ],
            loopStep: 4,
        },
        tip: {
            summary: "Classifies which bricks the renderer can see, publishes their compact level-set pages in place, proves whether each accepted B8 surface remains representable at B4, and commits frame control. Nothing is expanded to a dense field and nothing crosses to the host.",
            reads: "committed sparse authority",
            writes: "compact level-set brick pages, presentation classification and generation-stamped surface proofs",
            feeds: "renderer level-set/grid consumers and the next topology plan",
        }
    }
} satisfies Record<AdvanceStageId, AdvanceStageDeclaration>);
export function advanceStage(stage: AdvanceStageId): AdvanceStageDeclaration {
    return ADVANCE_STAGE_DECLARATIONS[stage];
}
