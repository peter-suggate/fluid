#!/usr/bin/env python3
"""Import the pinned Fei et al. water scenes without replacing their topology.

Source XML is an offline input, never bundled into browser JS. Geometry becomes
little-endian WCL1 arrays, gzip-compressed deterministically. Small scene metadata
retains every physical source, support program and original scalar setting.
"""
from __future__ import annotations

import argparse
import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b43993d775a0f9e3a9e32611f26936f2528fac37"
RAW = f"https://raw.githubusercontent.com/raymondyfei/libwetcloth/{COMMIT}/"
EXCLUDED = {"pore_test_acetal", "pore_test_oil", "pore_test_honey"}
CM = 0.01
LIQUID_DEFAULTS = {
    "liquidDensity": 1.0, "restContactAngle": math.radians(40.8),
    "surfTensionCoeff": 72.8, "viscosity": 8.9e-3,
    "yarnDiameter": 0.01, "restVolumeFraction": 0.4, "poreRadius": 0.005,
    "yazdchiPower": 1.6, "useNonlinearDrag": 1, "applyPressureSolid": 1,
    "solveSolid": 1, "initNonuniformFraction": 0, "surfTension": 0,
    "computeViscosity": 0, "useVaryingFraction": 0,
}
SCALAR_TAGS = set("viscosity surfTensionCoeff flipCoeff elastoFlipCoeff elastoFlipAsymCoeff halfThickness yarnDiameter restVolumeFraction multiLevel solidCohesion elastoAdvectCoeff cohesion useNonlinearDrag bendingScheme yazdchiPower softCohesion particleCellMultiplier useVaryingFraction useLevelSetForce restContactAngle fabricThreadCount dragByFutureSolid solveSolid initNonuniformFraction surfTension checkDivergence surfTensionSmoothingStep correctionStep applyPressureSolid".split())
ELASTIC_TAGS = set("radius youngsModulus poissonRatio collisionMultiplier attachMultiplier density viscosity baseRotation accumulateWithViscous accumulateViscousOnlyForBendingModes frictionAngle friction_angle".split())
ROOT_TAGS = set("description duration integrator collision bucketinfo camera liquidinfo simplegravity ElasticParameters distancefield particle cloth hair script".split())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def numeric(value: str) -> float | str:
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite source value: {value}")
        return result
    except ValueError:
        if value in ("tantheta", "sintheta", "angle"):
            return value
        raise


def values(node: ET.Element | None, admitted: set[str]) -> dict:
    if node is None:
        return {}
    result = {}
    for child in node:
        if child.tag not in admitted or set(child.attrib) != {"value"}:
            raise ValueError(f"Unknown scalar setting {child.tag}: {child.attrib}")
        if child.tag in result:
            raise ValueError(f"Duplicate scalar setting {child.tag}")
        result[child.tag] = numeric(child.attrib["value"])
    return result


def vec(a: dict, keys: tuple[str, ...], scale: float = 1.0) -> list[float]:
    return [float(a.get(k, 0)) * scale for k in keys]


def packed(kind: str, data: list) -> bytes:
    a = array.array("f" if kind == "f32" else "I", data)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def field(node: ET.Element, index: int) -> dict:
    a = node.attrib
    kind = a["type"]
    if kind not in {"sphere", "box", "cylinder", "intersect", "file"}:
        raise ValueError(f"Unsupported source distance field {kind}")
    usage = a.get("usage", "solid")
    if usage not in {"solid", "source", "terminator"}:
        raise ValueError(f"Unknown distance-field usage {usage}")
    result = {
        "index": index, "kind": kind, "usage": usage,
        "group": int(a.get("group", 0)), "inside": bool(int(a.get("inside", 0))),
        "sampled": bool(int(a.get("sampled", 1))), "sourceAttributes": a,
    }
    if kind == "intersect":
        result["children"] = [int(v) for v in a["i"].split()]
        if not result["children"] or any(i < 0 or i >= index for i in result["children"]):
            raise ValueError("Source CSG must reference earlier fields")
    else:
        result.update(center_m=vec(a, ("cx", "cy", "cz"), CM),
                      axis=vec(a, ("rx", "ry", "rz")), angle_rad=float(a.get("rw", 0)))
        if kind == "sphere":
            result["radius_m"] = float(a["radius"]) * CM
        elif kind == "box":
            result["halfExtents_m"] = vec(a, ("ex", "ey", "ez"), CM)
            result["roundingRadius_m"] = float(a.get("radius", 0)) * CM
        elif kind == "cylinder":
            result["radius_m"] = float(a["radius"]) * CM
            result["halfLength_m"] = float(a["halflength"]) * CM
            result["roundingRadius_m"] = float(a.get("corner", 0)) * CM
        else:
            result.update(filename=a["filename"], scale=float(a.get("scale", 1)),
                          sampleSpacing_m=float(a.get("dx", 0)) * CM)
        durations = list(node.findall("duration"))
        # Upstream fields without durations sample once at time zero.
        result["emission"] = [{
            "start_s": float(d.get("start", 0)), "end_s": float(d.get("end", 0)),
            "velocity_m_s": vec(d.attrib, ("vx", "vy", "vz"), CM),
        } for d in durations] if durations else [{"start_s": 0, "end_s": 0, "velocity_m_s": vec(a, ("vx", "vy", "vz"), CM)}]
    return result


def script(node: ET.Element) -> dict:
    a = node.attrib
    if a["type"] not in {"rotate", "translate"}:
        raise ValueError(f"Unsupported support script {a['type']}")
    start, end = float(a["start"]), float(a["end"])
    result = {"kind": a["type"], "group": int(a.get("group", 0)),
              "start_s": start, "end_s": end, "easing": "cubic",
              "easeStart_s": (end - start) / 3, "easeEnd_s": (end - start) / 3,
              "global": bool(int(a.get("global", 0))), "sourceAttributes": a}
    if a["type"] == "translate":
        result["displacement_m"] = vec(a, ("x", "y", "z"), CM)
    else:
        result["axis"] = vec(a, ("x", "y", "z"))
        result["angle_rad"] = float(a["w"])
        if "ox" in a:
            result["origin_m"] = vec(a, ("ox", "oy", "oz"), CM)
    return result


def import_scene(path: Path, record: dict) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if sha(raw) != record["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    root = ET.fromstring(raw)
    if root.tag != "scene" or any(c.tag not in ROOT_TAGS for c in root):
        raise ValueError(f"Unknown XML root content in {path}")
    source_liquid = values(root.find("liquidinfo"), SCALAR_TAGS)
    liquid = {**LIQUID_DEFAULTS, **source_liquid}
    particle_nodes = root.findall("particle")
    n = len(particle_nodes)
    arrays: dict[str, tuple[str, list]] = {key: (kind, []) for key, kind in [
        ("restPositions_m", "f32"), ("initialVelocities_m_s", "f32"),
        ("fixed", "u32"), ("groups", "u32"), ("solidFraction", "f32"),
        ("triangles", "u32"), ("triangleMaterials", "u32"),
        ("rodOffsets", "u32"), ("rodIndices", "u32"), ("rodMaterials", "u32") ]}
    for p in particle_nodes:
        if set(p.attrib) - {"x", "v", "fixed", "group", "vf"}:
            raise ValueError(f"Unknown particle attributes {p.attrib}")
        for k, attr in [("restPositions_m", "x"), ("initialVelocities_m_s", "v")]:
            v = [float(x) * CM for x in p.get(attr, "0 0 0").split()]
            if len(v) != 3 or not all(math.isfinite(x) for x in v):
                raise ValueError("Malformed particle vector")
            arrays[k][1].extend(v)
        arrays["fixed"][1].append(int(p.get("fixed", 0)))
        arrays["groups"][1].append(int(p.get("group", 0)))
        arrays["solidFraction"][1].append(float(p.get("vf", 1)) if liquid["initNonuniformFraction"] else liquid["restVolumeFraction"])
    materials = []
    for i, e in enumerate(root.findall("ElasticParameters")):
        v = values(e, ELASTIC_TAGS)
        radius = float(v.get("radius", .018)) * CM
        # The parser's second radius has its own default, even when radius
        # is explicitly changed. Do not replace elliptical material sections.
        biradius = float(v.get("biradius", .018)) * CM
        materials.append({"id": f"material-{i}", "radius_m": radius,
                          "radiusB_m": biradius, "thickness_m": radius + biradius,
                          "youngModulus_Pa": float(v.get("youngsModulus", 6.687e5)) * .1,
                          "poissonRatio": v.get("poissonRatio", .35),
                          "density_kg_m3": float(v.get("density", 1.3)) * 1000,
                          "solidViscosity_Pa_s": float(v.get("viscosity", 0)) * .1,
                          "sourceSettings": v})
    cloths = []
    for c in root.findall("cloth"):
        mat = int(c.get("params", 0))
        if mat < 0 or mat >= len(materials):
            raise ValueError("Unknown cloth material")
        first = len(arrays["triangles"][1]) // 3
        for f in c:
            if f.tag != "face":
                raise ValueError("Unknown cloth element")
            ids = [int(x) for x in f.attrib["i"].split()]
            if len(ids) != 3 or len(set(ids)) != 3 or any(x < 0 or x >= n for x in ids):
                raise ValueError("Invalid triangle indices")
            arrays["triangles"][1].extend(ids)
            arrays["triangleMaterials"][1].append(mat)
        cloths.append({"firstTriangle": first, "triangleCount": len(c), "material": mat})
    arrays["rodOffsets"][1].append(0)
    for h in root.findall("hair"):
        mat = int(h.get("params", 0))
        ids = [int(p.attrib["i"]) for p in h.findall("p")]
        if "start" in h.attrib:
            if ids:
                raise ValueError("Ambiguous rod topology")
            start = int(h.attrib["start"])
            ids = list(range(start, start + int(h.attrib["count"])))
        if len(ids) < 2 or any(x < 0 or x >= n for x in ids) or not 0 <= mat < len(materials):
            raise ValueError("Invalid rod topology")
        arrays["rodIndices"][1].extend(ids)
        arrays["rodOffsets"][1].append(len(arrays["rodIndices"][1]))
        arrays["rodMaterials"][1].append(mat)
    descriptors = []
    chunks = []
    offset = 0
    for key, (kind, data) in arrays.items():
        descriptors.append({"name": key, "type": kind, "offset": offset, "count": len(data)})
        b = packed(kind, data)
        chunks.append(b)
        offset += len(b)
    header = {"version": 1, "sourceSha256": sha(raw), "vertexCount": n,
              "triangleCount": len(arrays["triangles"][1]) // 3,
              "rodCount": len(arrays["rodMaterials"][1]), "arrays": descriptors}
    head = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    head += b" " * (-len(head) % 4)
    payload = b"WCL1" + struct.pack("<I", len(head)) + head + b"".join(chunks)
    asset_path = f"/assets/wetcloth/{path.stem}.wcl.gz"
    camera = root.find("camera")
    source_camera = None if camera is None else {
        "distance_m": float(camera.get("dist")) * CM,
        "radius_m": float(camera.get("radius")) * CM,
        "fov_deg": float(camera.get("fov")),
        "center_m": vec(camera.find("center").attrib, ("x", "y", "z"), CM),
        "rotation": {k: float(v) for k, v in camera.find("rotation").attrib.items()},
    }
    gravity = root.find("simplegravity")
    fields = [field(e, i) for i, e in enumerate(root.findall("distancefield"))]
    child_ids = {i for f in fields for i in f.get("children", [])}
    for f in fields:
        f["root"] = f["index"] not in child_ids
    bucket = root.find("bucketinfo").attrib
    metadata = {
        "id": record["id"], "name": path.stem.replace("_", " "), "family": record["family"],
        "source": {"commit": COMMIT, "path": record["source_path"], "url": record["source_url"], "sha256": sha(raw)},
        "asset": {"kind": "asset", "url": asset_path, "sha256": sha(payload), "vertexCount": n},
        "counts": {"vertices": n, "triangles": header["triangleCount"], "rods": header["rodCount"]},
        "duration_s": float(root.find("duration").get("time")),
        "dt_s": float(root.find("integrator").get("dt")),
        "cellSize_m": float(bucket["size"]) * CM / int(bucket["numcells"]),
        "gravity_m_s2": vec(gravity.attrib, ("fx", "fy", "fz"), CM) if gravity is not None else [0, 0, 0],
        "materials": materials, "cloths": cloths, "fields": fields,
        "scripts": [script(e) for e in root.findall("script")], "camera": source_camera,
        "liquid": {"density_kg_m3": float(liquid["liquidDensity"]) * 1000,
                   "dynamicViscosity_Pa_s": float(liquid["viscosity"]) * .1,
                   "surfaceTension_N_m": float(liquid["surfTensionCoeff"]) * .001,
                   "contactAngle_rad": liquid["restContactAngle"],
                   "fiberDiameter_m": float(liquid["yarnDiameter"]) * CM,
                   "solidFraction": liquid["restVolumeFraction"]},
        "sourceSettings": {"liquid": source_liquid, "integrator": root.find("integrator").attrib,
                           "bucket": bucket, "collision": root.find("collision").attrib},
        "auditNotes": record["audit_notes"], "externalDependencies": record["external_dependencies"],
        "reproductionStatus": "source-imported-unverified",
    }
    return metadata, gzip.compress(payload, compresslevel=9, mtime=0)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-root", type=Path, default=Path("/tmp/fluid-wetcloth-research"))
    cli.add_argument("--fetch", action="store_true", help="Fetch missing pinned source XML, verifying inventory hashes")
    cli.add_argument("--check", action="store_true", help="Compare generated files without writing")
    args = cli.parse_args()
    inventory = json.loads((ROOT / "docs/plans/wetcloth-scene-inventory-2026-11-05.json").read_text())
    outputs: dict[Path, bytes] = {}
    scenes = []
    for record in inventory["scenes"]:
        path = Path(record["source_path"])
        if path.stem in EXCLUDED:
            continue
        source = args.source_root / path
        if not source.exists() and args.fetch:
            source.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(RAW + path.as_posix(), source)
        metadata, payload = import_scene(source, record)
        outputs[ROOT / "public" / metadata["asset"]["url"].lstrip("/")] = payload
        scenes.append(metadata)
    catalog = {"schemaVersion": 1, "upstreamCommit": COMMIT, "sourceUnits": "CGS", "runtimeUnits": "SI",
               "scope": "36 released water inputs; non-water liquids excluded", "scenes": scenes}
    outputs[ROOT / "lib/cloth/scenes/source-catalog.json"] = (json.dumps(catalog, indent=2, ensure_ascii=True) + "\n").encode()
    failures = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    if failures:
        raise SystemExit("Stale wet-cloth imports: " + ", ".join(failures))
    print(f"{'Verified' if args.check else 'Imported'} {len(scenes)} water sources; {sum(len(v) for v in outputs.values()):,} bytes")


if __name__ == "__main__":
    main()
