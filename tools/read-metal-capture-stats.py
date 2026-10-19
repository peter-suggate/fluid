#!/usr/bin/env python3
"""Read compiler evidence from a local Xcode Metal capture without replaying it.

Usage: python3 tools/read-metal-capture-stats.py CAPTURE.gputrace --out DIR

This reader supports the MTSP 0x400 / cidx v0 files produced by the installed
Xcode. These are diagnostic file formats, not a public API; reject other
versions rather than interpreting them as this layout. Every statistic is
joined by the captured pipeline object ID to an actual dispatch. Preserve the
capture as the authority. Instruction counts are static, not executed counts;
spilled bytes are the compiler's allocation figure, not measured traffic.
"""
import argparse
from collections import Counter
import json
import plistlib
import re
import struct
import zlib
from pathlib import Path


class CaptureStore:
    def __init__(self, path):
        self.path = Path(path)
        data = (self.path / "index").read_bytes()
        magic, version, buckets, count, repeated = struct.unpack_from("<5I", data)
        assert magic == 1667851384 and version == 0 and count == repeated, "unsupported capture index"
        entries = 20 + buckets * 12
        lengths = entries + count * 24
        cursor = lengths + count * 2
        self.entries = {}
        self.stores = {}
        for index in range(count):
            length = struct.unpack_from("<H", data, lengths + index * 2)[0]
            name = data[cursor:cursor + length].rstrip(b"\0").decode()
            cursor += length
            self.entries[name] = struct.unpack_from("<IIQII", data, entries + index * 24)
        assert cursor == len(data), "capture index size mismatch"

    def read(self, name):
        external = self.path / name
        if external.is_file():
            return external.read_bytes()
        size, compressed, offset, flags, store = self.entries[name]
        assert flags in (0, 1), "unsupported storage flags"
        if store not in self.stores:
            self.stores[store] = (self.path / f"store{store}").read_bytes()
        data = self.stores[store][offset:offset + compressed]
        data = zlib.decompress(data) if flags else data
        assert len(data) == size, "capture object size mismatch"
        return data


def archive_dictionary(data):
    archive = plistlib.loads(data)
    assert archive.get("$archiver") == "NSKeyedArchiver"
    objects = archive["$objects"]

    def decode(value):
        if isinstance(value, plistlib.UID):
            return decode(objects[value.data])
        if isinstance(value, dict):
            if "NS.keys" in value:
                return {decode(k): decode(v) for k, v in zip(value["NS.keys"], value["NS.objects"])}
            if "NS.objects" in value:
                return [decode(v) for v in value["NS.objects"]]
            return {k: decode(v) for k, v in value.items() if k != "$class"}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(archive["$top"]["root"])


def records(path):
    data = path.read_bytes()
    assert data[:4] == b"MTSP" and struct.unpack_from("<I", data, 4)[0] == 0x400, "unsupported MTSP"
    cursor = 8
    while cursor < len(data):
        size, opcode = struct.unpack_from("<Ii", data, cursor)
        assert size >= 40 and cursor + size <= len(data), "invalid capture record"
        record = data[cursor:cursor + size]
        end = record.index(b"\0", 36)
        signature = record[36:end].decode("ascii")
        arguments = record[(end + 4) // 4 * 4:]
        yield cursor, opcode, signature, arguments
        cursor += size
    assert cursor == len(data)


def text_at(data, offset):
    return data[offset:].split(b"\0", 1)[0].decode()


def instruction_locations(remarks, instruction_count):
    """Compiler basic-block locations, not per-source-line execution costs.

    The constant program has its own instructions, excluded from the headline
    pipeline instruction count. Check the remaining sum before using locations.
    A block's debug location can cover multiple operations and inlined helpers.
    """
    locations = Counter()
    total = 0
    for part in remarks.split("--- "):
        if not re.search(r"Name: +InstructionMix\n", part):
            continue
        if re.search(r"Function: +agc.main.constant_program(?:\.\S+)?\n", part):
            continue
        count = sum(map(int, re.findall(r"- INST_[^:\n]*: +'([0-9]+)'", part)))
        total += count
        location = re.search(r"DebugLoc: +\{ File: program_source,\s+Line: (\d+)", part)
        if location and int(location[1]) > 0:
            locations[int(location[1])] += count
    return {"instructionSum": total, "matchesPipelineCount": total == instruction_count,
            "topSourceLocations": [{"line": line, "staticBlockInstructions": count}
                                   for line, count in locations.most_common(20)]}


def inspect_capture(path, output):
    store = CaptureStore(path)
    stats = {}
    descriptors = {}
    functions = {}
    libraries = {}
    sources = {}
    # Only resources actually used in the captured frame, not the separately
    # retained unused-device-resources collection.
    for resource in sorted(path.glob("device-resources-*")):
        for offset, op, signature, args in records(resource):
            if op == -16200 and signature == "CUt":  # compute pipeline creation
                descriptors[struct.unpack_from("<Q", args, len(args) - 8)[0]] = text_at(args, 8)
            elif op == -16290 and signature == "CS":  # library -> function
                functions[struct.unpack_from("<Q", args, len(args) - 8)[0]] = struct.unpack_from("<Q", args)[0]
            elif op == -16305 and signature == "CU<b>Ut":  # source -> library
                libraries[struct.unpack_from("<Q", args, len(args) - 8)[0]] = text_at(args, 8)
            elif op == -12272 and signature == "CiUul":
                name = text_at(args, 12)
                data = store.read(name)
                if not data.startswith(b"bplist"):
                    continue
                decoded = archive_dictionary(data)
                if "Instruction count" not in decoded:
                    continue
                pipeline = struct.unpack_from("<Q", args)[0]
                stats[pipeline] = (name, decoded, {"file": resource.name, "offset": offset})

    labels, pipelines, shared, ordinals = {}, {}, {}, {}
    dispatches = []
    for offset, op, signature, args in records(path / "capture"):
        if len(args) < 8 or not signature.startswith("C"):
            continue
        encoder = struct.unpack_from("<Q", args)[0]
        if op == -16339 and signature == "CS":
            labels[encoder] = text_at(args, 8)
            ordinals[encoder] = 0
        elif op == -16337 and signature == "Ct":
            pipelines[encoder] = struct.unpack_from("<Q", args, 8)[0]
        elif op == -16328 and signature == "Culul":
            length, index = struct.unpack_from("<QQ", args, 8)
            shared.setdefault(encoder, {})[index] = length
        elif (op == -16327 and signature == "C@3ul@3ul") or (op == -16220 and signature == "Ctul@3ul"):
            pipeline = pipelines.get(encoder)
            ordinals[encoder] = ordinals.get(encoder, 0) + 1
            indirect = op == -16220
            item = {
                "stage": labels.get(encoder), "ordinal": ordinals[encoder],
                "pipeline": hex(pipeline or 0), "captureOffset": offset,
                "dispatchKind": "indirect" if indirect else "direct",
                "groups": None if indirect else struct.unpack_from("<QQQ", args, 8),
                "threads": struct.unpack_from("<QQQ", args, 24 if indirect else 32),
                "threadgroupMemoryBindings": dict(shared.get(encoder, {})),
            }
            if indirect:
                item["indirectBuffer"] = hex(struct.unpack_from("<Q", args, 8)[0])
                item["indirectOffset"] = struct.unpack_from("<Q", args, 16)[0]
            if pipeline in descriptors:
                descriptor_file = descriptors[pipeline]
                descriptor = store.read(descriptor_file)
                assert struct.unpack_from("<Q", descriptor)[0] == 120, "unsupported compute descriptor"
                function = struct.unpack_from("<Q", descriptor, 8)[0]
                item["descriptorFile"] = descriptor_file
                item["pipelineLabel"] = text_at(descriptor, 24)
                if function in functions and functions[function] in libraries:
                    item["sourceFile"] = libraries[functions[function]]
            if pipeline in stats:
                name, values, reference = stats[pipeline]
                item["statsFile"] = name
                item["statsReference"] = reference
                item["stats"] = {k: v for k, v in values.items() if isinstance(v, (int, float, bool))}
                remarks = values.get("Remarks", "")
                item["spillRemarks"] = [part.strip() for part in remarks.split("--- ")
                                        if "NumSpills:" in part or "NumReloads:" in part]
                item["stackRemarks"] = [part.strip() for part in remarks.split("--- ")
                                        if re.search(r"Name: +StackSize\n", part)]
                item["instructionLocations"] = instruction_locations(remarks, values["Instruction count"])
                if "sourceFile" in item:
                    source = item["sourceFile"]
                    sources[source] = store.read(source)
            dispatches.append(item)

    assert dispatches and any(d.get("stats") for d in dispatches), "no compiler statistics joined to dispatches"
    output.mkdir(parents=True, exist_ok=True)
    for name, data in sources.items():
        # Capture source strings include their C terminator. Strip only that
        # terminator so ordinary text tools can inspect the generated MSL.
        (output / f"{name}.metal").write_bytes(data.removesuffix(b"\0"))
    result = {"capture": str(path), "format": "MTSP 0x400 / cidx v0",
              "note": "Static compiler figures, not executed instruction or spill traffic counts. Memory bindings are encoded dynamic lengths; unused bindings may persist.",
              "dispatches": dispatches}
    (output / "dispatch-stats.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "dispatches": len(dispatches),
                      "withStats": sum("stats" in d for d in dispatches), "sources": len(sources)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    inspect_capture(arguments.capture.resolve(), arguments.out.resolve())
