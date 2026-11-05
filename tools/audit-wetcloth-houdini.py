#!/usr/bin/env python3
"""Read named textual records from pinned HouNC archives for appearance review.

This does not execute Houdini expressions or certify shader equivalence. The
retained raw graph connections and parameters allow a later faithful material
translation rather than guessing a paper color from a preview screenshot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

COMMIT = 'b43993d775a0f9e3a9e32611f26936f2528fac37'
MATERIAL = re.compile(r'^(basecolor|rough|reflect|reflecttint|metallic|sheen|sheentint|transparency|transcolor|ior|coat|coatrough|emitcolor|basecolor_usePointColor|dryreflect|wetreflect)\s')
CAMERA = re.compile(r'^(t|r|s|focal|aperture|projection|res|winsize|win|near|far)\s')
RENDER = re.compile(r'^(camera|override_camerares|res_fraction|res_override|vm_colorspace|vm_picture)\s')
GRAPH = re.compile(r'^(parmname|parmtype|floatdef|colordef|vectordef|signature|input[1-4]|bias|min|max|value|color|const|amount|scale)\s')

def records(data: bytes) -> dict[str, str]:
    if not data.startswith(b'HouNC\x1a'):
        raise ValueError('Expected a released Houdini noncommercial archive')
    result = {}
    for part in data.split(b'HouNC\x1a')[1:]:
        header, separator, body = part.partition(b'\0')
        match = re.search(rb'((?:obj|out|shop|mat)/[^\x00]+)$', header)
        if not separator or not match:
            continue
        name = match.group(1).decode('ascii')
        if name.endswith(('.parm', '.def', '.init')):
            if name in result:
                raise ValueError(f'Duplicate Houdini record {name}')
            result[name] = body.decode('utf8', errors='strict')
    return result

def audit(path: Path) -> dict:
    raw = path.read_bytes()
    source = records(raw)
    retained = {}
    for name, body in source.items():
        lines = body.splitlines()
        if name.endswith('.parm') and name.startswith(('shop/', 'mat/')) and name.count('/') == 1:
            selected = [line for line in lines if MATERIAL.match(line)]
        elif name.endswith('.parm') and re.match(r'obj/cam[^/]*\.parm$', name):
            selected = [line for line in lines if CAMERA.match(line)]
        elif name.endswith('.parm') and name.startswith('out/'):
            selected = [line for line in lines if RENDER.match(line)]
        elif name.endswith('.parm') and name.startswith('obj/') and 'snippet\t' in body:
            # VEX snippets can span lines. Retain the parameter record exactly.
            selected = lines
        elif name.startswith('shop/cloth/') and name.count('/') == 2 and name.endswith(('.def', '.parm', '.init')):
            # Retain connections and scalar/color defaults, without embedding
            # Houdini's large UI templates or every inactive parameter type.
            if name.endswith('.def'):
                match = re.search(r'inputsNamed3\s*\{.*?\n\}', body, re.S)
                selected = [match.group(0)] if match else []
            elif name.endswith('.init'):
                selected = [line for line in lines if line.startswith('type = ')]
            else:
                selected = [line for line in lines if GRAPH.match(line) or MATERIAL.match(line)]
        else:
            selected = []
        if selected:
            retained[name] = '\n'.join(selected)
    return {'sourcePath': f'houdini/{path.name}',
            'sourceUrl': f'https://github.com/raymondyfei/libwetcloth/blob/{COMMIT}/houdini/{path.name}',
            'sha256': hashlib.sha256(raw).hexdigest(), 'byteLength': len(raw),
            'recordCount': len(source), 'records': retained,
            'interpretation': 'Selected raw Houdini parameters and direct cloth graph connections; expressions and shader execution are not evaluated. Not a complete HIP conversion.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = {'schemaVersion': 1, 'commit': COMMIT, 'reproductionValidated': False,
              'scenes': [audit(path) for path in args.archives]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
