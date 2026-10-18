"""Bake all authored hero GLBs using Blender 5.2; publish headers only on success."""
import argparse, concurrent.futures, json, os, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--resume',action='store_true')
a=p.parse_args(); blender=os.environ.get('BLENDER','/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender')
out=ROOT/'public/assets/hero-garden-blender';meta=ROOT/'lib/core/imported-assets/metadata';logs=ROOT/'artifacts/hero-garden-blender/bake';logs.mkdir(parents=True,exist_ok=True)
assets=json.loads((out/'manifest.json').read_text())['assets']
def bake(asset):
    name=asset['id']
    if not(a.resume and (out/asset['volume']).exists() and (meta/(name+'-volume.json')).exists()):
        with (logs/(name+'.log')).open('w') as log:
            subprocess.run([blender,'-b','--threads','2','--python-exit-code','1','--python',str(ROOT/'tools/blender/bake_mesh_volume.py'),'--','--collection','hero-garden-blender','--asset',name,'--cell',str(asset['cellSize']),'--band-cells',str(asset.get('bandCells',3))],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Baked '+name,flush=True)
    return name,json.loads((meta/(name+'-volume.json')).read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool: headers=dict(pool.map(bake,assets))
(meta/'hero-garden-volumes.json').write_text(json.dumps(headers,indent=2)+'\n')
print('Published '+str(len(headers))+' verified asset headers',flush=True)
