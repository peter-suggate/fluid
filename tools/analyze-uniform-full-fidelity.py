"""Compare full-occupancy atlas captures. Requires numpy and repo Node/tsx.

Recompute owner-aware quality with the existing CPU census when raw fields are
available. This script never starts a GPU device or changes acceptance gates.
"""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]
captures=root/'artifacts/uniform-full-fidelity-2026-10-03'
output=root/'docs/plans/uniform-coarse-first-2026-10-03/full-fidelity-evidence.json'
rows=[]
for path in sorted(captures.glob('*.json')):
    d=json.loads(path.read_text())
    if 'sourceFingerprint' not in d: continue
    row={'capture':path.stem,**{k:d.get(k) for k in ['sourceFingerprint','sourceFingerprintAfter','experimentFingerprintBefore','experimentFingerprintAfter','adapter','atlasMode','atlasEdge','atlasCounts','atlasMetadataBytes','sceneId','kind','frames','dt_s','warmup','discardFrames','values','throughput','lattice','setup_ms','fieldHashes','failure','validationErrors','summary']}}
    row['allocationBytes']=d.get('initial',{}).get('allocatedBytes')
    row['finalWork']={k:v for k,v in d.get('final',{}).items() if k.startswith(('uniformMixed','uniformPressure','volume','maxSpeed','encodedSteps'))}
    family='mini64-fine' if path.stem.startswith('mini64-fine-') else 'fig9' if path.stem.startswith('fig9-') else None
    if family and not d.get('failure'):
        baseline=f'{family}-native-a'
        bpath=captures/(baseline+'.json')
        if bpath.exists() and not json.loads(bpath.read_text()).get('failure'):
            b=json.loads(bpath.read_text())
            for k in ['sceneId','kind','frames','dt_s','discardFrames','lattice','values','scene','sourceFingerprint','sourceFingerprintAfter','experimentFingerprintBefore','experimentFingerprintAfter']:
                if d.get(k)!=b.get(k):raise ValueError(f'{path.stem} mismatch: {k}')
            row['baseline']=baseline
            row['exactFinalHashes']=d.get('fieldHashes')==b.get('fieldHashes')
            row['fieldDifferences']={}
            for field in ['volume','velocity','phi','tiles']:
                dtype=np.uint32 if field=='tiles' else np.float32
                ext='u32' if field=='tiles' else 'f32'
                a=np.fromfile(captures/path.stem/f'{field}.{ext}',dtype=dtype)
                ref=np.fromfile(captures/baseline/f'{field}.{ext}',dtype=dtype)
                if a.shape!=ref.shape or not np.isfinite(a).all():raise ValueError(f'{path.stem}: invalid {field}')
                delta=a.astype(np.float64)-ref.astype(np.float64)
                denom=float(np.linalg.norm(ref.astype(np.float64)))
                row['fieldDifferences'][field]={'values':a.size,'differentValues':int(np.count_nonzero(a!=ref)),'maxAbsolute':float(np.max(np.abs(delta))),'relativeL2':float(np.linalg.norm(delta)/denom) if denom else None}
    rows.append(row)
summary=[]
for family in ['mini64-fine','fig9']:
    selected=[r for r in rows if r['capture'].startswith(family+'-') and r.get('throughput') and not r.get('failure')]
    native=[r['summary']['msPerStep'] for r in selected if r['atlasMode']=='native']
    if not native:continue
    base=statistics.mean(native)
    for mode in ['native','dense','affine','table']:
        arms=[r for r in selected if r['atlasMode']==mode]
        if not arms:continue
        times=[r['summary']['msPerStep'] for r in arms]
        summary.append({'family':family,'mode':mode,'n':len(times),'msPerStep':times,'mean':statistics.mean(times),'overheadPercent':100*(statistics.mean(times)/base-1),'allFinalHashesExact':all(r.get('exactFinalHashes') for r in arms)})
frozen=[]
for path in sorted(captures.glob('frozen-*.json')):
    d=json.loads(path.read_text())
    frozen.append({'capture':path.stem,**{k:d.get(k) for k in ['sceneId','mode','edge','samples','fine','identity','sourceBefore','sourceAfter','scope','setup_ms','dimensions','adapter','inputHashes','outputHashes','replayedHashes','summary','gpu_ms','validationErrors','checkpoint','failure','invalidMeasurement']},
        'acceptedWork':{k:v for k,v in d.get('accepted',{}).items() if k.startswith(('uniformMixed','uniformPressure','encodedSteps'))}})
valid_frozen=[r for r in frozen if not r.get('failure') and not r.get('invalidMeasurement')]
frozen_summary=[]
if valid_frozen:
    baseline=next(r for r in valid_frozen if r['capture']=='frozen-mini64-native-a')
    for r in valid_frozen:
        for key in ['sceneId','edge','fine','identity','sourceBefore','sourceAfter','dimensions','inputHashes','checkpoint']:
            if r[key]!=baseline[key]:raise ValueError(f"{r['capture']} frozen mismatch: {key}")
        if r['sourceBefore']!=r['sourceAfter'] or r['validationErrors'] or r['outputHashes']!=r['replayedHashes']:
            raise ValueError(f"{r['capture']} invalid frozen replay")
        r['exactNativeOutput']=r['outputHashes']==baseline['outputHashes']
        r['exactNativeAcceptedWork']=r['acceptedWork']==baseline['acceptedWork']
    native=statistics.mean(r['summary']['mean'] for r in valid_frozen if r['mode']=='native')
    for mode in ['native','dense','affine','table']:
        arms=[r for r in valid_frozen if r['mode']==mode]
        if not arms:continue
        means=[r['summary']['mean'] for r in arms]
        frozen_summary.append({'mode':mode,'runs':len(arms),'samplesPerRun':[r['samples'] for r in arms],
            'runMeans_ms':means,'mean_ms':statistics.mean(means),'overheadPercent':100*(statistics.mean(means)/native-1),
            'exactNativeOutput':all(r['exactNativeOutput'] for r in arms),'exactNativeAcceptedWork':all(r['exactNativeAcceptedWork'] for r in arms)})
quality_path=captures/'field-quality.json'
quality=json.loads(quality_path.read_text()) if quality_path.exists() else {}
quality_names=[name for name in ['fig9-native-a','fig9-affine-a','fig9-table-a','fig9-dense-a'] if (captures/name/'volume.f32').exists()]
if quality_names:
    census_script=r'''
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {uniformQualityCensus} from './tools/uniform-quality-census.ts';
const [directory,...names]=process.argv.slice(1), result={};
for(const name of names){
 const d=JSON.parse(readFileSync(join(directory,name+'.json'),'utf8'));
 const read=(file,Type)=>{const b=readFileSync(join(directory,name,file));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
 const {projection,phiSlice,...quality}=uniformQualityCensus([d.lattice.nx,d.lattice.ny,d.lattice.nz],read('tiles.u32',Uint32Array),read('volume.f32',Float32Array),read('phi.f32',Float32Array));
 result[name]=quality;
}
process.stdout.write(JSON.stringify(result));
'''
    quality.update(json.loads(subprocess.check_output(['node','--import','tsx','--input-type=module','-e',census_script,str(captures),*quality_names],cwd=root,text=True)))
    quality_path.write_text(json.dumps(quality,indent=2)+'\n')
diagnostics=[]
smoke_baseline=captures/'mini64-native-smoke.json'
if smoke_baseline.exists():
    baseline=json.loads(smoke_baseline.read_text())
    reference={q['frame']:q for q in baseline['qualitySnapshots']}
    for name in ['mini64-native-repeat','smoke-affine','mini64-table-smoke','mini64-identity-smoke']:
        path=captures/(name+'.json')
        if not path.exists():continue
        d=json.loads(path.read_text())
        frames=[]
        for q in d['qualitySnapshots']:
            b=reference[q['frame']]
            frames.append({'frame':q['frame'],'massDelta_cells':q['mass']-b['mass'],
                'centroidDelta_h':[a-r for a,r in zip(q['centroid_cells'],b['centroid_cells'])],
                'projectionMaxDifference':float(np.max(np.abs(np.array(q['projection'])-b['projection']))),
                'phiSliceMaxDifference_m':float(np.max(np.abs(np.array(q['phiSlice'])-b['phiSlice']))),
                'fieldHashes':q.get('fieldHashes'),
                'exactNativeFields':{k:v==q.get('fieldHashes',{}).get(k) for k,v in b.get('fieldHashes',{}).items()}})
        diagnostics.append({'capture':name,'baseline':'mini64-native-smoke','frames':frames})
output.write_text(json.dumps({'ownerQuality':quality,'numericalDiagnostics':diagnostics,'frozenSummary':frozen_summary,'frozen':frozen,'scope':'Full-occupancy, static mirror page placement. Same numerical operators and automatic masks. No sparse allocation, residency transitions or renderer benchmark. Dense mode specializes to original native shader source.','summary':summary,'captures':rows},indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(json.dumps(frozen_summary,indent=2))
print(output)
if '--plot' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),layout='constrained')
    modes=['native','dense','affine','table']
    labels=['Native','Dense specialization','Arithmetic pages','Translation table']
    colours=['#245c70','#4c8775','#bb7628','#ba523c']
    for ax,records,value,title,note in [
        (axes[0],frozen_summary,'mean_ms','Controlled all-h frame replay','Identical inputs, outputs and accepted work\n32 replays/run; restore, CPU planning and rendering excluded'),
        (axes[1],[r for r in summary if r['family']=='fig9'],'mean','Figure 9 sustained simulation','Adapted modes change the final trajectory\n120 timed steps/run; setup and rendering excluded')]:
        by_mode={r['mode']:r for r in records}
        vals=[by_mode[m][value] for m in modes]
        ax.barh(labels,vals,color=colours,height=.62)
        for i,(m,v) in enumerate(zip(modes,vals)):
            overhead=by_mode[m]['overheadPercent']
            suffix='' if m=='native' else f' ({overhead:+.1f}%)'
            ax.text(v+.25,i,f'{v:.2f} ms{suffix}',va='center',fontsize=9)
        ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.56)
        ax.set_xlabel('GPU time, ms' if value=='mean_ms' else 'Wall-clock time per step, ms')
        ax.set_title(title,fontsize=12,pad=16)
        ax.spines[['top','right']].set_visible(False)
        ax.text(0,-.27,note,transform=ax.transAxes,fontsize=9,va='top')
    fig.suptitle('Generic atlas addressing adds full-detail cost\nFull-occupancy addressing experiment · Apple M1 Max / Dawn / Metal',fontsize=14)
    plot=output.with_name('full-fidelity-timing.png')
    fig.savefig(plot,dpi=170,bbox_inches='tight');print(plot)
