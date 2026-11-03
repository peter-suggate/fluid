import json,sys,bisect,collections,re
D=sys.argv[1]
def ts(s):
  m,rest=s.split(':');p=rest.split('.');return int(m)*60e6+int(p[0])*1e6+int(p[1])*1e3+(int(p[2]) if len(p)>2 else 0)
def dur(s):
  v,u=s.split();v=float(v);return v*{'µs':1,'ms':1e3,'s':1e6,'ns':1e-3}[u]
lab={};pid=None
for l in open(D+'/encoders.ndjson'):
  r=json.loads(l)
  if (r['process'] or '').startswith('node'): lab[r['encoder-id']]=r['encoder-label'].replace('Dawn_ComputePassEncoder_','')
own=[];other=[]
for l in open(D+'/gpu-intervals.ndjson'):
  r=json.loads(l);s=ts(r['start']);e=s+dur(r['duration'])
  if (r['process'] or '').startswith('node'): own.append((s,e,lab.get(r['encoder-id'],'?unlabelled')))
  else: other.append((s,e))
own.sort();other.sort()
ostarts=[o[0] for o in other]
def contended(s,e):
  i=bisect.bisect_left(ostarts,e)
  for j in range(max(0,i-50),i):
    if other[j][1]>s and other[j][0]<e: return True
  return False
names={}
for l in open(D+'/counter-info.ndjson'):
  r=json.loads(l); names[r.get('counter-id') or r.get('id')]=r.get('name')
# counters: per timestamp, mean over partitions
samp=collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(D+'/counter-value.ndjson'):
  r=json.loads(l); samp[ts(r['timestamp'])][r['counter-id']].append(float(r['value']))
stimes=sorted(samp)
labeled=[o for o in own if o[2]!='?unlabelled']
t0=labeled[0][0] if labeled else 0
anchor='Uniform mixed all-4h solid record / buildRegions'
adv=sum(1 for o in own if o[2].startswith(anchor) and o[0]>=t0)
agg=collections.defaultdict(lambda:{'t':0.0,'n':0,'c':collections.defaultdict(list)})
for s,e,L in own:
  if s<t0: continue
  key=re.sub(r' width=\d+','',L)
  a=agg[key];a['t']+=e-s;a['n']+=1
  i=bisect.bisect_left(stimes,s);j=bisect.bisect_right(stimes,e)
  if j>i and not contended(s,e):
    for t in stimes[i:j]:
      for cid,vals in samp[t].items(): a['c'][names.get(cid,cid)].append(sum(vals)/len(vals))
tot=sum(a['t'] for a in agg.values())
print('advances',adv,'labelled GPU ms/advance %.2f'%(tot/1e3/max(adv,1)),'counter ids',names)
rows=sorted(agg.items(),key=lambda x:-x[1]['t'])
out=[]
for k,a in rows:
  c={n:sum(v)/len(v) for n,v in a['c'].items()}
  out.append({'label':k,'ms_per_adv':a['t']/1e3/adv,'dispatches_per_adv':a['n']/adv,'us_per_dispatch':a['t']/a['n'],'samples':len(a['c'].get('Compute Occupancy',[])),**c})
json.dump(out,open(sys.argv[2],'w'),indent=0)
for r in out[:70]:
  print('%6.3f ms %5.1f disp %7.1f us  occ %5s alu %5s llc %5s rd %5s wr %5s n=%d  %s'%(r['ms_per_adv'],r['dispatches_per_adv'],r['us_per_dispatch'],
   *['%.1f'%r[x] if x in r else '-' for x in ['Compute Occupancy','ALU Utilization','GPU Last Level Cache Utilization','GPU Read Bandwidth','GPU Write Bandwidth']],r['samples'],r['label'][:90]))
