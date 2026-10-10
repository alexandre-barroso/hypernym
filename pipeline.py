"""Portable reproduction with fixed training-selected configurations.

Original core implementations are imported unchanged. No pickle, foreign model
code, dataset extraction or implicit download. Test references are read only by
score, after prediction files exist. This is a reproduction, not a new preregistration.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,os,time,zipfile
import numpy as np
from label_transfer import Query,RidgeMoments,lexical_folds
from label_transfer_final import QueryInput,FEATURES,AUX,prepare,build_features,context_for
from label_transfer_r4 import ARMS,LAMBDAS,RIDGE_NAMES,responses
from predict_label_transfer_final import restore_model,system_inventory,predict
from score_label_transfer_final import score_outputs,final_statistics,METRICS
from vector_io import parse_vector,parse_header,parse_row,merge_moments,eigendecomposition
ROOT=Path(__file__).resolve().parent
SPEC=json.loads((ROOT/'configurations.json').read_text())
INPUTS=json.loads((ROOT/'inputs.json').read_text())
PREFIX='SemEval2018-Task9/'

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def authenticate(p,name):
 if digest(p)!=INPUTS[name]['sha256']:raise ValueError('Pinned input changed: '+name)
def dataset(path,stage):
 authenticate(path,'dataset')
 with zipfile.ZipFile(path) as z:
  tr=z.read(PREFIX+'training/data/1A.english.training.data.txt').decode().splitlines()
  gold=z.read(PREFIX+'training/gold/1A.english.training.gold.txt').decode().splitlines()
  vocabulary=set(z.read(PREFIX+'vocabulary/1A.english.vocabulary.txt').decode().splitlines())
  train=[Query(f'training:{i}',s.split('\t')[0].lower(),s.split('\t')[1],tuple(g.split('\t'))) for i,(s,g) in enumerate(zip(tr,gold),1)]
  assert len(train)==len(tr)==len(gold)==1500
  if stage=='train':return train,vocabulary
  lines=z.read(PREFIX+'test/data/1A.english.test.data.txt').decode().splitlines();seen={q.word for q in train}
  queries=[];official=[]
  for i,s in enumerate(lines,1):
   word,category=s.split('\t')
   if word.lower() not in seen:queries.append(QueryInput(f'test:{i}',word.lower(),category));official.append(i)
  assert len(queries)==1497 and set(range(1,1501))-set(official)=={185,281,1390}
  if stage=='queries':return train,vocabulary,queries,official
  assert stage=='gold'
  raw=z.read(PREFIX+'test/gold/1A.english.test.gold.txt').decode().splitlines()
  gold=[tuple(raw[i-1].strip().split('\t')) for i in official]
  assert all(g and len(set(g))==len(g) for g in gold)
  return train,vocabulary,queries,official,gold

def vectors(a):
 name=a.resource;authenticate(a.vectors_zip,name)
 train,vocab,queries,official=dataset(a.dataset_zip,'queries');records=train+queries
 wanted={q.word for q in records};dimension=SPEC[name]['dimension'];selected={};seen=set();block=[];n=0;mean=np.zeros(dimension);m2=np.zeros((dimension,dimension))
 out=a.run/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'vectors.npz').exists():raise FileExistsError('Existing vector cache preserved')
 with zipfile.ZipFile(a.vectors_zip) as z,z.open(INPUTS[name]['member']) as f:
  declared=None
  if name=='fasttext':declared,width=parse_header(f.readline());assert width==dimension and declared==999994
  for raw in f:
   word,v=parse_row(raw,dimension) if name=='fasttext' else parse_vector(raw.decode('utf-8'),dimension)
   if word in seen:raise ValueError('Duplicate released lexical row')
   seen.add(word)
   if word in ('<unk>','</s>'):continue
   block.append(v)
   if word in wanted:selected[word]=v
   if len(block)==4096:n,mean,m2=merge_moments(n,mean,m2,np.asarray(block));block=[]
  if block:n,mean,m2=merge_moments(n,mean,m2,np.asarray(block))
 if declared is not None:assert len(seen)==declared
 eigenvalues,components,checks=eigendecomposition(m2/(n-1))
 available=np.array([q.word in selected for q in records]);matrix=np.zeros((len(records),dimension))
 for i,q in enumerate(records):
  if available[i]:matrix[i]=selected[q.word]
 np.savez_compressed(out/'vectors.npz',matrix=matrix,available=available,mean=mean,components=components,eigenvalues=eigenvalues)
 save(out/'vectors.json',{'resource':name,'archive_sha256':INPUTS[name]['sha256'],'pca_rows':n,'actual_rows':len(seen),'records':[{'identifier':q.identifier,'word':q.word,'category':q.category} for q in records],'official_test_rows':official,'numerical_checks':checks})
 print(name,'PCA rows',n,'train/query coverage',int(available[:1500].sum()),int(available[1500:].sum()),flush=True)

def cache(a):
 p=a.run/a.resource
 with np.load(p/'vectors.npz',allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 train,vocab,queries,official=dataset(a.dataset_zip,'queries');records=train+queries
 assert read(p/'vectors.json')['records']==[{'identifier':q.identifier,'word':q.word,'category':q.category} for q in records]
 return p,arr,train,vocab,queries,records

def model_record(m):
 return {'features':[FEATURES[i] for i in m.feature_indices],'feature_indices':list(m.feature_indices),'mean':m.mean.tolist(),'scale':m.scale.tolist(),'coefficient':m.coefficient.tolist(),'intercept':m.intercept,'regularization':m.regularization,'fit_queries':m.fit_queries,'condition_number':m.condition_number}

def fit(a):
 p,z,train,vocab,queries,records=cache(a);spec=SPEC[a.resource]
 if (p/'models.json').exists():raise FileExistsError('Existing models preserved')
 views=prepare(z['matrix'][:1500],z['available'][:1500],z['mean'],z['components'],spec['all_prefix_schedule'],spec['component_removal']['removed'])
 assignment=lexical_folds(train,range(1500),folds=5,seed='label-transfer-r3-20260926');moments=RidgeMoments(len(FEATURES))
 for fold in range(5):
  refs=[i for i in range(1500) if assignment[i]!=fold];qs=[i for i in range(1500) if assignment[i]==fold];ctx=context_for(train,refs,vocab)
  for start in range(0,len(qs),128):
   b=build_features(views,train,qs[start:start+128],ctx,requested=tuple(AUX),single=spec['single_prefix']['selected']['dimension'],variance=spec['variance_schedule']['dimensions'])
   moments.add(b.features,responses(train,b,ctx),b.allowed,b.eligible)
  print(a.resource,'fit fold',fold,flush=True)
 models={}
 for arm,cols in ARMS.items():
  for name,lam in zip(RIDGE_NAMES[arm],LAMBDAS):models[name]=model_record(moments.fit(feature_indices=cols,regularization=lam))
 for aux in ['single','residual','variance','matched_count']:models['ridge_control_'+aux]=model_record(moments.fit(feature_indices=tuple(range(12))+(AUX[aux],),regularization=spec['fixed_feature_panel_regularization']))
 expected=read(ROOT/(a.resource+'_expected_models.json'))['models'];maximum=0
 for name,m in models.items():
  assert m['feature_indices']==expected[name]['feature_indices'] and m['fit_queries']==expected[name]['fit_queries']
  for field in ['mean','scale','coefficient','intercept']:
   error=float(np.max(np.abs(np.asarray(m[field])-expected[name][field])));maximum=max(maximum,error)
   if not np.allclose(m[field],expected[name][field],rtol=1e-8,atol=1e-10):raise ValueError(('Model differs',name,field,error))
 save(p/'models.json',{'models':models,'maximum_difference_from_original':maximum});print('PASS fitted models',a.resource,maximum,flush=True)

def prediction(a):
 p,z,train,vocab,queries,records=cache(a);spec=SPEC[a.resource]
 models={k:restore_model(v) for k,v in read(p/'models.json')['models'].items()};systems,aliases=system_inventory(spec,models)
 if a.selected_only:systems=[s for s in systems if s.name in {spec['final_selections'][x] for x in ['B0_family','B1_all_family','B1_coarse_family']}]
 out=p/'predictions';out.mkdir(exist_ok=False);rng=np.random.Generator(np.random.PCG64(2026092606));observations=[];first={};diags={};candidates=None
 for repeat in range(a.repeats+1):
  for i in rng.permutation(len(systems)):
   s=systems[i];start=time.perf_counter();pred,d,candidates=predict(s,z['matrix'],z['available'],z['mean'],z['components'],records,range(1500),range(1500,len(records)),vocab,models,spec,batch_size=128);elapsed=time.perf_counter()-start
   h=hashlib.sha256(np.asarray(pred,dtype='<i4').tobytes()).hexdigest()
   if repeat==0:np.save(out/(s.name+'.npy'),pred,allow_pickle=False);first[s.name]=h;diags[s.name]=d
   else:assert first[s.name]==h and d==diags[s.name]
   observations.append({'system':s.name,'repeat':repeat,'warmup':repeat==0,'seconds':elapsed});print(a.resource,repeat,s.name,round(elapsed,3),flush=True)
 ctx=context_for(train,range(1500),vocab)
 save(out/'manifest.json',{'systems':[s.name for s in systems],'aliases':aliases,'candidates':list(candidates),'counts':ctx.counts.tolist(),'diagnostics':diags,'prediction_hashes':first,'timing':observations,'selected_only':a.selected_only,'timing_scope':'Resident raw inputs through ranked output; excludes file I/O, model fitting and gold scoring.'})

def score(a):
 # This is the only operation that opens the test reference member.
 for name in SPEC:assert (a.run/name/'predictions/manifest.json').exists()
 train,vocab,queries,official,gold=dataset(a.dataset_zip,'gold');expected=read(ROOT/'expected_results.json');data={}
 for name in SPEC:
  p=a.run/name/'predictions';m=read(p/'manifest.json');pred=np.stack([np.load(p/(s+'.npy'),allow_pickle=False) for s in m['systems']]);scores,relevance=score_outputs(pred,m['candidates'],gold)
  for i,s in enumerate(m['systems']):
   got=dict(zip(METRICS,scores[i].mean(axis=0).tolist()));want=expected[name]['all_system_metrics'][s]
   for metric in METRICS:assert np.isclose(got[metric],want[metric],rtol=1e-10,atol=1e-12),(name,s,metric,got[metric],want[metric])
  data[name]={'systems':m['systems'],'aliases':m['aliases'],'predictions':pred,'scores':scores,'candidates':m['candidates'],'counts':m['counts'],'usable':next(iter(m['diagnostics'].values()))['usable']}
  print('PASS',name,len(m['systems']),'systems; every mean metric matches original',flush=True)
 if all(not read(a.run/n/'predictions/manifest.json')['selected_only'] for n in SPEC):
  result=final_statistics(data,[q.word for q in queries],[q.category for q in queries],gold,SPEC,vocab,replicates=100000,seed=2026092605)
  for n in SPEC:assert np.allclose(result['resources'][n]['contrasts']['primary']['bonferroni_primary_97_5_interval'],expected[n]['primary']['bonferroni_primary_97_5_interval'],rtol=1e-10,atol=1e-12)
  save(a.run/'statistics.json',result)
 artifacts=[]
 for name in SPEC:
  for f in sorted((a.run/name).rglob('*')):
   if f.is_file():artifacts.append({'path':str(f.relative_to(a.run)),'sha256':digest(f)})
 if (a.run/'statistics.json').exists():artifacts.append({'path':'statistics.json','sha256':digest(a.run/'statistics.json')})
 save(a.run/'verification.json',{'artifacts':artifacts,'status':'PASS','evaluated_rows':1497,'systems':{n:len(d['systems']) for n,d in data.items()},'full_statistics':(a.run/'statistics.json').exists(),'scope':'Recomputed supplied-reference metrics; no novelty, human-meaning or hardware-time guarantee.'})

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['vectors','fit','predict','score']);p.add_argument('--run',required=True,type=Path);p.add_argument('--dataset-zip',required=True,type=Path);p.add_argument('--resource',choices=list(SPEC));p.add_argument('--vectors-zip',type=Path);p.add_argument('--selected-only',action='store_true');p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
 if a.stage!='score' and not a.resource:p.error('--resource required')
 if a.stage=='vectors' and not a.vectors_zip:p.error('--vectors-zip required')
 if a.repeats<1:p.error('--repeats must be positive; repeat zero is warm-up')
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
  if os.environ.get(key)!='1':p.error('Launch with '+key+'=1')
 {'vectors':vectors,'fit':fit,'predict':prediction,'score':score}[a.stage](a)
if __name__=='__main__':main()
