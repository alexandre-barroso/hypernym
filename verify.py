"""One verification entry point with explicit coverage and nonzero failure exits."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--selection',type=Path);a=p.parse_args()
r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_*.py'],cwd=ROOT)
if r.returncode:raise SystemExit(r.returncode)
if a.results:
 x=json.loads((a.results/'verification.json').read_text());assert x['status']=='PASS' and x['evaluated_rows']==1497
 for rec in x['artifacts']:
  target=(a.results/rec['path']).resolve();assert target.is_relative_to(a.results.resolve())
  assert hashlib.sha256(target.read_bytes()).hexdigest()==rec['sha256'],rec['path']
 print('PASS authenticated scored-run files:',len(x['artifacts']),'systems:',x['systems'],'full statistics:',x['full_statistics'])
else:print('PASS synthetic/numerical tests. Test-result reproduction NOT CHECKED; follow README and supply --results.')
if a.selection:
 spec=json.loads((ROOT/'configurations.json').read_text())
 for resource in ['glove','fasttext']:
  x=json.loads((a.selection/resource/'verification.json').read_text())
  assert x['status']=='PASS' and x['training_queries']==1500
  assert x['final_selections']==spec[resource]['final_selections']
  assert x['single_prefix_dimension']==spec[resource]['single_prefix']['selected']['dimension']
  assert x['maximum_mean_difference']<=1e-12 and x['maximum_prefix_difference']<=1e-12
  assert len(x['complete_prefix_ap'])==len(spec[resource]['all_prefix_schedule'])
  print('PASS training-selection replay:',resource,'prefixes:',len(x['complete_prefix_ap']))
