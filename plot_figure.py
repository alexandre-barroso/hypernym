"""Plot only archived, audited results; no model fitting or outcome selection."""
from pathlib import Path
import argparse, hashlib, json, os
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'run/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AUTHOR=os.environ.get('MANUSCRIPT_AUTHOR','')

def save(fig,base,data,alt):
    base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',metadata={'Author':AUTHOR,'Title':base.stem,'Creator':'Matplotlib','Subject':''})
    fig.savefig(base.with_suffix('.png'),dpi=450,bbox_inches='tight',metadata={'Author':AUTHOR,'Title':base.stem})
    fig.savefig(base.with_suffix('.jpg'),dpi=450,bbox_inches='tight',pil_kwargs={'quality':96})
    base.with_suffix('.json').write_text(json.dumps({'data':data,'alt_text':alt,'author':AUTHOR},ensure_ascii=False,indent=2)+'\n')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    e={'resources':{k:{'contrasts':{'primary':v['primary']}} for k,v in json.loads((ROOT/'expected_results.json').read_text()).items()}}
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained',gridspec_kw={'width_ratios':[1.45,1]})
    data=[]
    for y,key in enumerate(['glove','fasttext']):
        c=e['resources'][key]['contrasts']['primary']; v=c['difference']['ap_capped_gold']; lo,hi=c['bonferroni_primary_97_5_interval']
        timings=json.loads((ROOT/(key+'_timing_summary.json')).read_text())
        ratio=timings[c['left']]['median_seconds']/timings[c['right']]['median_seconds']
        axs[0].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o' if y==0 else 's',color='black',capsize=4)
        axs[1].barh(y,ratio,color=['#555555','#aaaaaa'][y],edgecolor='black',height=.5)
        axs[1].text(ratio+.035,y,f'{ratio:.2f}',va='center')
        data.append({'resource':key,'delta_ap':v,'adjusted_primary_interval':[lo,hi],'prediction_time_ratio':ratio})
    axs[0].axvline(0,color='.5',linestyle='--',linewidth=.9);axs[0].set_yticks([0,1],['GloVe','fastText']);axs[0].invert_yaxis()
    axs[0].set_xlabel('Change in mean AP at depth 15');axs[0].set_title('(a) Dense recurrence minus base',fontsize=10)
    axs[1].axvline(1,color='black',linestyle='--',linewidth=.8);axs[1].set_yticks([0,1],['GloVe','fastText']);axs[1].invert_yaxis();axs[1].set_xlim(0,3)
    axs[1].set_xlabel('Prediction time / base time');axs[1].set_title('(b) Median of five runs',fontsize=10)
    for ax in axs:ax.set_ylim(1.55,-.55)
    save(fig,ROOT/'figures/recurrence_utility_cost',data,'Two adjusted primary intervals for the mean AP increment both include zero. Dense recurrence takes 1.26 times the GloVe base prediction time and 2.55 times the fastText base time. Timing ratios are descriptive medians, not confidence intervals.')

if __name__=='__main__':main()
