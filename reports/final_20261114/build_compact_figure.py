"""Replot existing JPEG70 transfer summaries for the manuscript; no new analysis."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd

script_dir = Path(__file__).resolve().parent
package_dir = script_dir.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path,
                    default=package_dir / 'evidence/figures/figure2_transfer_data.csv',
                    help='Existing four-row transfer summary (default: package evidence/figures)')
parser.add_argument('--output-dir', type=Path, default=script_dir,
                    help='Directory for PNG, PDF, caption and receipt (default: package report/)')
args = parser.parse_args()
source = args.source.resolve()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
data = pd.read_csv(source)
assert len(data) == 4 and set(data.condition) == {'jpeg70'}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
fig, axes = plt.subplots(1,2,figsize=(6.6,2.3))
fig.subplots_adjust(left=.105,right=.982,bottom=.21,top=.735,wspace=.43)
handles = []
for donor, target, colour, marker, line in [
    ('ADM','BigGAN','0.08','o','-'),('BigGAN','ADM','0.48','s','--')]:
    frame = data[data.donor.eq(donor)].set_index('domain').loc[['donor_tuning','heldout_development']]
    for ax, column in zip(axes,['global_minus_local_auroc','quality_gate_alpha_global']):
        values = frame[column].to_numpy()
        ax.plot([0,1],values,color=colour,marker=marker,linestyle=line,linewidth=1.15,markersize=4.6)
        for x,value in enumerate(values):
            label = f'{value:+.3f}' if ax is axes[0] else f'{value:.1%}'
            dy = (6 if value >= 0 else -13) if ax is axes[0] else (6 if donor == 'ADM' else -13)
            ax.annotate(label,(x,value),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=8.5)
    handles.append(Line2D([],[],color=colour,marker=marker,linestyle=line,linewidth=1.15,markersize=4.6,
                          label=f'{donor} source → {target} target'))
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.0),frameon=False,
           ncol=2,fontsize=8.5,handlelength=1.8,columnspacing=1.6)
for ax in axes:
    ax.set_xticks([0,1],['Source selection','Held-out target'])
    ax.set_xlim(-.25,1.25)
    ax.tick_params(labelsize=8.5,pad=3)
    ax.grid(axis='y',color='.9',linewidth=.5)
    ax.set_axisbelow(True)
axes[0].set_title('A  Expert advantage (JPEG70)',loc='left',fontsize=9.5,weight='bold',pad=8)
axes[0].set_ylabel('Global − local AUROC',fontsize=9)
axes[0].set_ylim(-.11,.11)
axes[0].set_yticks([-.1,-.05,0,.05,.1])
axes[0].axhline(0,color='.3',linestyle=':',linewidth=.8)
axes[1].set_title('B  Gate preference (JPEG70)',loc='left',fontsize=9.5,weight='bold',pad=8)
axes[1].set_ylabel('Mean global weight',fontsize=9)
axes[1].set_ylim(.22,.87)
axes[1].set_yticks([.3,.5,.7])
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].axhline(.5,color='.3',linestyle=':',linewidth=.8)
for suffix in ['png','pdf']:
    fig.savefig(out/f'figure4_transfer_mismatch.{suffix}',dpi=300,facecolor='white')
plt.close(fig)
caption = ('Figure 4. JPEG70 development diagnostics. (A) The stronger expert changes between donor-only '
           'source selection and the held-out generator. (B) The frozen quality gate retains a mean preference '
           'on the same side of equal weighting. Source cells contain 124 real and 62 donor fake images; '
           'target cells contain 300 real and 300 fake images, so mean weights also reflect different class '
           'composition. Values average three gate initializations and three shared-pool data seeds. '
           'These aggregate preferences describe the observed transfer mismatch, not an optimal weight for every image.\n')
(out/'figure4_transfer_mismatch_caption.txt').write_text(caption,encoding='utf-8')
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
receipt = {'operation':'replot existing four-row transfer summary; no new metrics',
           'source':str(source.relative_to(package_dir)).replace('\\','/') if source.is_relative_to(package_dir) else str(source),
           'source_sha256':source_hash,
           'figure_inches':[6.6,2.3],'source_unchanged':True,
           'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('figure4_transfer_mismatch.*')},
           'visual_qa':'pending separate inspection'}
(out/'compact_figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':str(out),'size_inches':[6.6,2.3],'source_unchanged':True}))
