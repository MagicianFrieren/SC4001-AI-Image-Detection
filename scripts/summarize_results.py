"""Independently verify saved predictions and render a compact result chart."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'results'
    with (folder/'metrics.csv').open(encoding='utf-8') as handle:
        metrics = list(csv.DictReader(handle))
    with (folder/'predictions.csv').open(encoding='utf-8') as handle:
        predictions = list(csv.DictReader(handle))
    manifest = [json.loads(line) for line in (folder/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for cell in metrics:
        selected = [r for r in predictions if r['model']==cell['model'] and
                    r['condition']==cell['condition'] and r['generator'] in ('Real',cell['generator'])]
        assert len({r['source_id'] for r in selected}) == int(cell['n']) == 200
        y = np.array([int(r['label']) for r in selected])
        p = np.array([float(r['probability_fake']) for r in selected])
        threshold = float(cell['threshold'])
        for metric, actual in [('auroc',roc_auc_score(y,p)), ('average_precision',average_precision_score(y,p)),
                               ('balanced_accuracy',balanced_accuracy_score(y,p>=threshold)),
                               ('real_false_positive_rate',np.mean(p[y==0]>=threshold))]:
            assert np.isclose(actual,float(cell[metric]),atol=1e-12)
    audit = {}
    for gen in ['Real','SD15','BigGAN','ADM']:
        rows = [r for r in manifest if r['generator']==gen]
        audit[gen] = {'formats':dict(Counter(r['source_format'] for r in rows)),
                      'dimensions':dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                      'count':len(rows)}
    verification = {'status':'passed','verified_metric_cells':len(metrics),
                    'selected_images':len(manifest),'source_audit':audit,
                    'checks':['Per-image prediction uniqueness and expected sample count',
                              'AUROC/AP/balanced accuracy/false-positive rate recomputed from saved predictions']}
    (folder/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax, generator in zip(axes,['SD15','BigGAN','ADM']):
        for model, label, color, marker in [('clip_clean','Clean training','#333333','o'),
                                            ('clip_augmented','Augmented training','#888888','s')]:
            rows = [next(r for r in metrics if r['model']==model and r['generator']==generator and r['condition']==c)
                    for c in ['clean','jpeg50','resize05']]
            ax.plot(range(3),[float(r['auroc']) for r in rows],label=label,color=color,marker=marker)
        ax.axhline(0.5,color='#bbbbbb',linestyle=':',linewidth=1)
        ax.set_title(generator+(' (seen)' if generator=='SD15' else ' (unseen)'))
        ax.set_xticks(range(3),['Clean','JPEG 50','Resize 0.5x'])
        ax.set_ylim(0,1.02)
        ax.grid(axis='y',alpha=0.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('AUROC')
    axes[0].legend(loc='lower left',frameon=False,fontsize=8)
    fig.suptitle('Cross-generator AI image detection: development results')
    fig.text(0.5,0.015,'Exploratory subset, one training seed; shared real pool. Dotted line: chance ranking.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,0.04,1,0.95])
    fig.savefig(folder/'pilot_auroc.png',dpi=180)
    print(json.dumps({k:v for k,v in verification.items() if k!='source_audit'},indent=2))


if __name__=='__main__':
    main()
