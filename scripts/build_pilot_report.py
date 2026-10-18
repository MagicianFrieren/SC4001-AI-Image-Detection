"""Build a reader-facing English report from the recorded experimental outputs."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pdfplumber
import pypdfium2 as pdfium
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'/'pdf'/'sc4001_j_preliminary_report_en.pdf'
TMP = ROOT/'tmp'/'pdfs'/'preliminary-report-en'
OUT.parent.mkdir(parents=True,exist_ok=True)
TMP.mkdir(parents=True,exist_ok=True)
with (ROOT/'results'/'metrics.csv').open(encoding='utf-8') as handle:
    METRICS = list(csv.DictReader(handle))
VERIFY = json.loads((ROOT/'results'/'verification.json').read_text(encoding='utf-8'))
assert VERIFY['status']=='passed' and VERIFY['verified_metric_cells']==36

for name,path in [('Body','times.ttf'),('BodyB','timesbd.ttf'),('Head','arial.ttf'),('HeadB','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+path))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyB',italic='Body',boldItalic='BodyB')
W,H = A4
M = 44
CW = W-2*M
INK,GRAY = HexColor('#202020'),HexColor('#555555')
STYLES = {
    'title':ParagraphStyle('title',fontName='HeadB',fontSize=20,leading=26,textColor=INK),
    'h2':ParagraphStyle('h2',fontName='HeadB',fontSize=11.5,leading=17,textColor=INK),
    'body':ParagraphStyle('body',fontName='Body',fontSize=10.5,leading=14.8,textColor=INK),
    'small':ParagraphStyle('small',fontName='Body',fontSize=9,leading=12.5,textColor=GRAY),
    'cell':ParagraphStyle('cell',fontName='Body',fontSize=10,leading=14,textColor=INK),
    'th':ParagraphStyle('th',fontName='HeadB',fontSize=9,leading=13,textColor=INK),
}
c = canvas.Canvas(str(OUT),pagesize=A4)
c.setTitle('Cross-Generator AI Image Detection: Preliminary Experimental Report')
c.setAuthor('SC4001 Project')
c.setSubject('Frozen CLIP baseline, cross-generator evaluation, augmentation, and dataset diagnostics')


def p(text,y,style='body',gap=8):
    obj = Paragraph(text,STYLES[style])
    _,height = obj.wrap(CW,H)
    assert y-height>=54, f'Page {c.getPageNumber()} overflow: {text[:60]}'
    obj.drawOn(c,M,y-height)
    return y-height-gap


def heading(text,y):
    return p(text,y,'h2',gap=5)


def begin(title,subtitle):
    c.setFillColor(GRAY)
    c.setFont('Head',8.5)
    c.drawString(M,H-33,'SC4001 | PRELIMINARY EXPERIMENTAL REPORT')
    y = p(title,H-56,'title',gap=7)
    return p(subtitle,y,'small',gap=15)


def finish():
    c.setFillColor(GRAY)
    c.setFont('Body',8)
    c.drawString(M,30,'Preliminary study | Exploratory development results | v0.2')
    c.drawRightString(W-M,30,f'{c.getPageNumber()} / 2')
    c.showPage()


def table(headers,rows,widths,y):
    content = [[Paragraph(escape(str(v)),STYLES['th']) for v in headers]]
    content += [[Paragraph(escape(str(v)),STYLES['cell']) for v in row] for row in rows]
    obj = Table(content,colWidths=widths,hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('LINEBELOW',(0,0),(-1,0),0.55,HexColor('#AAAAAA')),
        ('LINEBELOW',(0,1),(-1,-1),0.25,HexColor('#CCCCCC')),
    ]))
    _,height = obj.wrap(CW,H)
    assert y-height>=54, 'Table overflow'
    obj.drawOn(c,M,y-height)
    return y-height-10


def metric(model,generator,condition):
    return next(r for r in METRICS if r['model']==model and r['generator']==generator and r['condition']==condition)


y = begin('Cross-Generator AI Image Detection','A preliminary study of frozen CLIP features and image-transformation robustness')
y = heading('1. Objective',y)
y = p('Evaluate whether a detector trained on one image generator transfers to other generators, and whether compression and resizing augmentation improves robustness. The baseline uses a frozen CLIP ViT-B/32 encoder and a logistic regression classifier.',y)
y = heading('2. Experimental setup',y)
y = p('A 1,200-image subset was sampled from four fixed Tiny-GenImage shards. The classifier was trained using real images and Stable Diffusion 1.5 (SD1.5) images only. BigGAN and ADM images were excluded from training and validation.',y)
y = table(['Partition','Real','SD1.5','BigGAN','ADM'],[
    ['Training',300,300,0,0],['Validation',100,100,0,0],['Development',100,100,100,100],
],[CW-4*80,80,80,80,80],y)
y = p('Each development comparison contains 100 real and 100 generated images; the same real pool is shared across generators. Images are converted to RGB and centre-cropped/resized to 224 × 224 pixels before any evaluation transformation.',y)
y = p('<b>Clean training:</b> original processed images. <b>Augmented training:</b> the same originals under clean, JPEG Q70, and 0.5× downsampling followed by restoration. Evaluation uses clean images, JPEG Q50, and 224→112→224 bicubic resizing. The encoder is frozen; 512-dimensional features feed the classifier (C = 1). Thresholds for secondary classification metrics are selected on clean validation images and then fixed.',y,'small')
y = heading('3. Results',y)
rows = []
for model,label in [('clip_clean','Clean'),('clip_augmented','Augmented')]:
    for generator,name in [('SD15','SD1.5 (seen)'),('BigGAN','BigGAN (unseen)'),('ADM','ADM (unseen)')]:
        rows.append([label,name]+[f"{float(metric(model,generator,c)['auroc']):.4f}" for c in ['clean','jpeg50','resize05']])
y = table(['Training','Generator','Clean','JPEG Q50','Resize 0.5×'],rows,[82,143,90,90,CW-405],y)
y = p('Values are AUROC: higher is better, and 0.5 represents chance ranking. AUROC is not classification accuracy. “Unseen” means absent from classifier training and validation. Both unseen generators are now development data, rather than reserved final tests.',y,'small')
intervals = []
for gen,label in [('SD15','SD1.5'),('BigGAN','BigGAN'),('ADM','ADM')]:
    r = metric('clip_clean',gen,'clean')
    intervals.append(f"{label}: {float(r['auc_ci_low']):.3f}–{float(r['auc_ci_high']):.3f}")
y = p('Clean-trained model, clean images: 95% bootstrap intervals are '+ '; '.join(intervals)+'. Each interval uses 500 class-stratified resamples and reflects sampling uncertainty within this subset.',y,'small')
y = p('<b>Finding:</b> strong discrimination on SD1.5 did not transfer to BigGAN or ADM. Augmentation slightly increased SD1.5 JPEG50 AUROC (0.8803 to 0.8899), but produced no consistent improvement on unseen generators.',y)
finish()

y = begin('Interpretation and Further Experiments','Dataset diagnostics, limitations, and proposed follow-up comparisons')
chart_height = CW*684/1980
c.drawImage(str(ROOT/'results'/'pilot_auroc.png'),M,y-chart_height,width=CW,height=chart_height)
y -= chart_height+12
y = heading('4. Dataset diagnostics',y)
y = p('A classifier using only original image width and height reached <b>0.9100 AUROC</b> on SD1.5. All sampled real images originated as JPEG files with varied dimensions. Generated sources were PNG: SD1.5 at 512 × 512, BigGAN at 128 × 128, and ADM at 256 × 256.',y)
y = p('These differences indicate dataset confounds. The CLIP classifier receives neither original dimensions nor file names, so the diagnostic does not establish which cues it used. Equal input dimensions and PNG conversion cannot erase earlier compression or resampling history.',y)
y = heading('5. Limitations',y)
y = p('The study uses a small third-party subset, four fixed shards, and one training seed. Generator labels are inherited and have not been independently authenticated. Hash and near-duplicate checks removed two candidates before splitting, but do not establish semantic independence. Shared real images also make generator comparisons dependent.',y)
y = p('Below-chance AUROC indicates reversed ranking under the fixed score convention; reversing scores after inspecting development labels would not constitute a valid generalization result. Quality-aware gating has not been evaluated. This ViT-B/32 protocol is not a reproduction of the UnivFD ViT-L/14 experiments.',y,'small')
y = heading('6. Proposed follow-up experiments',y)
y = p('<b>Control preprocessing differences.</b> Apply common resizing and re-encoding procedures symmetrically to real and generated images. Reassess both detection performance and quality/size diagnostics, while accounting for residual source-history differences.',y)
y = p('<b>Compare training-generator coverage.</b> Compare one-generator and two-generator training with the same encoder, classifier, and total training-image budget. Evaluate on development generators excluded from classifier training.',y)
y = p('<b>Test expert complementarity before fusion.</b> Establish whether global and local branches make different errors before introducing quality-aware gating. Reserve separate generators or samples for final evaluation before further model selection.',y)
y = p('Evidence: recorded sample manifest, predictions, metrics, and verification outputs; experiment revision 7e425dc. All 36 metric cells were independently recomputed from saved predictions.<br/>Sources: <link href="https://huggingface.co/datasets/TheKernel01/Tiny-GenImage" color="#555555"><u>Tiny-GenImage dataset card</u></link>; <link href="https://github.com/WisconsinAIVision/UniversalFakeDetect" color="#555555"><u>UnivFD reference implementation</u></link>; <link href="https://github.com/MagicianFrieren/SC4001-AI-Image-Detection" color="#555555"><u>project repository</u></link>.',y,'small')
finish()
c.save()

checks = []
with pdfplumber.open(OUT) as pdf:
    assert len(pdf.pages)==2
    for i,page in enumerate(pdf.pages,1):
        text = page.extract_text() or ''
        assert len(text)>500
        assert not any(term in text.lower() for term in ['laptop','local machine','group chat','codex'])
        for obj in page.chars+page.rects+page.lines:
            for key in ('non_stroking_color','stroking_color'):
                color = obj.get(key)
                if isinstance(color,(list,tuple)) and len(color)==3:
                    assert max(color)-min(color)<0.0001
        checks.append({'page':i,'characters':len(text),'links':len(page.hyperlinks)})
document = pdfium.PdfDocument(str(OUT))
for index in range(len(document)):
    document[index].render(scale=1.6).to_pil().save(TMP/f'page-{index+1}.png')
(TMP/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'bytes':OUT.stat().st_size,'checks':checks}))
