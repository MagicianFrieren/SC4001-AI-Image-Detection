"""Build the two-page Chinese discussion report from measured pilot outputs.

Use the bundled document Python runtime (reportlab, pdfplumber, pypdfium2).
"""
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
OUT = ROOT/'output'/'pdf'/'sc4001_j_pilot_report_zh.pdf'
TMP = ROOT/'tmp'/'pdfs'/'pilot-report'
OUT.parent.mkdir(parents=True,exist_ok=True)
TMP.mkdir(parents=True,exist_ok=True)
with (ROOT/'results'/'metrics.csv').open(encoding='utf-8') as handle:
    METRICS = list(csv.DictReader(handle))
META = json.loads((ROOT/'results'/'run_metadata.json').read_text(encoding='utf-8'))
DATA = json.loads((ROOT/'results'/'data_preparation.json').read_text(encoding='utf-8'))
VERIFY = json.loads((ROOT/'results'/'verification.json').read_text(encoding='utf-8'))
assert VERIFY['status']=='passed' and VERIFY['verified_metric_cells']==36

pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('HeiB','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('Song',normal='Song',bold='HeiB',italic='Song',boldItalic='HeiB')
W,H = A4
M = 44
CW = W-2*M
INK = HexColor('#202020')
GRAY = HexColor('#555555')
STYLES = {
    'title':ParagraphStyle('title',fontName='HeiB',fontSize=21,leading=30,textColor=INK,wordWrap='CJK'),
    'h2':ParagraphStyle('h2',fontName='HeiB',fontSize=12,leading=19,textColor=INK,wordWrap='CJK'),
    'body':ParagraphStyle('body',fontName='Song',fontSize=10.5,leading=17,textColor=INK,wordWrap='CJK'),
    'small':ParagraphStyle('small',fontName='Song',fontSize=9,leading=14,textColor=GRAY,wordWrap='CJK'),
    'cell':ParagraphStyle('cell',fontName='Song',fontSize=9.5,leading=14,textColor=INK,wordWrap='CJK'),
    'th':ParagraphStyle('th',fontName='HeiB',fontSize=9,leading=14,textColor=INK,wordWrap='CJK'),
}
c = canvas.Canvas(str(OUT),pagesize=A4)
c.setTitle('SC4001 J方向初步可行性测试报告')
c.setAuthor('SC4001 选题讨论材料 / Codex辅助整理')
c.setSubject('已完成的本机初测、跨生成器结果、数据偏差与下一步建议')


def p(text,y,style='body',gap=9,x=M,width=CW):
    obj = Paragraph(text,STYLES[style])
    _,height = obj.wrap(width,H)
    assert y-height>=54, f'Page {c.getPageNumber()} overflow: {text[:35]}'
    obj.drawOn(c,x,y-height)
    return y-height-gap


def heading(text,y):
    return p(text,y,'h2',gap=6)


def begin(title,subtitle):
    c.setFillColor(GRAY)
    c.setFont('Hei',8.5)
    c.drawString(M,H-34,'SC4001 · J方向 · 小组选题讨论')
    y = p(title,H-58,'title',gap=9)
    return p(subtitle,y,'small',gap=17)


def finish():
    c.setFillColor(GRAY)
    c.setFont('Song',8)
    c.drawString(M,30,'初测 v0.1 · 探索性开发结果 · 非最终课程报告')
    c.drawRightString(W-M,30,f'{c.getPageNumber()} / 2')
    c.showPage()


def table(headers,rows,widths,y):
    content = [[Paragraph(escape(str(value)),STYLES['th']) for value in headers]]
    content += [[Paragraph(escape(str(value)),STYLES['cell']) for value in row] for row in rows]
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
    return y-height-11


def metric(model,generator,condition):
    return next(row for row in METRICS if row['model']==model and row['generator']==generator and row['condition']==condition)


y = begin('J方向初步可行性测试','任务：判断图片是真实还是AI生成，并检查训练中未见生成器上的表现。')
y = heading('目前能得出的判断',y)
y = p('<b>本机可以低成本跑通；跨生成器泛化仍是主要难点。</b>普通基线在SD1.5上表现较好，换成BigGAN、ADM后接近随机水平。简单增强没有带来一致改善。J值得进一步探索，但这轮结果还不能支持质量门控方案有效。',y)
y = heading('这次具体测了什么',y)
y = p('从第三方Tiny-GenImage的4个固定分片中取样，共1,200张。冻结CLIP ViT-B/32，提取512维特征，仅训练线性分类头。数据划分如下；BigGAN、ADM不参与训练或阈值选择。',y)
y = table(['分区','真实图','SD1.5','BigGAN','ADM'],[
    ['训练',300,300,0,0],['验证',100,100,0,0],['开发检查',100,100,100,100],
],[CW-4*82,82,82,82,82],y)
y = p('每个开发比较为100张真实图＋100张对应生成图，三个比较共享真实池。先统一RGB和224×224裁剪，再施加变换；普通训练只用原图，增强训练使用原图、JPEG Q70和0.5倍缩放后还原三种视图。',y,'small')
y = heading('实测结果：AUROC',y)
rows = []
for model,label in [('clip_clean','普通训练'),('clip_augmented','增强训练')]:
    for generator,name in [('SD15','SD1.5（已见）'),('BigGAN','BigGAN（未见）'),('ADM','ADM（未见）')]:
        rows.append([label,name]+[f"{float(metric(model,generator,condition)['auroc']):.4f}" for condition in ['clean','jpeg50','resize05']])
y = table(['训练方式','开发检查生成器','原图','JPEG Q50','0.5倍缩放'],rows,[83,133,90,90,CW-396],y)
y = p('AUROC衡量真假得分的区分排序，越高越好，0.5为随机水平；它不是准确率。低于0.5表示在当前得分方向上出现反向排序，不能看完开发标签后翻转分数，再当作泛化成功。',y,'small')
intervals = []
for gen in ['SD15','BigGAN','ADM']:
    row = metric('clip_clean',gen,'clean')
    intervals.append(f"{gen} {float(row['auc_ci_low']):.3f}–{float(row['auc_ci_high']):.3f}")
y = p('普通训练／原图的95%自助法区间：'+ '；'.join(intervals)+'。区间基于每格500次分层重采样，只描述本子集的抽样不确定性。',y,'small')
y = heading('本机成本与核对',y)
y = p(f"RTX 4050（6GB）；正式特征提取与训练评估共<b>{META['run_seconds']:.1f}秒</b>，其中特征提取{META['feature_extraction_seconds']:.1f}秒；不含下载、环境配置与数据准备。36组指标均已从逐图预测文件独立重算，结果一致。",y)
finish()

y = begin('结果怎么理解，下一步讨论什么','这轮验证了实施可行性，并暴露了泛化与数据控制问题。')
chart_height = CW*684/1980
c.drawImage(str(ROOT/'results'/'pilot_auroc.png'),M,y-chart_height,width=CW,height=chart_height)
y -= chart_height+12
y = heading('数据偏差：不能把高分直接当成生成痕迹',y)
y = p('仅用原始宽高训练分类器，在SD1.5原图上就得到<b>0.9100 AUROC</b>。所选真实图全部源自JPEG、尺寸多样；SD1.5全部为512×512 PNG，BigGAN为128×128 PNG，ADM为256×256 PNG。',y)
y = p('CLIP分支没有接收原始尺寸或文件名；上述结果提示数据混杂，不能据此认定CLIP具体用了哪种线索。统一输入尺寸、转存PNG，也不能消除原来的压缩、分辨率与重采样历史。',y,'small')
y = heading('本轮结论的边界',y)
y = p('样本来自固定4个第三方分片，仅一个训练种子；生成器标签未逐张独立认证。划分前剔除了2张近重复候选，但不能保证内容独立。三组比较共享真实图，不能当成完全独立的三次实验。',y)
y = p('BigGAN、ADM结果已被查看，后续只能作为开发参考；最终实验需另留未用于选择方法的生成器／样本。此次未训练质量门控，也不是UnivFD论文ViT-L/14设置的严格复现。',y,'small')
y = heading('建议下一轮的小范围验证',y)
y = p('<b>1. 先控制数据处理差异。</b>对真假图对称施加统一的缩放与重编码规则，再检查尺寸／质量诊断和检测结果；保留对原有处理痕迹的限制说明。',y)
y = p('<b>2. 增加一个训练生成器。</b>用同样的CLIP与线性头，比较单生成器和双生成器训练，控制训练总量，并保留开发生成器，观察泛化掉分是否缓解。',y)
y = p('<b>3. 再判断要不要做融合。</b>先检查全局与局部分支是否互补，再决定是否投入质量感知门控。当前表格没有提供门控收益的证据。',y)
y = heading('群里建议先讨论',y)
y = p('是否继续选J，并把下一轮目标定为“先改善跨生成器泛化、控制数据偏差”？如果继续，可先分工做数据核查和双生成器基线，暂缓扩展模型。以上是建议，尚未执行。',y)
y = p('证据：results中的样本清单、逐图预测、指标及核对记录；实验代码版本7e425dc。<br/>来源：<link href="https://huggingface.co/datasets/TheKernel01/Tiny-GenImage" color="#555555"><u>Tiny-GenImage数据卡</u></link>；<link href="https://github.com/WisconsinAIVision/UniversalFakeDetect" color="#555555"><u>UnivFD参考实现</u></link>；<link href="https://github.com/MagicianFrieren/SC4001-AI-Image-Detection" color="#555555"><u>项目私有仓库</u></link>。代码执行与报告整理使用Codex辅助。',y,'small')
finish()
c.save()

checks = []
with pdfplumber.open(OUT) as pdf:
    assert len(pdf.pages)==2
    for i,page in enumerate(pdf.pages,1):
        text = page.extract_text() or ''
        assert len(text)>500
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
message = ('J方向先做了一轮本机小测试：1,200张图，冻结CLIP，只训练线性分类头。'
           '在训练见过的SD1.5上AUROC是0.967，但换到BigGAN、ADM后只有0.471、0.560（0.5是随机水平），简单增强也没有一致改善。'
           '模型运行约64秒，不含数据准备，算力上可以做。主要问题是跨生成器泛化，以及数据里的格式／分辨率偏差：仅用原始尺寸在SD1.5上就能到0.91。'
           '建议先讨论要不要继续J；如果继续，下一轮先控制数据偏差、比较单／双生成器训练，再决定是否做质量门控。详细设置和结果在附带的两页报告里。')
(ROOT/'output'/'sc4001_j_group_message_zh.txt').write_text(message+'\n',encoding='utf-8')
print(json.dumps({'output':str(OUT),'bytes':OUT.stat().st_size,'checks':checks},ensure_ascii=False))
