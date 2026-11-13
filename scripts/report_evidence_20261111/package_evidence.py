"""Collect exact evidence inputs and code, verify files, and build a local ZIP."""
from __future__ import annotations
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/analysis/report_evidence_20261111"
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def save(p, value): p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    manifest = load(OUT/"source_manifest.json")
    sources = {r["source"]:r for r in manifest["sources"]}
    expected = load(OUT/"figures/figure_verification.json")["source_sha256"]
    protocol = load(OUT/"supplement/run_v2_portable/protocol.json")
    expected.update(protocol["input_sha256"])
    complete = "output/analysis/project_status_20261108/sources/round8_complete.json"
    expected[complete] = digest(ROOT/complete)
    for rel, sha in sorted(expected.items()):
        rel=rel.replace("\\","/")
        src,dst=ROOT/rel, OUT/"sources"/rel
        assert digest(src)==sha, rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        assert digest(dst)==sha
        sources[rel]=dict(source=rel,snapshot=dst.relative_to(OUT).as_posix(),sha256=sha,bytes=src.stat().st_size)
    for row in sources.values():
        assert digest(ROOT/row["source"])==row["sha256"]
        assert digest(OUT/row["snapshot"])==row["sha256"]
    manifest["sources"]=sorted(sources.values(),key=lambda r:r["source"])
    manifest["scope"]="Exact saved aggregates, development predictions and documentary sources for table/figure/diagnostic reproduction; no images or fitted models."
    save(OUT/"source_manifest.json",manifest)
    code = ROOT/"scripts/report_evidence_20261111"
    for src in code.glob("*.py"):
        dst=OUT/"code/scripts/report_evidence_20261111"/src.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    import numpy,pandas,matplotlib,scipy,sklearn
    deps={"numpy":numpy.__version__,"pandas":pandas.__version__,"matplotlib":matplotlib.__version__,"scipy":scipy.__version__,"scikit-learn":sklearn.__version__}
    (OUT/"code/requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in deps.items())+"\n",encoding="utf-8")
    verify_code='''import hashlib,json,pathlib,zipfile\nroot=pathlib.Path(__file__).resolve().parent\nm=json.loads((root/"PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))\nfor row in m["files"]:\n p=root/row["path"]\n assert p.is_file(),row["path"]\n assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"],row["path"]\nprint(json.dumps({"status":"passed","files":len(m["files"])}))\n'''
    (OUT/"verify_package.py").write_text(verify_code,encoding="utf-8")
    save(OUT/"package_verification.json",dict(status="passed",source_snapshots=len(sources),source_hashes_match_originals=True,source_inputs_for_figures_and_supplement_complete=True,new_model_training_runs=0,final_test_images_or_per_image_predictions_read=False,dependencies=deps,scope="Packaged input hashes and required source availability; detailed arithmetic and bootstrap checks recorded separately.",audit_followup="Five missing portability sources noted by independent review are now included, plus reliability_cells.csv required for figures."))
    entries=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name!="PACKAGE_MANIFEST.json" and "__pycache__" not in p.parts:
            entries.append(dict(path=p.relative_to(OUT).as_posix(),sha256=digest(p),bytes=p.stat().st_size))
    save(OUT/"PACKAGE_MANIFEST.json",dict(date="2026-11-11",files=entries,scope="All deliverable files except this self-referential manifest."))
    archive=OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in entries:
            z.write(OUT/row["path"],OUT.name+"/"+row["path"])
        z.write(OUT/"PACKAGE_MANIFEST.json",OUT.name+"/PACKAGE_MANIFEST.json")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in entries:
            assert hashlib.sha256(z.read(OUT.name+"/"+row["path"])).hexdigest()==row["sha256"]
    archive.with_suffix(".zip.sha256").write_text(digest(archive)+"  "+archive.name+"\n",encoding="utf-8")
    print(json.dumps(dict(status="passed",sources=len(sources),packaged_files=len(entries)+1,zip=str(archive),bytes=archive.stat().st_size),ensure_ascii=False))


if __name__=="__main__": main()
