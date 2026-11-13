"""Assemble byte-preserved repository and November source; never run experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile
import argparse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/analysis/submission_revision_20261114'
TREE=OUT/'source_tree'
MAIN='42059e3d0eabcb6393efcb9dd3b879092963af96'
ANALYSIS='e498d0ed997e679a5db22b4b3ce05db34e6b31a0'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def archive_into(commit,paths=()):
    tarpath=ROOT/'tmp'/f'submission_snapshot_{commit[:8]}.tar'
    with tarpath.open('wb') as f:
        subprocess.run(['git','archive','--format=tar',commit,*paths],cwd=ROOT,stdout=f,check=True)
    with tarfile.open(tarpath) as tar:
        for member in tar.getmembers():
            dest=(TREE/member.name).resolve()
            assert dest.is_relative_to(TREE.resolve()),member.name
            assert not member.issym() and not member.islnk(),member.name
        tar.extractall(TREE,filter='data')

def copy_rel(rel):
    src=ROOT/rel; dst=TREE/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): assert digest(src)==digest(dst),f'Conflicting bytes: {rel}'
    else: shutil.copy2(src,dst)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-after-main',action='store_true');args=parser.parse_args()
    if TREE.exists():
        assert args.resume_after_main,'Use a new source-tree location; do not overwrite a submission snapshot.'
        # Recovery is only allowed for an exact, unmodified main archive.
        expected={}
        listing=subprocess.check_output(['git','ls-tree','-r',MAIN],cwd=ROOT,text=True)
        for row in listing.splitlines():
            meta,name=row.split('\t',1);expected[name]=meta.split()[2]
        actual={p.relative_to(TREE).as_posix() for p in TREE.rglob('*') if p.is_file()}
        assert actual==set(expected),'Partial tree differs from pinned main archive'
        for name,blob in expected.items():
            data=(TREE/name).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob,name
    else:
        TREE.mkdir(parents=True)
        archive_into(MAIN)
    # Bring the teammate's analysis into the snapshot without modifying the checkout.
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',ANALYSIS],cwd=ROOT,text=True).splitlines()
    chosen=[p for p in files if p.startswith('results/analysis_c/') or ('analysis_c' in p and p.endswith('.py'))]
    assert chosen
    archive_into(ANALYSIS,chosen)
    studies=['gate_diagnosis','gate_failure_mechanism_20261108','multigen_gate_20261108','recompression_reliability_20261108','paired_view_global_20261108']
    for name in studies:
        for src in (ROOT/'tmp'/name).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and src.suffix not in ['.log','.pyc']:
                copy_rel(src.relative_to(ROOT))
    for folder in ['scripts/report_evidence_20261111','scripts/submission_revision_20261114']:
        for src in (ROOT/folder).glob('*.py'): copy_rel(src.relative_to(ROOT))
    for name in ['gate_diagnosis_20261108','gate_failure_mechanism_20261108','multigen_gate_20261108','recompression_reliability_20261108','paired_view_global_20261108']:
        for src in (ROOT/'output/analysis'/name).rglob('*'):
            if src.is_file() and not any(part in ['features','canary_features','__pycache__'] for part in src.parts):
                copy_rel(src.relative_to(ROOT))
    closure=json.loads((OUT/'review/dependency_closure.json').read_text(encoding='utf-8'))
    for item in closure['source_files']:
        path=Path(item['path']); copy_rel(path)
        assert digest(TREE/path)==item['sha256'],str(path)
    shutil.copytree(ROOT/'output/analysis/report_evidence_20261111',OUT/'evidence')
    sources=[]
    for p in sorted(TREE.rglob('*')):
        if p.is_file(): sources.append({'path':p.relative_to(TREE).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
    record={'main_commit':MAIN,'analysis_c_commit':ANALYSIS,'source_files':sources,'november_source_closure_checked':len(closure['source_files']),'preservation':'Source bytes preserved; original tmp names are relative paths inside this delivered tree, not references to a user temporary directory.','excluded_inputs':'Raw images, pretrained CLIP weights, R8/recompression feature caches and six large R8 runtime bundles; exact references and hashes are in review/input_artifact_manifest.json.','executed_model_experiments':0}
    (OUT/'source_tree_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','source_files':len(sources),'source_MB':sum(x['bytes'] for x in sources)/1e6,'closure_files':len(closure['source_files'])}))

if __name__=='__main__':main()
