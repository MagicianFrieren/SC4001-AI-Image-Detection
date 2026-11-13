import hashlib,json,pathlib,zipfile
root=pathlib.Path(__file__).resolve().parent
m=json.loads((root/"PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
for row in m["files"]:
 p=root/row["path"]
 assert p.is_file(),row["path"]
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"],row["path"]
print(json.dumps({"status":"passed","files":len(m["files"])}))
