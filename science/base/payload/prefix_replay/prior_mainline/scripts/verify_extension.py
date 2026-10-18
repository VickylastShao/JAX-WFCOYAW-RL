from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
for row in manifest['files']:
 p=root/row['path']
 assert p.is_file() and p.stat().st_size==row['bytes'],row['path']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],row['path']
print('verified',len(manifest['files']),'members')
