from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path('/home/research/rosetta-replication-v1')
STORE=ROOT/'storage';STORE.mkdir(exist_ok=True)
REPO='schnaus/unpaired-rosetta-embeddings'
REV='b39767e18cdb5031020a0e1c4b8d9fba6ae13453'
VISION='dinov2_vit-b14@224_mean'
paths=[f'embeddings/{d}/{m}/{n}.pt' for d in ['coco_train2014','coco_val2014'] for m,n in [('vision',VISION),('language','mpnet')]]
for d in ['CIFAR-10','CIFAR-100','ImageNet-100']:
 folder='language' if d=='ImageNet-100' else 'language_b953684b7ed3a62cbea9fe4e30935ee2'
 paths.extend([f'embeddings/{d}/vision/{VISION}.pt',f'embeddings/{d}/{folder}/mpnet.pt',f'embeddings/{d}/labels.pt'])
api=HfApi();infos=api.get_paths_info(REPO,paths,repo_type='dataset',revision=REV)
by={x.path:x for x in infos}
assert set(by)==set(paths)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
manifest={'schema_version':1,'repository':REPO,'revision':REV,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'files':[]}
for name in paths:
 info=by[name]
 assert info.lfs is not None,name
 manifest['files'].append({'path':name,'size_bytes':info.size,'sha256':info.lfs.sha256})
with (ROOT/'inputs-lock.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
receipt={'status':'in_progress','input_lock_sha256':sha(ROOT/'inputs-lock.json'),'files':[]}
for item in manifest['files']:
 name=item['path'];expected=item['sha256'];target=STORE/name;cache=Path('/opt/rosetta/storage')/name
 if cache.is_file():
  assert cache.stat().st_size==item['size_bytes'] and sha(cache)==expected, name
  target.parent.mkdir(parents=True,exist_ok=True)
  assert not target.exists();target.symlink_to(cache)
  origin='reused_verified_cache'
 else:
  target=Path(hf_hub_download(REPO,name,repo_type='dataset',revision=REV,local_dir=STORE))
  origin='downloaded_from_pinned_release'
 assert target.stat().st_size==item['size_bytes'] and sha(target)==expected,name
 receipt['files'].append({**item,'acquisition':origin,'verified_at_utc':datetime.now(timezone.utc).isoformat()})
 (ROOT/'inputs-acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'status':'verified','file':name,'bytes':item['size_bytes']}),flush=True)
receipt['status']='complete';receipt['completed_at_utc']=datetime.now(timezone.utc).isoformat()
(ROOT/'inputs-acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('ALL13INPUTSVERIFIED',flush=True)
