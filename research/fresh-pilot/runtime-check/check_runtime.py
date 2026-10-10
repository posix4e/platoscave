#!/usr/bin/env python3
"""One explicitly synthetic runtime check; no human packet or scientific score."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import hashlib, importlib.util, io, json, platform, time
from datetime import datetime, timezone
import numpy as np
import torch
import PIL
from PIL import Image, ImageOps

start=time.monotonic()
r=Path('/home/research/fresh-objects-v1')
lock=json.loads((r/'release-lock.json').read_text())
for name, expected in lock['files'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected, name
spec=importlib.util.spec_from_file_location('locked_fresh_extractor',r/'extract.py')
ex=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
assert torch.cuda.is_available(), 'CUDA unavailable'
torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
# Artificial pixel gradient in memory. Never passed through human intake or stored as data.
y,x=np.indices((280,360))
pixels=np.stack((x%256,y%256,(x+y)%256),axis=-1).astype(np.uint8)
buffer=io.BytesIO()
Image.fromarray(pixels).save(buffer,format='PNG')
transformed=ex.preprocess_bytes(buffer.getvalue(),np,Image,ImageOps)
assert transformed.shape==(3,224,224) and transformed.dtype==np.float32
model=ex.load_dino(Path('/opt/rosetta/epic-experiment/dino-source'),Path('/opt/rosetta/clean-example/models/dino_vitbase16_pretrain.pth'),torch).to('cuda',dtype=torch.float32)
tensor=torch.from_numpy(transformed).unsqueeze(0).to('cuda')
with torch.inference_mode():
    first=model(tensor).float().cpu().numpy()
    second=model(tensor).float().cpu().numpy()
assert first.shape==(1,768) and np.isfinite(first).all()
assert np.array_equal(first,second), 'Repeat inference changed bytes'
image=ex.unit_rows(first,np,'synthetic runtime image')
rows=[{'sample_id':'SOFTWARE-FIXTURE-ONLY','caption':'This is a software test only.'}]
text,coverage=ex.text_features(rows,Path('/opt/rosetta/clean-example/models/glove.6B.zip'),np)
assert text.shape==(1,300) and np.isfinite(text).all()
record={
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Software runtime readiness only. Artificial pixels and a fixed software-test string; not human data, not a competence result, not an unpaired result.',
 'source_commit':'3be06bc11a001dbe3894c3060da937a6437e026e',
 'protocol_sha256':lock['files']['protocol.json'],
 'extractor_sha256':lock['files']['extract.py'],
 'release_lock_sha256':ex.sha256(r/'release-lock.json'),
 'checks':{'published_code_hashes':True,'actual_dino_checkpoint_and_source_hashes':True,'actual_glove_archive_and_member_hashes':True,'dino_output_shape':[1,768],'glove_output_shape':[1,300],'dino_repeat_bit_identical':True,'finite_l2_normalized_outputs':bool(np.allclose(np.linalg.norm(image,axis=1),1,atol=1e-4) and np.allclose(np.linalg.norm(text,axis=1),1,atol=1e-4))},
 'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pillow':PIL.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'float32':True,'tf32':False,'deterministic_algorithms':True},
 'artificial_fixture_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
 'human_photographs_used':0,'human_captions_used':0,'human_packet_sealed':False,'alignment_fits_run':0,
 'elapsed_seconds':round(time.monotonic()-start,3),
 'check_script_sha256':ex.sha256(Path(__file__))
}
output=Path('/home/research/fresh-runtime-readiness.json')
with output.open('x') as f:
    json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2))
