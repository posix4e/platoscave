"""Synthetic JSON/byte fixtures only: no tensors, original data, models or fits.

Small row counts and the tensor numerical checker are patched in these unit tests.
The production builder has neither a reduced-count flag nor a validation bypass.
Every artificial artifact exists only inside a TemporaryDirectory.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_feature_lock as builder

r = builder.runner
x = builder.extraction


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='synthetic-feature-lock-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.counts = patch.dict(r.SPLITS, {'train2014': 2, 'val2014': 2}, clear=True)
        self.counts.start(); self.addCleanup(self.counts.stop)
        self.extraction_sha = 'e' * 64
        self.extraction_lock = {'extractor_sha256': 'a' * 64, 'datasets': {},
                                'models': {model: {'runtime_versions': {'python': 'artificial'}} for model in x.MODEL_CONTRACT}}
        self.inputs = {'source_root': str(self.root), 'python_executable': str(self.root/'python'),
                       'locks': {name: str(self.root/(name+'.json')) for name in r.LOCK_NAMES},
                       'datasets': {}, 'baseline_results': {str(seed): str(self.root/f'{seed}.json') for seed in r.SEEDS},
                       'feature_inputs': {}}
        self.publisher = {'repository': 'schnaus/unpaired-rosetta-embeddings',
                          'revision': 'b39767e18cdb5031020a0e1c4b8d9fba6ae13453', 'files': []}
        self.rows = {}
        for split, offset in [('train2014', 0), ('val2014', 100)]:
            annotation = {'images': [{'id': offset+i, 'file_name': f'COCO_{split}_{offset+i:012d}.jpg'} for i in (2,1)],
                          'annotations': [{'id': offset*10+i*10+c, 'image_id': offset+i, 'caption': 'artificial words'} for i in (2,1) for c in range(5)]}
            path = self.root/(split+'-captions.json'); self.write(path, annotation)
            rows = x.ordered_rows(annotation, split)
            manifest = {'split': split, 'annotation_sha256': r.digest(path), 'images':
                        [{'row': i, 'image_id': row['image_id'], 'file': row['file'], 'sha256': 'f'*64} for i,row in enumerate(rows)]}
            mp = self.root/(split+'-images.json'); self.write(mp, manifest)
            self.inputs['datasets'][split] = {'annotation': str(path), 'image_manifest': str(mp)}
            self.extraction_lock['datasets'][split] = {'annotation_sha256': r.digest(path), 'image_manifest_sha256': r.digest(mp)}
        self.dataset_lock, self.rows = builder.datasets(self.extraction_lock, self.inputs['datasets'], r.BoundFiles())
        for model,spec in r.MODEL_SPECS.items():
            for split in r.SPLITS:
                key = f'{model}/{split}'; folder = self.root/(model+'-'+split); folder.mkdir()
                tensor = folder/'features.pt'; tensor.write_bytes(('ARTIFICIAL NON-TENSOR '+key).encode())
                self.inputs['feature_inputs'][key] = str(tensor if spec['kind']=='released' else folder)
                if spec['kind']=='released':
                    self.publisher['files'].append({'path': f"embeddings/coco_{split}/{spec['modality']}/{spec['source_model_id']}.pt", 'sha256':r.digest(tensor),'size_bytes':tensor.stat().st_size})
                    continue
                shape = [2,spec['dimension']] if spec['modality']=='vision' else [2,7,spec['dimension']]
                binding = {'protocol_id':r.PROTOCOL_ID,'model':model,'split':split,'extractor_sha256':'a'*64,
                           'lock_sha256':self.extraction_sha,'row_count':2,'dataset':self.extraction_lock['datasets'][split]}
                metadata = {**binding,'feature_sha256':r.digest(tensor),'shape':shape,'dtype':'float32',
                            'model_contract':x.MODEL_CONTRACT[model], 'preprocessing_id':x.PREPROCESSING_ID,'no_training':True,
                            'runtime_versions':{'python':'artificial'}, 'input_hashes':{k:x.PINNED[k] for k in x.REQUIRED[model]},
                            **self.rows[split]}
                self.write(folder/'features.json',metadata)
                self.write(folder/'COMPLETE.json',{**binding,'status':'complete','last_completed_row':1,'shape':shape,
                          'feature_sha256':r.digest(tensor),'metadata_sha256':r.digest(folder/'features.json')})
        self.bindings = builder.expand_bindings(self.inputs)

    def write(self,path,value):
        Path(path).write_text(json.dumps(value)+'\n')

    def candidate(self):
        return {'datasets':self.dataset_lock,'extraction_lock_sha256':self.extraction_sha,
                'features':builder.feature_entries(self.dataset_lock,self.bindings['features'],self.publisher,r.BoundFiles())}

    def validate_identities(self,lock):
        # Identity checks are production code. Only numeric tensor inspection is
        # omitted because these fixtures are explicitly byte strings, not tensors.
        with patch.object(r,'validate_tensor') as numerical:
            paths=r.feature_paths(lock,self.bindings['features'],self.rows,self.extraction_lock,self.publisher,
                                  r.BoundFiles(),SimpleNamespace(extraction=x))
            self.assertEqual(numerical.call_count,14)
            return paths

    def test_fixed_fourteen_entries_and_private_paths(self):
        lock=self.candidate(); self.assertEqual(len(lock['features']),14)
        self.assertEqual(sum(v['kind']=='released' for v in lock['features'].values()),4)
        self.assertEqual(sum(v['kind']=='extracted' for v in lock['features'].values()),10)
        r.no_local_paths(lock); builder.path_leaves(self.bindings)
        self.assertEqual(len(self.validate_identities(lock)),14)

    def test_missing_extra_or_relative_input_rejected(self):
        for change in ('missing','extra','relative'):
            with self.subTest(change=change):
                bad=copy.deepcopy(self.inputs)
                if change=='missing': bad['feature_inputs'].pop('elmo/val2014')
                if change=='extra': bad['feature_inputs']['other/val2014']=str(self.root)
                if change=='relative': bad['source_root']='relative'
                with self.assertRaises(ValueError): builder.expand_bindings(bad)

    def test_failed_and_incomplete_attempt_rejected(self):
        folder=Path(self.inputs['feature_inputs']['dino/train2014'])
        (folder/'FAILED.json').write_text('{}')
        with self.assertRaises(ValueError): builder.expand_bindings(self.inputs)
        (folder/'FAILED.json').unlink()
        marker=r.read_json(folder/'COMPLETE.json'); marker['status']='started'; self.write(folder/'COMPLETE.json',marker)
        with self.assertRaises(ValueError): self.candidate()
        (folder/'COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError): self.candidate()

    def test_partial_and_late_failure_markers_block_builder_and_feature_consumers(self):
        folder=Path(self.inputs['feature_inputs']['dino/train2014'])
        for marker in ('FAILED.json','COMPLETE.json.partial','FAILED.json.partial'):
            with self.subTest(marker=marker):
                candidate=self.candidate()
                assembled=r.BoundFiles(); builder.expand_bindings(self.inputs,assembled)
                downstream=r.BoundFiles()
                with patch.object(r,'validate_tensor'):
                    r.feature_paths(candidate,self.bindings['features'],self.rows,self.extraction_lock,self.publisher,
                                    downstream,SimpleNamespace(extraction=x))
                (folder/marker).write_text('{"interrupted":')
                with self.assertRaisesRegex(ValueError,'Conflicting or partial'): builder.expand_bindings(self.inputs)
                with self.assertRaisesRegex(ValueError,'Conflicting or partial'): self.validate_identities(candidate)
                with self.assertRaisesRegex(ValueError,'Conflicting or partial'): assembled.recheck()
                with self.assertRaisesRegex(ValueError,'Conflicting or partial'): downstream.recheck()
                (folder/marker).unlink()

    def test_completion_metadata_and_dataset_identity_guards(self):
        folder=Path(self.inputs['feature_inputs']['dino/train2014'])
        original_metadata=r.read_json(folder/'features.json'); original_marker=r.read_json(folder/'COMPLETE.json')
        for field,bad_value in [('model','moco'),('dataset',{}),('image_ids',[999,1000]),('lock_sha256','b'*64)]:
            with self.subTest(field=field):
                metadata=copy.deepcopy(original_metadata); metadata[field]=bad_value; self.write(folder/'features.json',metadata)
                marker=copy.deepcopy(original_marker); marker['metadata_sha256']=r.digest(folder/'features.json'); self.write(folder/'COMPLETE.json',marker)
                with self.assertRaises(ValueError): self.validate_identities(self.candidate())
        self.write(folder/'features.json',original_metadata); self.write(folder/'COMPLETE.json',original_marker)
        lock=self.candidate(); lock['features']['dino/train2014']['sha256']='c'*64
        with self.assertRaises(ValueError): self.validate_identities(lock)

    def test_dataset_manifest_order_and_byte_changes_rejected(self):
        manifest_path=Path(self.inputs['datasets']['train2014']['image_manifest'])
        manifest=r.read_json(manifest_path); manifest['images'].reverse(); self.write(manifest_path,manifest)
        # Even if the changed manifest hash is presented, row order is rejected.
        self.extraction_lock['datasets']['train2014']['image_manifest_sha256']=r.digest(manifest_path)
        with self.assertRaises(ValueError): builder.datasets(self.extraction_lock,self.inputs['datasets'],r.BoundFiles())

    def test_explicit_protocol_sha_and_draft_protocol_rejected(self):
        path=self.root/'protocol.json'; self.write(path,{'schema_version':1,'protocol_id':r.PROTOCOL_ID,'status':'draft'})
        with self.assertRaises(ValueError): builder.assemble(path,'0'*64,{})
        with self.assertRaises(ValueError): builder.assemble(path,r.digest(path),{})

    def test_draft_extraction_lock_rejected(self):
        draft=Path(self.inputs['locks']['extraction']); self.write(draft,{'schema_version':1,'protocol_id':r.PROTOCOL_ID,'status':'draft'})
        protocol={'schema_version':1,'protocol_id':r.PROTOCOL_ID,'status':'frozen_before_model_control_fits',
                  'upstream_commit':r.UPSTREAM,'seeds':r.SEEDS,'vision_models':r.VISION,'text_models':r.TEXT,
                  'methods':r.METHODS,'models':r.MODEL_SPECS,'counts':r.COUNTS,'algorithm_defaults':r.ALGORITHM,
                  'new_rosetta_fits':60,'baseline_replay':{'vision':'dinov2','text':'mpnet','absolute_tolerance':1e-10,'require_all_five_before_expansion':True},
                  'locks':{name:r.digest(draft) for name in r.LOCK_NAMES},
                  'code_sha256':{name:r.digest(builder.ROOT/name) for name in r.CODE_NAMES},
                  'execution':{'num_workers':4,'num_threads':4,'device':'cpu','evaluation_block_size':4096,'torch_threads':16,'torch_interop_threads':16}}
        path=self.root/'artificial-protocol.json'; self.write(path,protocol)
        with self.assertRaisesRegex(ValueError,'final extraction lock'):
            builder.assemble(path,r.digest(path),self.inputs)

    def test_full_preflight_failure_publishes_nothing(self):
        out,private=self.root/'feature-lock.json',self.root/'private-bindings.json'
        inputs=self.root/'inputs.json'; self.write(inputs,self.inputs)
        args=SimpleNamespace(out=out,bindings_out=private,protocol=self.root/'absent-protocol.json',
                             protocol_sha256='a'*64,inputs=inputs)
        with patch.object(builder,'assemble',return_value=(self.candidate(),self.bindings,r.BoundFiles())), \
             patch.object(r,'locked_inputs',side_effect=ValueError('artificial full preflight failure')):
            with self.assertRaisesRegex(ValueError,'full preflight failure'): builder.build(args)
        self.assertFalse(out.exists()); self.assertFalse(private.exists())

    def test_existing_outputs_rejected_without_touching_them(self):
        out=self.root/'existing.json'; out.write_bytes(b'keep original')
        with self.assertRaises(ValueError): builder.check_outputs(out,self.root/'bindings.json')
        self.assertEqual(out.read_bytes(),b'keep original')
        with self.assertRaises(ValueError): builder.check_outputs(self.root/'new.json',builder.ROOT/'private.json')


if __name__=='__main__':
    unittest.main()
