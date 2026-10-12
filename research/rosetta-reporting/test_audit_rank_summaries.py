"""Bounded invented inputs only. No real scientific output or runner imports."""
import copy
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import audit_rank_summaries as a


def encode(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def frozen_group(values, *, source=False, contrast=False, operands=None):
    names = a.SOURCE_METRICS if source else a.METRICS
    individuals = {str(k): {'status': 'available', 'metrics': v} for k, v in values.items()}
    if contrast:
        for item in individuals.values():
            item['missing_masks' if source else 'missing_entries'] = []
    group = {'status': 'available', 'ddof': 1,
        'mean': {m: statistics.mean(v[m] for v in values.values()) for m in names},
        'sample_sd': {m: statistics.stdev(v[m] for v in values.values()) for m in names}}
    if source:
        group.update(required=list(values), available=list(values), individual=individuals,
            minimum={m: min(v[m] for v in values.values()) for m in names},
            maximum={m: max(v[m] for v in values.values()) for m in names})
    else:
        group.update(required_seeds=list(a.SEEDS), available_seeds=list(a.SEEDS), per_seed=individuals)
    if operands is not None:
        group['operands' if source else 'terms'] = operands
    return group


def model_descriptive(scores):
    result = {'metrics': list(a.METRICS), 'cell_scores': {}, 'contrasts': {}}
    def metric(v, t, seed, method):
        return scores[f'{v}/{t}/{seed}/{method}']
    def contrast(label, terms):
        values = {s: {m: math.fsum(c * metric(v, t, s, method)[m] for c,v,t,method in terms)
                        for m in a.METRICS} for s in a.SEEDS}
        result['contrasts'][label] = frozen_group(values, contrast=True,
            operands=[dict(coefficient=c, vision=v, text=t, method=method) for c,v,t,method in terms])
    for v in a.VISION:
        for t in a.TEXT:
            for method in a.METHODS:
                result['cell_scores'][f'{v}/{t}/{method}'] = frozen_group({s: metric(v,t,s,method) for s in a.SEEDS})
            for label, left, right in [('orthogonal_pairing', 'orthogonal_paired', 'orthogonal_shuffled'),
                    ('linear_pairing', 'linear_paired', 'linear_shuffled'),
                    ('rosetta_minus_orthogonal_paired', 'rosetta', 'orthogonal_paired'),
                    ('rosetta_minus_linear_paired', 'rosetta', 'linear_paired')]:
                contrast(f'within/{v}/{t}/{label}', [(1,v,t,left),(-1,v,t,right)])
    for method in a.METHODS:
        for t in a.TEXT[1:]:
            contrast(f'primary/text/{t}/{method}', [(1,'dinov2',t,method),(-1,'dinov2','mpnet',method)])
        for v in a.VISION[1:]:
            contrast(f'primary/vision/{v}/{method}', [(1,v,'mpnet',method),(-1,'dinov2','mpnet',method)])
            for t in a.TEXT[1:]:
                contrast(f'interaction/{v}/{t}/{method}', [(1,v,t,method),(-1,v,'mpnet',method),(-1,'dinov2',t,method),(1,'dinov2','mpnet',method)])
    return result


def source_descriptive(scores, masks, families):
    result = {'metrics': list(a.SOURCE_METRICS), 'per_mask_five_seed': {}, 'within_seed_random_mask_spread': {}, 'within_seed_contrasts': {}}
    views = ['original_overall', *a.STRATA]
    for mask in masks:
        for view in views:
            result['per_mask_five_seed'][f'{mask}/{view}'] = frozen_group({s: scores[f'{mask}/{s}'][view] for s in a.SEEDS}, source=True)
    for family, names in families.items():
        for view in views:
            result['within_seed_random_mask_spread'][f'{family}/{view}'] = {
                str(s): frozen_group({mask: scores[f'{mask}/{s}'][view] for mask in names}, source=True) for s in a.SEEDS}
    for view in a.STRATA:
        for label, left, right in [('C_minus_mean_Rv','C','Rv'), ('D_minus_mean_Rp','D','Rp'), ('C_minus_O','C','O'), ('D_minus_C','D','C')]:
            others = families.get(right, [right])
            values = {s: {m: scores[f'{left}/{s}'][view][m] - statistics.mean(scores[f'{other}/{s}'][view][m] for other in others)
                        for m in a.SOURCE_METRICS} for s in a.SEEDS}
            result['within_seed_contrasts'][f'{view}/{label}'] = frozen_group(values, source=True, contrast=True, operands=[left,*others])
    return result


def tensor_bytes(record):
    stream = io.BytesIO()
    torch.save(record, stream)
    return stream.getvalue()


def source_metrics(ranks, higher, ties):
    """Fixture's frozen-style float arithmetic, deliberately not audit arithmetic."""
    order = sorted(ranks); n = len(ranks)
    return {'num_queries': n, 'gallery_size': 4, 'foscttm_denominator': 3,
        'foscttm': sum((r-1)/3 for r in ranks)/n, 'mean_rank': sum(ranks)/n,
        'median_rank': (order[(n-1)//2] + order[n//2])/2,
        **{f'recall_at_{k}': sum(max(0, min(1, (k-h)/t)) for h,t in zip(higher,ties))/n for k in (1,5,10)}}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root/'inputs'; self.inputs.mkdir()
        self.cells = self.inputs/'cells'; self.cells.mkdir()
        self.rows = patch.object(a, 'ROWS', 4); self.rows.start(); self.addCleanup(self.rows.stop)
        self.strata = patch.object(a, 'STRATA', {'never_known_spc_overlap': 2, 'original_spc_overlap': 2, 'all': 4})
        self.strata.start(); self.addCleanup(self.strata.stop)
        self.pins = copy.deepcopy(a.PINS)
        self.pinpatch = patch.object(a, 'PINS', self.pins); self.pinpatch.start(); self.addCleanup(self.pinpatch.stop)
        self.args = SimpleNamespace(summary=self.inputs/'summary.json', acceptance=self.inputs/'accepted.json',
            study_receipt=self.inputs/'stage.json', protocol=self.inputs/'protocol.json', masks=None, masks_sha256=None,
            attempts_root=self.cells, out=self.root/'reports'/'report.json')

    def prepare(self, study):
        self.args.study = study
        self.args.study_receipt = self.inputs/f'{study}-stage-complete.json'
        protocol = {'protocol_id': 'rosetta-model-controls-v2' if study == 'model' else 'rosetta-source-controls-v1',
            'seeds': list(a.SEEDS), 'counts': {'validation': 4}, 'research_design': {'reporting': {'invented': True}}}
        if study == 'source':
            protocol.update(selection_seeds={'Rv': list(range(1000,1010)), 'Rp': list(range(2000,2010))}, row_identity_sha256='c'*64)
        self.args.protocol.write_bytes(encode(protocol)); self.args.protocol_sha256 = a.sha(self.args.protocol.read_bytes())
        self.pins[study]['protocol'] = self.args.protocol_sha256
        self.summary = {'schema_version': 1, 'protocol_id': protocol['protocol_id'], 'status': 'complete', 'inventory_findings': [],
            'input_hashes': {**self.pins[study], 'prespecified_reporting': a.object_hash(protocol['research_design']['reporting']),
                'private_bindings':'a'*64, 'runner_bindings':'b'*64}, 'attempts': {}, 'observations': {}}
        if study == 'model':
            self.make_model()
        else:
            self.make_source(protocol)
        return self.refresh()

    def refresh(self):
        self.args.summary.write_bytes(encode(self.summary)); self.args.summary_sha256 = a.sha(self.args.summary.read_bytes())
        self.args.acceptance.write_bytes(encode({'summary_sha256': self.args.summary_sha256}))
        self.args.acceptance_sha256 = a.sha(self.args.acceptance.read_bytes())
        ticket={'study':self.args.study, 'preflight':False, 'out':str(self.args.summary)}
        (self.inputs/'ticket.json').write_bytes(encode(ticket))
        self.args.study_receipt.write_bytes(encode({'summary_path': str(self.args.summary), 'summary_sha256': self.args.summary_sha256,
            'verification': str(self.inputs), 'ticket_sha256': a.object_hash(ticket)}))
        self.args.study_receipt_sha256 = a.sha(self.args.study_receipt.read_bytes())
        return self.args

    def save_rank(self, selected, record, method=None):
        target = self.cells/selected
        if method: target /= method
        target.mkdir(parents=True, exist_ok=True); data = tensor_bytes(record); (target/'ranks.pt').write_bytes(data)
        return a.sha(data)

    def make_model(self):
        mapping = {'validation_image_ids':[30,10,40,20], 'validation_indices':[2,0,3,1],
            'query_order':[2,0,3,1], 'key_order':[1,3,0,2], 'partner_in_shuffled_keys':[3,2,1,0],
            'sort_positions_by_image_id':[1,3,0,2], 'training_split_hashes':{'indices_x':'a'*64, 'indices_y':'b'*64}}
        self.summary.update(planned={'cells':12,'seeds':5,'method_slots':300},
            baseline_replay_gate={'status':'satisfied','eligible_reference_cell_completions_by_seed':{str(s):1 for s in a.SEEDS}},
            validation_order_and_coco_id_mapping={str(s):copy.deepcopy(mapping) for s in a.SEEDS})
        scores = {}; options = [[1.,2.,3.,4.], [1.,1.5,2.5,3.], [3.5,1.,4.,2.], [4.,4.,4.,4.]]
        for vi,v in enumerate(a.VISION):
            for ti,t in enumerate(a.TEXT):
                for si,s in enumerate(a.SEEDS):
                    cell=f'{v}/{t}/{s}'; selected=f'{v}--{t}/seed-{s}/attempt-01'
                    attempt={'attempt':selected,'terminal_marker':'COMPLETE.json','terminal_manifest_verified':True,
                        'split_verified':True,'terminal_partial_markers':[],'findings':[],'methods':{},'input_hashes':{}}
                    for mi,m in enumerate(a.METHODS):
                        ranks=options[(vi+ti+si+mi)%4]; metrics={'foscttm':sum((r-1)/3 for r in ranks)/4,'mean_rank':sum(ranks)/4}
                        record={k:torch.tensor(mapping[k],dtype=torch.int64) for k in a.MODEL_ORDER}
                        record['ranks']=torch.tensor(ranks,dtype=torch.float64); rank_sha=self.save_rank(selected,record,m)
                        check={'status':'verified','order_ids_checked_against_original_rng':True,'gallery_count_matches':True,
                            'validation_count':4,'recomputed_metrics':metrics,'comparisons':{metric:{'exactly_equal':True} for metric in a.METRICS},
                            'canonical_ranks_by_coco_id_sha256':a.object_hash([ranks[i] for i in mapping['sort_positions_by_image_id']])}
                        attempt['methods'][m]={'status':'complete','eligible':True,'findings':[],'rank_check':check,
                            'reported_metrics':metrics,'ranks_sha256':rank_sha}
                        attempt['input_hashes'][f'{m}/ranks.pt']=rank_sha
                        self.summary['observations'][f'{cell}/{m}']={'status':'available','retry_history_findings':[],
                            'repeat_agreement':None,'selected_attempt':selected,'metrics':metrics}
                        scores[f'{cell}/{m}']=metrics
                    self.summary['attempts'][cell]=[attempt]
        self.summary['descriptive']=model_descriptive(scores)

    def make_source(self, protocol):
        families={f:[f'{f}-{s}' for s in protocol['selection_seeds'][f]] for f in ('Rv','Rp')}
        masks=['O','C','D',*families['Rv'],*families['Rp']]
        strata={'never_known_spc_overlap':[0,2],'original_spc_overlap':[1,3],'all':[0,1,2,3]}
        bundle={'protocol_id':protocol['protocol_id'],'protocol_sha256':self.args.protocol_sha256,
            'row_identity_sha256':protocol['row_identity_sha256'],'strata':strata,
            'masks':{m:{'mask_sha256':a.sha(m.encode())} for m in masks}}
        self.args.masks=self.inputs/'masks.json'; self.args.masks.write_bytes(encode(bundle))
        self.args.masks_sha256=a.sha(self.args.masks.read_bytes()); self.pins['source']['masks']=self.args.masks_sha256
        self.summary['input_hashes']['masks']=self.args.masks_sha256
        mapping={'validation_order':[2,0,3,1],'query_order':[3,2,1,0],'key_order':[1,0,2,3], 'canonical_coco_image_ids':[10,20,30,40]}
        self.summary.update(planned_masks=23,planned_fits=115,schedule=[],validation_order_and_coco_ids={str(s):copy.deepcopy(mapping) for s in a.SEEDS})
        scores={}
        for mi,mask in enumerate(masks):
            for si,seed in enumerate(a.SEEDS):
                identity=f'{mask}/{seed}'; selected=f'{mask}/seed-{seed}/attempt-01'
                # Exact ties include 1/3 Recall credit; alternate rows to vary fixed strata.
                higher=[0,0,2,3]; ties=[3,1,2,1]
                offset=(mi+si)%4; higher=higher[offset:]+higher[:offset]; ties=ties[offset:]+ties[:offset]
                ranks=[float(h+(t+1)/2) for h,t in zip(higher,ties)]
                raw={'ranks':ranks,'num_higher':higher,'tie_count':ties}
                record={k:torch.tensor(v,dtype=torch.float64 if k=='ranks' else torch.int64) for k,v in {**mapping,**raw}.items()}
                rank_sha=self.save_rank(selected,record)
                views={'original_overall':mapping['validation_order'],**strata}
                metrics={view:source_metrics(*([raw[k][i] for i in indices] for k in ('ranks','num_higher','tie_count'))) for view,indices in views.items()}
                reported={'original_overall':{m:metrics['original_overall'][m] for m in a.METRICS},'overall':metrics['original_overall'],
                    'fixed_strata':{v:metrics[v] for v in strata},'rank_order':'canonical_original_file_row',
                    'tie_policy':'expected_credit_under_uniform_order_within_exact_score_tie_block','overall_reduction_order':'original_randomized_validation_order'}
                check={'status':'verified','differences':[],'original_evaluation_order_and_coco_ids_checked':True,
                    'canonical_ranks_higher_ties_sha256':a.object_hash(raw),'recomputed':reported}
                attempt={'attempt':selected,'terminal_manifest_verified':True,'terminal_partial_markers':[],'findings':[],
                    'terminal_markers':['COMPLETE.json'],'eligible':True,'training_order_verified':True,'input_hashes':{'ranks.pt':rank_sha},
                    'result':{'artifact_hashes':{'ranks.pt':rank_sha}},'reported_metrics':reported,'recomputed_rank_metrics':check}
                self.summary['attempts'][identity]=[attempt]
                scores[identity]={view:{m:metrics[view][m] for m in a.SOURCE_METRICS} for view in views}
                self.summary['observations'][identity]={'status':'available','complete_markers':1,'verified_complete_attempts':1,
                    'retry_history_findings':[],'selected_attempt':selected,'scores':scores[identity]}
                self.summary['schedule'].append({'index':len(self.summary['schedule']),'mask':mask,'seed':seed,'mask_sha256':bundle['masks'][mask]['mask_sha256']})
        self.summary['descriptive']=source_descriptive(scores,masks,families)

    def test_model_all_identities_contrasts_and_nonzero_difference(self):
        self.prepare('model')
        # A numeric discrepancy is retained, not rejected or labeled scientific failure.
        group=self.summary['descriptive']['cell_scores']['dinov2/mpnet/rosetta']; group['mean']['foscttm'] += 0.01
        result=a.audit(self.refresh()); report=json.loads(self.args.out.read_text())
        self.assertEqual(result['scientific_acceptance'],'not_assessed')
        self.assertEqual(len(report['selected_artifacts']),300)
        self.assertEqual(report['coverage']['contrast_series'],103)
        self.assertEqual(report['comparison_counts']['rank_metrics']['scalars'],600)
        self.assertEqual(report['comparison_counts']['aggregates_from_saved_scalars']['scalars'],2282)
        found=[r for r in report['comparisons']['aggregates_from_saved_scalars'] if r['field']=='cell_scores/dinov2/mpnet/rosetta/mean/foscttm'][0]
        self.assertAlmostEqual(found['signed_difference_frozen_minus_independent'],0.01)
        self.assertNotIn(str(self.root),self.args.out.read_text())

    def test_source_all_masks_views_contrasts_and_fractional_recall(self):
        a.audit(self.prepare('source')); report=json.loads(self.args.out.read_text())
        self.assertEqual(len(report['selected_artifacts']),115)
        self.assertEqual(report['coverage']['five_seed_series'],92)
        self.assertEqual(report['coverage']['random_mask_spread_series'],40)
        self.assertEqual(report['coverage']['contrast_series'],12)
        self.assertEqual(report['comparison_counts']['rank_metrics']['scalars'],2760)
        self.assertEqual(report['comparison_counts']['aggregates_from_saved_scalars']['scalars'],8976)
        rows=report['comparisons']['rank_metrics']
        recall=next(r for r in rows if r['field']==f'O/{a.SEEDS[0]}/original_overall/recall_at_1')
        self.assertEqual(recall['independent_exact_fraction'],'1/3')
        self.assertFalse(recall['exactly_equal_at_reported_precision'])
        self.assertTrue(recall['binary64_exactly_equal'])
        stratum=next(r for r in rows if r['field']==f'O/{a.SEEDS[0]}/never_known_spc_overlap/recall_at_1')
        self.assertEqual(stratum['independent_exact_fraction'],'1/6')

    def test_even_median_ties_and_independent_sample_sd(self):
        result=a.rank_metrics([2.,1.,3.5,4.],[0,0,2,3],[3,1,2,1])
        self.assertEqual(result['median_rank'],Fraction(11,4))
        self.assertEqual(result['recall_at_1'],Fraction(1,3))
        self.assertEqual(result['foscttm'],Fraction(13,24))
        self.assertAlmostEqual(float(a.stats([{'x':Fraction(1)},{'x':Fraction(3)}],['x'])['sample_sd']['x']),math.sqrt(2))

    def test_missing_observation_and_missing_contrast_rejected(self):
        self.prepare('model'); original=copy.deepcopy(self.summary)
        del self.summary['observations'][next(iter(self.summary['observations']))]
        with self.assertRaisesRegex(ValueError,'coverage'): a.audit(self.refresh())
        self.summary=original; del self.summary['descriptive']['contrasts'][next(iter(self.summary['descriptive']['contrasts']))]
        with self.assertRaisesRegex(ValueError,'coverage'): a.audit(self.refresh())
        self.assertFalse(self.args.out.exists())

    def test_missing_source_mask_and_wrong_contrast_operands_rejected(self):
        self.prepare('source'); original=copy.deepcopy(self.summary)
        self.summary['observations'][next(iter(self.summary['observations']))]['status']='unavailable'
        with self.assertRaisesRegex(ValueError,'unavailable'): a.audit(self.refresh())
        self.summary=original
        group=next(iter(self.summary['descriptive']['within_seed_contrasts'].values())); group['operands'].pop()
        with self.assertRaisesRegex(ValueError,'operands'): a.audit(self.refresh())

    def test_mapping_wrong_hash_and_unaccepted_rejected(self):
        self.prepare('model'); original=copy.deepcopy(self.summary)
        self.summary['validation_order_and_coco_id_mapping'][str(a.SEEDS[0])]['validation_image_ids'][0]=999
        with self.assertRaises(ValueError): a.audit(self.refresh())
        self.summary=original; self.refresh()
        rank=next(self.cells.rglob('ranks.pt')); rank.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA-256'): a.audit(self.args)
        self.args.acceptance_sha256='0'*64
        with self.assertRaisesRegex(ValueError,'SHA-256'): a.audit(self.args)

    def test_receipt_type_ticket_and_existing_or_input_output_rejected(self):
        self.prepare('model')
        ticket_path=self.inputs/'ticket.json'; ticket=json.loads(ticket_path.read_text()); ticket['preflight']=True
        ticket_path.write_bytes(encode(ticket))
        receipt=json.loads(self.args.study_receipt.read_text()); receipt['ticket_sha256']=a.object_hash(ticket)
        self.args.study_receipt.write_bytes(encode(receipt)); self.args.study_receipt_sha256=a.sha(self.args.study_receipt.read_bytes())
        with self.assertRaisesRegex(ValueError,'ticket'): a.audit(self.args)
        self.refresh(); preflight=self.inputs/'preflight-model.json'; preflight.write_bytes(self.args.study_receipt.read_bytes())
        self.args.study_receipt=preflight
        with self.assertRaisesRegex(ValueError,'preflight'): a.audit(self.args)
        self.args.study_receipt=self.inputs/'model-stage-complete.json'; self.refresh()
        self.args.out=self.cells/'unsafe.json'
        with self.assertRaisesRegex(ValueError,'fresh output'): a.audit(self.args)
        self.args.out=self.root/'existing.json'; self.args.out.write_text('preserve me')
        with self.assertRaisesRegex(ValueError,'fresh output'): a.audit(self.args)
        self.assertEqual(self.args.out.read_text(),'preserve me')

    def test_relocated_final_receipt_keeps_original_ticket_binding(self):
        self.prepare('model')
        recorded='/original/private/verifications/verify-0099'
        ticket={'study':'model','preflight':False,'final_audit':True,'out':recorded+'/summary.json'}
        (self.inputs/'ticket.json').write_bytes(encode(ticket))
        self.args.study_receipt=self.inputs/'final-audit-model.json'
        self.args.study_receipt.write_bytes(encode({'summary_path':ticket['out'],'summary_sha256':self.args.summary_sha256,
            'verification':recorded,'ticket_sha256':a.object_hash(ticket)}))
        self.args.study_receipt_sha256=a.sha(self.args.study_receipt.read_bytes())
        result=a.audit(self.args)
        self.assertEqual(result['status'],'comparison_generated')
        self.assertNotIn(recorded,self.args.out.read_text())

    def test_source_tie_mismatch_rejected_even_with_new_artifact_hash(self):
        self.prepare('source'); identity=f'O/{a.SEEDS[0]}'
        attempt=self.summary['attempts'][identity][0]; path=self.cells/attempt['attempt']/'ranks.pt'
        record=torch.load(path,weights_only=True); record['tie_count'][0]=1
        data=tensor_bytes(record); path.write_bytes(data); digest=a.sha(data)
        attempt['input_hashes']['ranks.pt']=attempt['result']['artifact_hashes']['ranks.pt']=digest
        with self.assertRaisesRegex(ValueError,'tie counts'): a.audit(self.refresh())


if __name__ == '__main__':
    unittest.main()
