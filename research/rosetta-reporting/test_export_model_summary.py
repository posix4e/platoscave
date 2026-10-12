"""Invented four-row, all-300-observation fixtures; no real outputs or tensor reads."""
import copy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_rank_summaries as audit
import export_rank_tables as ranks
import export_model_summary as projection
import test_audit_rank_summaries as audit_fixture

encode = audit_fixture.encode
fixture_controller_ticket_hash = audit_fixture.fixture_controller_ticket_hash


class ModelSummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=audit_fixture.AuditTests('test_even_median_ties_and_independent_sample_sd')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare('model')  # Creates synthetic input tensors; they are never loaded.
        self.root=self.fixture.root;self.inputs=self.fixture.inputs;self.summary=self.fixture.summary
        self.summary['recorded_at_utc']='2026-10-12T00:00:00+00:00'
        self.args=SimpleNamespace(summary=self.fixture.args.summary,acceptance=self.fixture.args.acceptance,
            stage_receipt=self.inputs/'model-stage-complete.json',protocol=self.fixture.args.protocol,
            audit=self.inputs/'model-audit.json',rank_manifest=self.inputs/'rank-manifest.json',
            out=self.root/'projection'/'model-summary.json')
        self.protocol=json.loads(self.args.protocol.read_text())
        self.protocol['research_design']['reporting']={k:f'Invented fixed {k} rule.' for k in projection.REPORTING_FIELDS}
        self.protocol['research_design'].update(stage='Invented selected exploratory panel.',
            prior_information='Invented prior selection disclosure.',interpretation_unit='Invented complete pipeline scope.',
            paired_controls={'extra_information':'Invented extra counterparts disclosure.'},
            evaluation={'direction':'lower is better','mean_rank_relation':'Invented dependent metric rule.'})
        self.summary['descriptive'].update(direction=projection.DESCRIPTIVE_DIRECTION,
            inference=projection.DESCRIPTIVE_INFERENCE)
        self.args.protocol.write_bytes(encode(self.protocol))
        self.args.protocol_sha256=ranks.digest(self.args.protocol.read_bytes())
        self.summary['input_hashes']['protocol']=self.args.protocol_sha256
        self.summary['input_hashes']['prespecified_reporting']=ranks.object_hash(self.protocol['research_design']['reporting'])
        pins={k:self.summary['input_hashes'][k] for k in ranks.PINNED}
        self.rows_patch=patch.object(ranks,'ROWS',4);self.rows_patch.start();self.addCleanup(self.rows_patch.stop)
        self.pins_patch=patch.object(ranks,'PINNED',pins);self.pins_patch.start();self.addCleanup(self.pins_patch.stop)
        for cell,attempts in self.summary['attempts'].items():
            for item in attempts:
                item.update(cell_status='complete',artifact_names=['started.json','COMPLETE.json'],retry_review=None,retry_review_binding_verified=False)
                item['input_hashes']['COMPLETE.json']='c'*64
                for method in item['methods'].values():
                    method['rank_check'].update(ranks_dtype='float64',midrank_step=0.5)
        for identity,item in self.summary['observations'].items():
            item.update(attempt_count=1,eligible_completions=1,selection_rule=projection.SELECTION_RULE)
        self.sync_per_seed()
        self.write_inputs()

    def sync_per_seed(self):
        for key,group in self.summary['descriptive']['cell_scores'].items():
            v,t,method=key.split('/')
            group['per_seed']={str(s):copy.deepcopy(self.summary['observations'][f'{v}/{t}/{s}/{method}']) for s in ranks.SEEDS}

    def write_inputs(self):
        self.args.summary.write_bytes(encode(self.summary));self.args.summary_sha256=ranks.digest(self.args.summary.read_bytes())
        self.args.acceptance.write_bytes(encode({'summary_sha256':self.args.summary_sha256}))
        self.args.acceptance_sha256=ranks.digest(self.args.acceptance.read_bytes())
        recorded='/private/original/verifications/verify-9999'
        self.ticket={'kind':'verification','study':'model','preflight':False,'final_audit':False,'out':recorded+'/summary.json',
            'runner_bindings_sha256':self.summary['input_hashes']['runner_bindings'],
            'summary_bindings_sha256':self.summary['input_hashes']['private_bindings'],
            'argv':['private-command','--secret-path',recorded],'unit':'PRIVATE_PROCESS_SENTINEL'}
        self.ticket_data=encode(self.ticket);(self.inputs/'ticket.json').write_bytes(self.ticket_data)
        stage={'summary_path':self.ticket['out'],'summary_sha256':self.args.summary_sha256,'verification':recorded,
            'ticket_sha256':fixture_controller_ticket_hash(self.ticket)}
        self.args.stage_receipt.write_bytes(encode(stage));self.args.stage_receipt_sha256=ranks.digest(self.args.stage_receipt.read_bytes())
        selected=[];partitions=[]
        for identity,o in self.summary['observations'].items():
            v,t,seed,m=identity.split('/');cell=f'{v}/{t}/{seed}'
            attempts=self.summary['attempts'][cell]
            matches=[x for x in attempts if x['attempt']==o['selected_attempt']]
            if not matches: continue  # Malformed fixture mutations are rejected by the exporter.
            attempt=matches[0];method=attempt['methods'][m]
            selected.append({'identity':identity,'selected_attempt':o['selected_attempt'],'terminal_marker':attempt['terminal_marker'],
                'ranks_sha256':method['ranks_sha256']})
            partitions.append({'file':f'{v}--{t}/seed-{seed}/{m}.csv.gz','vision':v,'text':t,'seed':int(seed),'method':m,
                'selected_attempt':o['selected_attempt'],'cell_terminal':attempt['terminal_marker'],
                'rank_artifact':o['selected_attempt']+f'/{m}/ranks.pt','rank_artifact_sha256':method['ranks_sha256'],
                'canonical_ranks_by_coco_id_sha256':method['rank_check']['canonical_ranks_by_coco_id_sha256'],
                'rows':4,'bytes':123,'sha256':'f'*64})
        def row(field,value):
            return {'field':field,'frozen':value,'independent':value,'exactly_equal_at_reported_precision':True,
                'binary64_exactly_equal':True,'signed_difference_frozen_minus_independent':0.0,'absolute_difference':0.0,
                'independent_exact_fraction':str(Fraction(value)),'difference_exact_fraction':'0'}
        metric_rows=[row(f'{identity}/{metric}',o['metrics'][metric]) for identity,o in self.summary['observations'].items() for metric in projection.METRICS]
        aggregate_rows=[]
        for category in ('cell_scores','contrasts'):
            for name,group in self.summary['descriptive'][category].items():
                for seed,value in group['per_seed'].items():
                    for metric in projection.METRICS:
                        aggregate_rows.append(row(f'{category}/{name}/individual/{seed}/{metric}',value['metrics'][metric]))
                for statistic in ('mean','sample_sd'):
                    for metric in projection.METRICS:
                        aggregate_rows.append(row(f'{category}/{name}/{statistic}/{metric}',group[statistic][metric]))
        comparisons={'rank_metrics':metric_rows,'aggregates_from_saved_scalars':aggregate_rows,
            'aggregates_from_reconstructed_metrics':copy.deepcopy(aggregate_rows)}
        self.audit_report={'schema_version':1,'study':'model','status':'comparison_generated','scientific_acceptance':'not_assessed',
            'numeric_tolerance':None,'coverage':{'scheduled_cells':60,'method_observations':300,'five_seed_series':60,'contrast_series':103},
            'inputs':{'summary_sha256':self.args.summary_sha256,'acceptance_sha256':self.args.acceptance_sha256,
                'study_receipt_sha256':self.args.stage_receipt_sha256,'protocol_sha256':self.args.protocol_sha256,
                'ticket_sha256':ranks.digest(self.ticket_data),'ticket_canonical_sha256':stage['ticket_sha256'],'masks_sha256':None,
                'audit_code_sha256':ranks.digest(Path(audit.__file__).read_bytes()),'accepted_reader_input_hashes':self.summary['input_hashes'],
                'canonical_coco_ids_sha256':'e'*64},
            'selected_artifacts':selected,'comparisons':comparisons,
            'comparison_counts':{name:{'scalars':len(rows),'binary64_differences':0,'nonzero_exact_or_decimal_differences':0} for name,rows in comparisons.items()}}
        self.manifest={'schema_version':1,'status':'complete','protocol_id':self.summary['protocol_id'],
            'summary_sha256':self.args.summary_sha256,'accepted_reader_receipt_sha256':self.args.acceptance_sha256,
            'exporter_sha256':ranks.digest(Path(ranks.__file__).read_bytes()),'reader_input_hashes':self.summary['input_hashes'],
            'partition_count':300,'rows_per_partition':4,'columns':list(ranks.HEADER),'files':partitions}
        self.write_reports()

    def write_reports(self):
        self.args.audit.write_bytes(encode(self.audit_report));self.args.audit_sha256=ranks.digest(self.args.audit.read_bytes())
        self.args.rank_manifest.write_bytes(encode(self.manifest));self.args.rank_manifest_sha256=ranks.digest(self.args.rank_manifest.read_bytes())

    def run_export(self):
        # Guard the central requirement: metadata presentation must never read tensors.
        with patch.object(ranks.torch,'load',side_effect=AssertionError('Tensor load forbidden')):
            result=projection.export_summary(self.args)
        return result,json.loads(self.args.out.read_text())

    def test_exact_scalar_contrast_and_complete_coverage(self):
        result,output=self.run_export()
        self.assertEqual(result['status'],'projection_prepared')
        self.assertFalse(result['publication_authorized']);self.assertEqual(result['scientific_acceptance'],'not_assessed')
        self.assertIsNone(output['publication_review_reference'])
        self.assertEqual(len(output['observations']),300)
        self.assertEqual(len(output['descriptive']['cell_scores']),60)
        self.assertEqual(len(output['descriptive']['contrasts']),103)
        self.assertEqual(output['descriptive']['direction'],self.summary['descriptive']['direction'])
        self.assertEqual(output['descriptive']['inference'],self.summary['descriptive']['inference'])
        self.assertEqual(output['scope']['paired_control_extra_information'],
                         self.protocol['research_design']['paired_controls']['extra_information'])
        self.assertEqual(output['scope']['interpretation_unit'],self.protocol['research_design']['interpretation_unit'])
        for key,o in output['observations'].items():
            self.assertEqual(o['metrics'],self.summary['observations'][key]['metrics'])
        for category in ('cell_scores','contrasts'):
            for key,group in output['descriptive'][category].items():
                original=self.summary['descriptive'][category][key]
                for field in ('status','required_seeds','available_seeds','ddof','mean','sample_sd'):
                    self.assertEqual(group[field],original[field])
                if category=='contrasts':self.assertEqual(group['terms'],original['terms'])
                for seed,value in group['per_seed'].items():
                    self.assertEqual(value['metrics'],original['per_seed'][seed]['metrics'])
        self.assertEqual(output['history_counts']['attempts'],60)
        self.assertNotIn('/private/original',self.args.out.read_text())
        self.assertNotIn('PRIVATE_PROCESS_SENTINEL',self.args.out.read_text())
        self.assertNotIn('validation_image_ids',self.args.out.read_text())

    def test_controller_ticket_known_answer_and_separate_compact_hash(self):
        for ticket in (audit_fixture.KNOWN_CONTROLLER_TICKET,
                       dict(reversed(list(audit_fixture.KNOWN_CONTROLLER_TICKET.items())))):
            self.assertEqual(projection.controller_ticket_hash(ticket), audit_fixture.KNOWN_CONTROLLER_TICKET_SHA256)
            self.assertEqual(projection.object_hash(ticket), ranks.object_hash(ticket))
            self.assertNotEqual(projection.object_hash(ticket), audit_fixture.KNOWN_CONTROLLER_TICKET_SHA256)

    def test_reformatted_ticket_accepts_controller_hash_rejects_compact_hash(self):
        self.ticket = dict(reversed(list(self.ticket.items())))
        ticket_data = json.dumps(self.ticket, indent=4).encode()  # Preserve content, change bytes and key order.
        (self.inputs/'ticket.json').write_bytes(ticket_data)
        stage = json.loads(self.args.stage_receipt.read_text())
        canonical = fixture_controller_ticket_hash(self.ticket)
        for accepted_hash in (ranks.object_hash(self.ticket), canonical):
            stage['ticket_sha256'] = accepted_hash
            self.args.stage_receipt.write_bytes(encode(stage))
            self.args.stage_receipt_sha256 = ranks.digest(self.args.stage_receipt.read_bytes())
            self.audit_report['inputs'].update(study_receipt_sha256=self.args.stage_receipt_sha256,
                ticket_sha256=ranks.digest(ticket_data), ticket_canonical_sha256=accepted_hash)
            self.write_reports()
            if accepted_hash != canonical:
                with self.assertRaisesRegex(ValueError, 'ticket'): self.run_export()
                self.assertFalse(self.args.out.exists())
            else:
                result, _ = self.run_export()
                self.assertEqual(result['status'], 'projection_prepared')

    def test_failed_history_and_private_detail_omission(self):
        cell=f'dino/glove/{ranks.SEEDS[0]}';old=self.summary['attempts'][cell][0];new=copy.deepcopy(old)
        old.update(terminal_marker='FAILED.json',cell_status='failed')
        old['input_hashes'].pop('COMPLETE.json');old['input_hashes']['FAILED.json']='a'*64
        old['methods']['linear_shuffled'].update(status='failed',eligible=False,diagnostic_status='failed',
            error_type='PRIVATE_PROCESS_SENTINEL',linear_pinv_diagnostic={'singular_values':[999.123,888.456],
                'stderr':'/home/private/PRIVATE_PROCESS_SENTINEL'})
        old['private_context']={'pid':998899,'command':'PRIVATE_PROCESS_SENTINEL'}
        new['attempt']=new['attempt'].replace('attempt-01','attempt-02')
        new['retry_review_binding_verified']=True
        new['retry_review']={'reviewed':True,'source_or_numerics_changed':False,'reason_kind':'external_execution_failure',
            'previous_attempt':'attempt-01','reason':'Failure with /home/private/PRIVATE_PROCESS_SENTINEL pid998899',
            'previous_artifacts':{'/home/private/PRIVATE_PROCESS_SENTINEL':'a'*64}}
        new['input_hashes']['retry-review.json']='b'*64
        self.summary['attempts'][cell]=[old,new]
        for method in ranks.METHODS:
            o=self.summary['observations'][cell+'/'+method]
            o.update(attempt_count=2,eligible_completions=1 if method=='linear_shuffled' else 2,
                selected_attempt=new['attempt'] if method=='linear_shuffled' else old['attempt'],
                repeat_agreement=None if method=='linear_shuffled' else True)
        self.sync_per_seed();self.write_inputs();_,output=self.run_export()
        history=output['attempt_history'][cell]
        self.assertEqual(output['history_counts']['attempts'],61)
        self.assertEqual(output['history_counts']['failed_terminal_attempts'],1)
        self.assertEqual(history[0]['terminal_marker'],'FAILED.json')
        self.assertEqual(history[0]['methods']['linear_shuffled']['status'],'failed')
        self.assertEqual(history[0]['methods']['linear_shuffled']['reported_metrics'],old['methods']['linear_shuffled']['reported_metrics'])
        self.assertEqual(history[1]['retry_review']['reason_text']['canonical_json_sha256'],ranks.object_hash(new['retry_review']['reason']))
        self.assertEqual(output['observations'][cell+'/rosetta']['cell_terminal'],'FAILED.json')
        text=self.args.out.read_text()
        for secret in ('PRIVATE_PROCESS_SENTINEL','998899','/home/private','singular_values','linear_pinv_diagnostic'):
            self.assertNotIn(secret,text)

    def test_reject_partial_missing_series_wrong_operands_and_history(self):
        original=copy.deepcopy(self.summary)
        changes=[lambda s:s.update(status='incomplete'),
            lambda s:s['descriptive'].update(direction='PRIVATE_PROCESS_SENTINEL'),
            lambda s:s['descriptive'].update(inference='Unrestricted publication approval.'),
            lambda s:s['descriptive']['contrasts'].pop(next(iter(s['descriptive']['contrasts']))),
            lambda s:s['descriptive']['cell_scores'].pop(next(iter(s['descriptive']['cell_scores']))),
            lambda s:next(iter(s['descriptive']['contrasts'].values()))['terms'][0].update(coefficient=-1),
            lambda s:next(iter(s['descriptive']['contrasts'].values()))['available_seeds'].pop(),
            lambda s:next(iter(s['observations'].values())).update(attempt_count=2),
            lambda s:next(iter(s['observations'].values())).update(selected_attempt='/home/private/unsafe')]
        for change in changes:
            with self.subTest(change=change):
                self.summary=copy.deepcopy(original);change(self.summary);self.write_inputs()
                with self.assertRaises((ValueError,KeyError)):projection.export_summary(self.args)
                self.assertFalse(self.args.out.exists())

    def test_reject_audit_manifest_and_receipt_mismatches(self):
        changes=[lambda:self.audit_report['inputs'].update(summary_sha256='0'*64),
            lambda:self.audit_report['comparisons']['rank_metrics'].pop(),
            lambda:self.audit_report['comparisons']['rank_metrics'][0].update(frozen=99.0),
            lambda:self.manifest['files'].pop(),
            lambda:self.manifest['files'][0].update(rank_artifact_sha256='0'*64),
            lambda:self.audit_report.update(scientific_acceptance='passed')]
        for change in changes:
            with self.subTest(change=change):
                self.write_inputs();change();self.write_reports()
                with self.assertRaises(ValueError):projection.export_summary(self.args)
        self.write_inputs();self.args.acceptance.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SHA-256'):projection.export_summary(self.args)
        self.write_inputs();self.ticket['preflight']=True
        ticket_data=encode(self.ticket);(self.inputs/'ticket.json').write_bytes(ticket_data)
        stage=json.loads(self.args.stage_receipt.read_text());stage['ticket_sha256']=fixture_controller_ticket_hash(self.ticket)
        self.args.stage_receipt.write_bytes(encode(stage));self.args.stage_receipt_sha256=ranks.digest(self.args.stage_receipt.read_bytes())
        with self.assertRaisesRegex(ValueError,'ticket'):projection.export_summary(self.args)
        self.write_inputs();(self.inputs/'ticket.json').unlink()
        with self.assertRaises(OSError):projection.export_summary(self.args)

    def test_changed_numeric_audit_differences_are_not_an_approval_rule(self):
        row=self.audit_report['comparisons']['rank_metrics'][0]
        row.update(independent=row['frozen']+0.125,binary64_exactly_equal=False,exactly_equal_at_reported_precision=False,
            signed_difference_frozen_minus_independent=-0.125,absolute_difference=0.125)
        self.audit_report['comparison_counts']['rank_metrics'].update(binary64_differences=1,nonzero_exact_or_decimal_differences=1)
        self.write_reports();_,output=self.run_export()
        self.assertEqual(output['independent_audit']['comparison_counts']['rank_metrics']['binary64_differences'],1)
        self.assertFalse(output['publication_authorized'])
        self.assertEqual(output['scientific_acceptance'],'not_assessed')
        self.assertEqual(output['observations'][next(iter(self.summary['observations']))]['metrics'],
                         self.summary['observations'][next(iter(self.summary['observations']))]['metrics'])

    def test_output_protection(self):
        self.args.out=self.inputs/'unsafe.json'
        with self.assertRaisesRegex(ValueError,'fresh output'):projection.export_summary(self.args)
        self.args.out=self.root/'existing.json';self.args.out.write_text('preserve')
        with self.assertRaisesRegex(ValueError,'fresh output'):projection.export_summary(self.args)
        self.assertEqual(self.args.out.read_text(),'preserve')
        self.args.out=self.root/'link.json';self.args.out.symlink_to(self.inputs/'missing.json')
        with self.assertRaisesRegex(ValueError,'fresh output'):projection.export_summary(self.args)


if __name__=='__main__':
    unittest.main()
