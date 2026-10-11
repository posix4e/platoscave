"""Artificial lifecycle fixtures only: no corpus, tensors, masks or fit imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import execution as e


class Backend:
    def __init__(self):
        self.units = {}; self.groups = {}; self.launches = []; self.lose_reply = False
    def gate(self):
        pass
    def launch(self, ticket):
        if ticket['unit'] in self.units:
            raise RuntimeError('Duplicate unit')
        self.launches.append(copy.deepcopy(ticket))
        group = '/' + ticket['unit']
        self.groups[group] = True
        self.units[ticket['unit']] = {'invocation_id': 'a' * 32, 'boot_id': 'boot-a', 'cgroup': group,
            'main_pid': 123, 'starttime_ticks': 99, 'active': 'active', 'sub': 'running', 'result': 'success', 'exec_code': '0', 'exec_status': '0'}
        if self.lose_reply:
            raise OSError('Disconnected after successful launch')
    def inspect(self, ticket):
        if ticket['unit'] not in self.units:
            raise e.Blocked('Unit missing')
        return copy.deepcopy(self.units[ticket['unit']])
    def populated(self, group):
        return self.groups.get(group, False)
    def end(self, ticket):
        unit = self.units[ticket['unit']]
        self.groups[unit['cgroup']] = False
        unit.update(main_pid=0, cgroup='', active='active', sub='exited', exec_code='1')


class Fixture:
    """An explicitly synthetic final-looking release to test metadata gates."""
    def __init__(self, root):
        self.root = root; self.files = {}; self.hashes = {}
        self.model = {'protocol_id': 'rosetta-model-controls-v2', 'status': 'frozen_before_model_control_fits',
            'seeds': e.SEEDS, 'vision_models': e.VISION, 'text_models': e.TEXT, 'methods': e.METHODS, 'code_sha256': {},
            'research_design': {'reporting': {'summarizer': {'sha256': '0' * 64}}}}
        self.source = {'protocol_id': 'rosetta-source-controls-v1', 'status': 'frozen_before_source_control_fits',
            'seeds': e.SEEDS, 'selection_seeds': {'Rv': [1000, 1001], 'Rp': [2000]}, 'code_sha256': {},
            'research_design': {'budget': {'status': 'frozen_before_source_control_masks_and_fits', 'fits': 30},
                                'reporting': {'implementation': {'sha256': '0' * 64}}}}
        self.put('launcher', Path(e.__file__).read_bytes(), raw=True)
        self.files['launcher'] = str(Path(e.__file__).resolve())
        for name in ('model_runner', 'source_runner', 'model_reader', 'source_reader'):
            self.put(name, b'# synthetic, deliberately not executable scientific code\n', raw=True)
        self.model['research_design']['reporting']['summarizer']['sha256'] = self.hashes['model_reader']
        self.source['research_design']['reporting']['implementation']['sha256'] = self.hashes['source_reader']
        self.put('model_protocol', self.model); self.put('source_protocol', self.source)
        self.features = {'status': 'frozen_before_model_control_fits', 'protocol_sha256': self.hashes['model_protocol'],
                         'features': {str(i): {} for i in range(14)}}
        names = ['O', 'C', 'D', 'Rv-1000', 'Rv-1001', 'Rp-2000']
        self.masks = {'protocol_sha256': self.hashes['source_protocol'], 'masks': {n: {'mask_sha256': str(i) * 64} for i, n in enumerate(names)}}
        self.put('features', self.features); self.put('masks', self.masks)
        self.schedule = e.derive_schedule(self.model, self.source, self.masks)
        self.put('schedule', self.schedule)
        self.put('prior_work', {'status': 'approved_before_execution', 'scope': 'synthetic'})
        self.resources = {'max_jobs': 1, 'classes': {j['id']: 'shared' for j in self.schedule['jobs']}, 'external_work_quiescent_receipt': 'prior_work'}
        self.put('resources', self.resources)
        runtime = root / 'python'; runtime.write_text('synthetic executable')
        cwd = root / 'source'; cwd.mkdir()
        baseline_results = {}
        for seed in e.SEEDS:
            path = root / f'baseline-{seed}.json'; e.write_new(path, {'synthetic': True})
            baseline_results[str(seed)] = str(path)
        rb = {'python_executable': str(runtime), 'source_root': str(cwd), 'baseline_results': baseline_results, 'baseline_replays': {}}
        self.put('model_bindings', rb); self.put('source_bindings', rb)
        self.lock = {'schema_version': 1, 'status': 'frozen_before_control_execution', 'failure_policy': 'stop_on_any_failure',
            'systemd_properties': e.PROPERTIES, 'studies': {'model': {'protocol': 'model_protocol', 'input': 'features'}, 'source': {'protocol': 'source_protocol', 'input': 'masks'}},
            'schedule': 'schedule', 'resources': 'resources', 'code': {**{n: n for n in ('launcher', 'model_runner', 'model_reader', 'source_runner', 'source_reader')}, 'dependencies': []},
            'environment': {'ATEN_CPU_CAPABILITY': 'avx2', 'MKL_CBWR': 'AVX2,STRICT', 'PYTHONNOUSERSITE': '1'}, 'files': self.hashes}
        self.bindings = {'files': self.files, 'python': str(runtime), 'cwd': str(cwd), 'registry_root': str(root / 'registry'), 'control_root': str(root / 'control')}
        for study in ('model', 'source'):
            self.bindings[study] = {'runner_bindings': self.files[study + '_bindings'], 'attempts_root': str(root / (study + '-cells')), 'private_trace_root': str(root / (study + '-private'))}
        self.save()
    def put(self, name, value, raw=False):
        path = self.root / (name + ('.py' if raw else '.json'))
        path.write_bytes(value if raw else e.encoded(value))
        self.files[name] = str(path); self.hashes[name] = e.digest(path)
    def save(self):
        self.lock_path = self.root / 'lock.json'; self.lock_path.write_bytes(e.encoded(self.lock))
        self.bindings_path = self.root / 'bindings.json'; self.bindings_path.write_bytes(e.encoded(self.bindings))
    def release(self):
        return e.Release(self.lock_path, e.digest(self.lock_path), self.bindings_path, e.digest(self.bindings_path))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.runtime_patch = patch.object(e, 'runtime_registry', return_value=self.root / 'runtime-registry'); self.runtime_patch.start()
        self.f = Fixture(self.root); self.r = self.f.release(); self.backend = Backend(); self.c = e.Controller(self.r, self.backend)
    def tearDown(self):
        self.runtime_patch.stop(); self.tmp.cleanup()
    def preflight_receipts(self):
        for study in ('model', 'source'):
            self.c.new_verification(study, [], preflight=True)
            directory = sorted((self.c.root / 'verifications').iterdir())[-1]
            self.reader_result(directory, {'status': 'incomplete', 'inventory_findings': [], 'baseline_replay_gate': {'status': 'unavailable'}})
            self.c.verification_tick()
    def launched(self):
        job = self.c.jobs[0]; self.c.launch_job(job)
        directory = self.c.directories(job)[0]
        return job, directory, e.read(directory / 'ticket.json')
    def summary(self, job, out, complete=True):
        relative = '/'.join(Path(out).parts[-3:])
        attempt = {'attempt': relative, 'terminal_marker': 'COMPLETE.json', 'terminal_manifest_verified': True,
                   'findings': [], 'methods': {m: {'eligible': complete} for m in e.METHODS}}
        return {'status': 'incomplete', 'input_hashes': {}, 'inventory_findings': [], 'baseline_replay_gate': {'status': 'unavailable'},
                'attempts': {job['key']: [attempt]},
                'observations': {job['key'] + '/' + m: {'status': 'available' if complete else 'unavailable'} for m in e.METHODS}}
    def reader_result(self, directory, summary):
        ticket = e.read(directory / 'ticket.json'); study = ticket['study']
        summary['input_hashes'] = {'protocol': self.r.hash_for(study, 'protocol'),
            'feature_lock' if study == 'model' else 'masks': self.r.hash_for(study, 'input'),
            'summarizer': self.r.lock['files'][study + '_reader'], 'private_bindings': ticket['summary_bindings_sha256'],
            'runner_bindings': ticket['runner_bindings_sha256']}
        e.write_new(ticket['out'], summary); self.backend.end(ticket)
        return ticket
    def test_schedule_contains_all_sixty_cells_and_declared_source_masks(self):
        jobs = self.r.schedule['jobs']
        self.assertEqual(len(jobs), 90); self.assertEqual(sum(j['study'] == 'model' for j in jobs), 60)
        self.assertEqual([j['seed'] for j in jobs[:5]], e.SEEDS)
        self.assertEqual([j['mask'] for j in jobs[60::5]], ['O', 'C', 'D', 'Rv-1000', 'Rv-1001', 'Rp-2000'])
    def test_draft_protocol_budget_or_missing_feature_cannot_dispatch(self):
        for field in ('protocol', 'budget', 'features'):
            with self.subTest(field=field):
                original = copy.deepcopy(self.f.model if field == 'protocol' else self.f.source if field == 'budget' else self.f.features)
                if field == 'protocol':
                    self.f.model['status'] = 'draft'; self.f.put('model_protocol', self.f.model)
                elif field == 'budget':
                    self.f.source['research_design']['budget']['status'] = 'proposed'; self.f.put('source_protocol', self.f.source)
                else:
                    self.f.features['features'].pop('0'); self.f.put('features', self.f.features)
                self.f.save()
                with self.assertRaises(e.Blocked): self.f.release()
                if field == 'protocol': self.f.model = original; self.f.put('model_protocol', original)
                elif field == 'budget': self.f.source = original; self.f.put('source_protocol', original)
                else: self.f.features = original; self.f.put('features', original)
                self.f.save()
    def test_schedule_omission_reorder_or_duplicate_rejected(self):
        for mutated in (self.f.schedule['jobs'][:-1], self.f.schedule['jobs'][::-1], self.f.schedule['jobs'] + [self.f.schedule['jobs'][0]]):
            self.f.put('schedule', {**self.f.schedule, 'jobs': mutated}); self.f.save()
            with self.assertRaises(e.Blocked): self.f.release()
    def test_mutated_bound_code_and_private_bindings_block(self):
        Path(self.f.files['source_reader']).write_text('changed')
        with self.assertRaises(e.Blocked): self.r.recheck()
        self.assertEqual(self.backend.launches, [])
    def test_fresh_runner_paths_are_not_precreated(self):
        job, directory, ticket = self.launched()
        self.assertFalse(Path(ticket['out']).exists()); self.assertFalse(Path(ticket['private_trace_dir']).exists())
        with self.assertRaises(e.Blocked): self.c.launch_job(job)
        self.assertEqual(len(self.backend.launches), 1)
    def test_controller_restart_reconnects_without_duplicate(self):
        self.preflight_receipts(); self.c.tick()
        restarted = e.Controller(self.r, self.backend)
        result = restarted.tick()
        self.assertEqual(result['counts']['running'], 1); self.assertEqual(len(self.backend.launches), 3)
    def test_lost_launch_reply_reconciles_existing_service(self):
        self.backend.lose_reply = True
        with self.assertRaises(e.Blocked): self.launched()
        self.backend.lose_reply = False
        self.assertEqual(list(self.c.states().values()).count('running'), 1)
        self.assertEqual(len(self.backend.launches), 1)
    def test_missing_unit_after_durable_intent_is_ambiguous_not_redispatched(self):
        job, directory, ticket = self.launched(); self.backend.units.clear()
        with self.assertRaises(e.Blocked): self.c.tick()
        self.assertEqual(len(self.backend.launches), 1)
    def test_controller_death_before_dispatch_preserves_reservation(self):
        job = self.c.jobs[0]; directory = self.c.root / 'jobs' / job['id'] / 'attempt-01'
        e.write_new(directory / 'ticket.json', {'synthetic': 'incomplete ticket'})
        with self.assertRaises((e.Blocked, KeyError)): self.c.tick()
        self.assertEqual(self.backend.launches, [])
    def test_changed_invocation_boot_or_reused_pid_blocks(self):
        job, directory, ticket = self.launched(); e.reconcile_service(directory, self.backend)
        initial = copy.deepcopy(self.backend.units[ticket['unit']])
        for field, value in [('invocation_id', 'b' * 32), ('boot_id', 'boot-b'), ('starttime_ticks', 100)]:
            self.backend.units[ticket['unit']][field] = value
            with self.assertRaises(e.Blocked): e.reconcile_service(directory, self.backend)
            self.backend.units[ticket['unit']] = copy.deepcopy(initial)
    def test_main_exit_does_not_release_descendant_cgroup(self):
        job, directory, ticket = self.launched(); e.reconcile_service(directory, self.backend)
        unit = self.backend.units[ticket['unit']]; unit.update(main_pid=0, sub='exited')
        self.assertEqual(e.reconcile_service(directory, self.backend)[0], 'running')
        self.backend.end(ticket)
        self.assertEqual(e.reconcile_service(directory, self.backend)[0], 'quiescent')
        self.assertTrue((directory / 'cgroup.json').exists())
    def test_changed_dispatch_ticket_is_rejected(self):
        job, directory, ticket = self.launched(); ticket['argv'].append('--changed')
        (directory / 'ticket.json').write_bytes(e.encoded(ticket))
        with self.assertRaises(e.Blocked): e.reconcile_service(directory, self.backend)
    def test_terminal_partial_or_conflict_never_complete(self):
        path = self.root / 'terminal'; e.write_new(path / 'COMPLETE.json', {})
        for name in ('FAILED.json', *e.PARTIALS):
            (path / name).write_text('{}')
            with self.assertRaises(e.Blocked): e.complete_marker(path)
            (path / name).unlink()
    def test_unknown_empty_cell_and_untracked_attempt_are_rejected(self):
        root = Path(self.r.bindings['model']['attempts_root']); (root / 'unknown').mkdir()
        with self.assertRaises(e.Blocked): self.c.inventory()
        (root / 'unknown').rmdir()
        (root / 'dinov2--mpnet' / f'seed-{e.SEEDS[0]}' / 'attempt-01').mkdir(parents=True)
        with self.assertRaises(e.Blocked): self.c.inventory()
    def test_both_study_preflights_precede_any_fit(self):
        result = self.c.tick()
        self.assertEqual(result['status'], 'preflight')
        self.assertEqual(self.backend.launches[0]['kind'], 'verification')
        self.assertEqual(self.backend.launches[0]['study'], 'model')
    def test_reader_death_reconnects_and_waits_for_global_quiescence(self):
        self.c.new_verification('model', [], preflight=True)
        self.assertTrue(e.Controller(self.r, self.backend).verification_tick())
        self.assertEqual(len(self.backend.launches), 1)
        # A running scientific unit suppresses all reader dispatch/acceptance.
        self.launched()
        result = self.c.tick(); self.assertEqual(result['counts']['running'], 1)
        self.assertEqual(len(self.backend.launches), 2)
    def test_exit_zero_without_valid_artifacts_is_not_success(self):
        job, directory, ticket = self.launched(); self.backend.end(ticket)
        self.c.new_verification('model', [job['id']])
        verification = sorted((self.c.root / 'verifications').iterdir())[-1]
        summary = self.summary(job, ticket['out'], complete=False)
        self.reader_result(verification, summary)
        self.c.verification_tick()
        self.assertEqual(e.read(directory / 'verified.json')['status'], 'failed')
        self.assertEqual(self.c.tick()['status'], 'blocked')
    def test_failed_service_retains_valid_artifacts_but_blocks_advancement(self):
        job = self.c.jobs[0]; directory, ticket = self.complete_fixture(job)
        self.backend.units[ticket['unit']].update(result='exit-code', exec_status='7')
        self.c.new_verification('model', [job['id']])
        verification = sorted((self.c.root / 'verifications').iterdir())[-1]
        self.reader_result(verification, self.combined_summary([(job, ticket)]))
        self.c.verification_tick()
        record = e.read(directory / 'verified.json')
        self.assertTrue(record['artifact_eligible']); self.assertFalse(record['service_success'])
        self.assertEqual(record['status'], 'failed'); self.assertEqual(record['exec_main_status'], '7')
        self.assertEqual(self.c.tick()['status'], 'blocked')
        self.assertTrue(Path(ticket['out'], 'COMPLETE.json').exists())
    def test_user_site_and_public_environment_path_gates(self):
        original = copy.deepcopy(self.f.lock['environment'])
        for environment in ({'ATEN_CPU_CAPABILITY': 'avx2', 'MKL_CBWR': 'AVX2,STRICT'}, {**original, 'HOME': '/private/user'}):
            self.f.lock['environment'] = environment; self.f.save()
            with self.assertRaises(e.Blocked): self.f.release()

    def test_partial_reader_output_and_wrong_hash_rejected(self):
        self.c.new_verification('model', [], preflight=True)
        directory = next((self.c.root / 'verifications').iterdir())
        ticket = self.reader_result(directory, {'inventory_findings': [], 'baseline_replay_gate': {'status': 'unavailable'}})
        Path(ticket['out'] + '.partial').write_text('interrupted')
        with self.assertRaises(e.Blocked): self.c.verification_tick()
        self.assertTrue((directory / 'rejected.json').exists())
    def test_five_reference_gate_cannot_be_opened_with_four(self):
        summary = {'baseline_replay_gate': {'status': 'satisfied'}}
        with self.assertRaises(e.Blocked): self.c.make_gate(summary, self.f.lock_path)
        with self.assertRaises(FileNotFoundError): self.c.launch_job(self.c.jobs[5])
        self.assertEqual(self.backend.launches, [])
    def test_max_two_requires_amendment_and_full_class_table(self):
        self.f.resources['max_jobs'] = 2; self.f.put('resources', self.f.resources); self.f.save()
        with self.assertRaises((e.Blocked, KeyError)): self.f.release()
        self.f.put('amendment', {'status': 'approved_before_execution'}); self.f.resources['amendment'] = 'amendment'
        self.f.put('resources', self.f.resources); self.f.save()
        self.assertEqual(self.f.release().resources['max_jobs'], 2)
    def test_serial_two_and_exclusive_admission_never_skip(self):
        jobs = self.c.jobs; states = {j['id']: 'pending' for j in jobs}; resources = copy.deepcopy(self.r.resources)
        self.assertEqual(len(e.candidate_batch(jobs, states, resources)), 1)
        resources['max_jobs'] = 2; self.assertEqual(len(e.candidate_batch(jobs, states, resources)), 2)
        resources['classes'][jobs[0]['id']] = 'exclusive'; self.assertEqual(len(e.candidate_batch(jobs, states, resources)), 1)
        states[jobs[0]['id']] = 'running'; self.assertEqual(e.candidate_batch(jobs, states, resources), [])
        states[jobs[0]['id']] = 'failed'; self.assertEqual(e.candidate_batch(jobs, states, resources), [])
    def test_batch_drains_before_reader_and_does_not_cross_stage(self):
        jobs = self.c.jobs; states = {j['id']: 'pending' for j in jobs}; resources = copy.deepcopy(self.r.resources); resources['max_jobs'] = 2
        for j in jobs[:4]: states[j['id']] = 'complete'
        self.assertEqual(len(e.candidate_batch(jobs, states, resources)), 1)
        states[jobs[4]['id']] = 'quiescent'; self.assertEqual(e.candidate_batch(jobs, states, resources), [])
    def test_retry_requires_actual_predecessor_hash_and_review(self):
        job, directory, ticket = self.launched(); out = Path(ticket['out']); e.write_new(out / 'started.json', {'synthetic': True})
        review = {'schema_version': 1, 'protocol_id': self.r.model['protocol_id'], 'protocol_sha256': self.r.hash_for('model', 'protocol'),
                  'feature_lock_sha256': self.r.hash_for('model', 'input'), 'vision': job['vision'], 'text': job['text'], 'seed': job['seed'],
                  'previous_attempt': 'attempt-01', 'previous_artifacts': {'started.json': e.digest(out / 'started.json')},
                  'reviewed': True, 'source_or_numerics_changed': False, 'reason_kind': 'external_execution_failure', 'reason': 'Synthetic reviewed interrupted process.'}
        e.review_retry(review, job, self.r, ticket, 1)
        for change in ({'reviewed': False}, {'source_or_numerics_changed': True}, {'previous_artifacts': {'started.json': '0' * 64}}):
            with self.assertRaises(e.Blocked): e.review_retry({**review, **change}, job, self.r, ticket, 1)
        e.write_new(out / 'COMPLETE.json.partial', {})
        with self.assertRaises(e.Blocked): e.review_retry(review, job, self.r, ticket, 1)
    def test_constructor_empty_output_cannot_be_fabricated_as_retry(self):
        job, directory, ticket = self.launched(); Path(ticket['out']).mkdir(parents=True)
        with self.assertRaises(e.Blocked): e.review_retry({}, job, self.r, ticket, 1)
    def complete_fixture(self, job):
        self.c.launch_job(job)
        directory = self.c.directories(job)[-1]; ticket = e.read(directory / 'ticket.json')
        out = Path(ticket['out']); e.write_new(out / 'started.json', {'synthetic': True})
        e.write_new(out / 'COMPLETE.json', {'files': {'started.json': e.digest(out / 'started.json')}})
        self.backend.end(ticket)
        return directory, ticket
    def combined_summary(self, jobs_and_tickets):
        result = {'status': 'incomplete', 'inventory_findings': [], 'baseline_replay_gate': {'status': 'unavailable'}, 'attempts': {}, 'observations': {}}
        for job, ticket in jobs_and_tickets:
            part = self.summary(job, ticket['out'])
            result['attempts'].update(part['attempts']); result['observations'].update(part['observations'])
        return result
    def test_older_accepted_result_invalid_in_new_summary_blocks_current(self):
        first, second = self.c.jobs[:2]
        d1, t1 = self.complete_fixture(first)
        self.c.new_verification('model', [first['id']])
        v1 = sorted((self.c.root / 'verifications').iterdir())[-1]
        self.reader_result(v1, self.combined_summary([(first, t1)])); self.c.verification_tick()
        d2, t2 = self.complete_fixture(second)
        self.c.new_verification('model', [second['id']])
        v2 = sorted((self.c.root / 'verifications').iterdir())[-1]
        summary = self.combined_summary([(first, t1), (second, t2)])
        summary['observations'][first['key'] + '/rosetta']['status'] = 'unavailable'
        self.reader_result(v2, summary)
        with self.assertRaises(e.Blocked): self.c.verification_tick()
        self.assertTrue((v2 / 'rejected.json').exists()); self.assertFalse((d2 / 'verified.json').exists())
    def test_crash_between_target_receipts_recovers_without_duplicate_fit(self):
        first, second = self.c.jobs[:2]
        d1, t1 = self.complete_fixture(first); d2, t2 = self.complete_fixture(second)
        self.c.new_verification('model', [first['id'], second['id']])
        v = sorted((self.c.root / 'verifications').iterdir())[-1]
        self.reader_result(v, self.combined_summary([(first, t1), (second, t2)]))
        original = e.write_or_check
        def crash(path, value):
            original(path, value)
            if path == d1 / 'verified.json': raise KeyboardInterrupt('Synthetic controller death')
        with patch.object(e, 'write_or_check', side_effect=crash):
            with self.assertRaises(KeyboardInterrupt): self.c.verification_tick()
        self.assertTrue((d1 / 'verified.json').exists()); self.assertFalse((v / 'accepted.json').exists())
        e.Controller(self.r, self.backend).verification_tick()
        self.assertTrue((d2 / 'verified.json').exists()); self.assertTrue((v / 'accepted.json').exists())
        self.assertEqual(len(self.backend.launches), 3)
    def test_final_manifest_catches_changed_bytes_with_unchanged_complete(self):
        job = self.c.jobs[0]; directory, ticket = self.complete_fixture(job)
        e.write_new(directory / 'verified.json', {'status': 'complete'})
        self.c.jobs = [job]
        for study in ('model', 'source'):
            root = Path(self.r.bindings[study]['attempts_root']).resolve()
            paths = [Path(ticket['out'])] if study == 'model' else []
            attempts = [{'attempt': str(p.relative_to(root)), 'artifact_names': sorted(x.name for x in p.iterdir()),
                         'input_hashes': {x.name: e.digest(x) for x in p.iterdir()}} for p in paths]
            summary_path = self.root / (study + '-final-synthetic.json')
            e.write_new(summary_path, {'attempts': {'synthetic': attempts}})
            e.write_new(self.c.root / ('final-audit-' + study + '.json'), {'summary_path': str(summary_path), 'summary_sha256': e.digest(summary_path)})
        self.c.final_artifact_check()
        Path(ticket['out'], 'started.json').write_text('changed')
        with self.assertRaises(e.Blocked): self.c.final_artifact_check()
    def test_failed_predecessor_changed_after_final_reader_is_rejected(self):
        root = self.root / 'history'; predecessor = root / 'cell' / 'seed-1' / 'attempt-01'
        e.write_new(predecessor / 'started.json', {'synthetic': True})
        e.write_new(predecessor / 'FAILED.json', {'synthetic': True})
        observation = {'attempt': str(predecessor.relative_to(root)), 'artifact_names': sorted(p.name for p in predecessor.iterdir()),
                       'input_hashes': {p.name: e.digest(p) for p in predecessor.iterdir()}}
        summary = {'attempts': {'synthetic': [observation]}}
        e.recheck_summary_history(summary, root, {predecessor.resolve()})
        (predecessor / 'FAILED.json').write_text('mutated failed predecessor')
        with self.assertRaises(e.Blocked): e.recheck_summary_history(summary, root, {predecessor.resolve()})

    def test_accepted_and_rejected_verifier_conflict_never_recovered(self):
        self.preflight_receipts()
        directory = sorted((self.c.root / 'verifications').iterdir())[-1]
        e.write_new(directory / 'rejected.json', {'synthetic': 'late publication failure'})
        with self.assertRaises(e.Blocked): self.c.verification_tick()
    def test_accepted_verifier_changed_invocation_blocks_restart(self):
        self.preflight_receipts()
        directory = sorted((self.c.root / 'verifications').iterdir())[-1]
        ticket = e.read(directory / 'ticket.json')
        self.backend.units[ticket['unit']]['invocation_id'] = 'f' * 32
        with self.assertRaises(e.Blocked): self.c.verification_tick()
    def test_max_two_controller_drains_both_before_reader(self):
        self.preflight_receipts(); self.r.resources['max_jobs'] = 2
        self.c.tick(); scientific = [t for t in self.backend.launches if t['kind'] == 'fit']
        self.assertEqual(len(scientific), 2)
        self.backend.end(scientific[0]); self.c.tick()
        self.assertEqual(len(self.backend.launches), 4)
        self.backend.end(scientific[1]); self.c.tick()
        self.assertEqual(len(self.backend.launches), 5)
        self.assertEqual(self.backend.launches[-1]['targets'], [j['id'] for j in self.c.jobs[:2]])
    def test_forged_preflight_existence_is_not_authorization(self):
        for study in ('model', 'source'):
            e.write_new(self.c.root / ('preflight-' + study + '.json'), {})
        with self.assertRaises((e.Blocked, KeyError)): self.c.tick()
        self.assertEqual(self.backend.launches, [])

    def test_registry_excludes_second_controller_or_other_execution(self):
        with e.controller_lock(self.r):
            with self.assertRaises(e.Blocked):
                with e.controller_lock(self.r): pass
        old = self.r.sha; self.r.sha = 'b' * 64
        with self.assertRaises(e.Blocked):
            with e.controller_lock(self.r): pass
        self.r.sha = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
