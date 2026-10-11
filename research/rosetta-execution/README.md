# Execute the complete control studies

This launcher is preparation code. No execution release is frozen here, and these files authorize no fits. The source-mask budget, model features, scientific protocols, resource policy and their hashes must all be final before dispatch. Feature extraction is an earlier, separate phase.

The schedule contains all 60 model cell/seed jobs, each running the five fixed methods. The five DINOv2–MPNet reference cells run first, followed by the other 55 cells. Source jobs then follow O, C, D, every declared Rv mask and every declared Rp mask, with the original five algorithm seeds within each mask. Ten Rv and ten Rp masks would give 115 source jobs; the launcher derives the number from the final explicit lists. Nothing may be appended to this execution later.

## Release and private bindings

The final public execution lock has these fields:

- `schema_version: 1`, `status: "frozen_before_control_execution"`, and `failure_policy: "stop_on_any_failure"`.
- `files`: logical file keys mapped to exact SHA-256 values. Include the two final protocols, 14-entry feature lock, full mask bundle, ordered schedule, resource policy, approved prior-work receipt, scientific code closure and the two private runner-binding file hashes. Private paths stay out of this lock.
- `studies.model` and `studies.source`: each has `protocol` and `input`, naming logical file keys. The model input is the feature lock; the source input is the mask bundle.
- `schedule` and `resources`: logical file keys.
- `code`: logical file keys under `launcher`, `model_runner`, `model_reader`, `source_runner`, `source_reader`, plus a `dependencies` list. The closure must include every helper hash declared in both scientific protocols and the exact reporting helpers.
- `environment`: explicit values for an `env -i` command. It must include `ATEN_CPU_CAPABILITY=avx2`, `MKL_CBWR=AVX2,STRICT`, and `PYTHONNOUSERSITE=1`. Only the path-free runtime settings allowed in `Release` are accepted. Any extra numerical settings require review before freezing them.
- `systemd_properties`: exactly the `PROPERTIES` object in `execution.py`: `Type=exec`, `ExitType=cgroup`, `RemainAfterExit=yes`, `Restart=no`, with runtime, start and stop timeouts set to infinity.

The model protocol and feature lock must have status `frozen_before_model_control_fits`. The source protocol must have status `frozen_before_source_control_fits`; its budget must have status `frozen_before_source_control_masks_and_fits` and a fit count matching the complete declared mask schedule. Both inputs must bind their final protocol hash.

The resource policy declares `max_jobs` and a `classes` entry for every job ID, with values `shared` or `exclusive`. The published [scheduling amendment](scheduling-amendment.json) and [resource policy](resource-policy.json) permit at most two adjacent shared jobs. Any model cell using MoCo or ELMo is exclusive. The full table contains 30 exclusive model jobs and 145 shared jobs (30 model jobs and all 115 source jobs). It preserves the fixed order and every prescribed job. Per-process numerical settings remain unchanged; no twofold speedup is assumed.

This operational amendment supersedes only the serial concurrency limit in the preserved source resource decision. It does not revise the frozen source protocol, mask budget or prepared masks. The 20 GiB allowance per shared job plus 16 GiB guest reserve is a planning estimate, not a measured upper bound. The all-14-feature preflight validates one memory-mapped tensor at a time with 512-row checks; it does not retain all fourteen tensors together. Recheck capacity before final execution, and delay admission if capacity is insufficient.

The final execution lock must bind both files. The policy names an `amendment` receipt with status `approved_before_execution` and an `external_work_quiescent_receipt` with that status. The latter has not yet been issued: the original baseline and feature extraction must finish first. It is reviewed evidence that earlier work is quiescent; the launcher does not discover unrelated processes from such a receipt. Publishing this resource policy authorizes no dispatch on its own.

Private immutable bindings contain:

```text
files: same logical keys as the public lock -> absolute file paths
python: exact runtime executable
cwd: exact upstream source root
registry_root: persistent resource registry directory
control_root: new private controller directory
model/source:
  runner_bindings: immutable scientific runner-binding JSON
  attempts_root: study's cells directory
  private_trace_root: separate private traceback root
```

The runner-binding files must also appear as logical keys `model_bindings` and `source_bindings` in `files`. Their runtime/source paths must agree with `python` and `cwd`. The control directory, both attempt trees and both traceback trees must be separate. A fixed per-user systemd runtime registry additionally prevents two configurations from choosing different locks and spending the same resource budget. Registry ownership is retained; another release requires explicit retirement or resolution, never automatic replacement.

The generated schedule is byte-bound by the release. It can be prepared, without fitting, with:

```sh
python execution.py schedule --model-protocol MODEL_PROTOCOL --source-protocol SOURCE_PROTOCOL --masks MASK_BUNDLE --out NEW_SCHEDULE
```

This command neither freezes its inputs nor validates a scientific release. The later `validate` command checks the complete final release and performs no dispatch:

```sh
python execution.py validate --lock EXECUTION_LOCK --lock-sha256 LOCK_SHA --bindings PRIVATE_BINDINGS --bindings-sha256 BINDINGS_SHA
```

Use the same four arguments with `tick` for one reconciliation/admission pass, or `run` for repeated passes every 30 seconds. `run` stops on completion, observed failure or an unresolved error. Run the controller in a separately retained service when it must survive an SSH session. A controller restart reconnects to already dispatched jobs; it does not create replacement fits.

## Dispatch and verification

Before the first fit, the unchanged full-study readers run in their exact runtime against both empty attempt trees. Each fit then runs in a retained systemd user service with a durable ticket written before dispatch. User lingering must already be enabled. The ticket fixes argv, working directory, environment, private binding hashes and fresh runner-owned output/trace paths. Logs and controller records live outside those paths.

The controller records and checks systemd InvocationID, boot identity, observed main PID/start time and cgroup. A main process exiting while descendants remain does not free a slot. A missing unit after dispatch intent, changed identity or uncertain launch remains blocked. Units and claims are never cleaned up or reused automatically.

Every active fit must become quiescent before a strict reader starts. At a prospectively approved maximum of two, fits run in fixed-order batches; both finish before verification. No smaller job skips an exclusive one. Readers also run as retained services, so controller death cannot leave their execution untracked.

The existing readers check the full attempt tree, exact source/runtime/data bindings, saved row orders, ranks and metrics. A new summary must preserve the eligibility of every previously accepted job in that study. The five reference cells must each have one verified complete attempt and agree with the untouched original results before expansion. The launcher writes a derived replay gate and immutable expanded-stage runner-binding snapshot. All model jobs must be complete before source jobs begin.

Scientific artifacts and process outcomes are reported separately. Valid artifacts from a service with a failing result or exit status remain available for inspection, but advancement stops. Success requires both strict scientific verification and a normal successful service exit. Exit zero alone never establishes scientific completion.

After all jobs finish, both complete studies receive another full reader audit. The launcher then rechecks all reader-bound history hashes and file inventories, including failed predecessors, and every accepted complete manifest. Only then may execution status become `complete`.

## Failures and explicit retries

All attempts, reader outputs, failures and unresolved dispatches remain in place. Any observed failure stops new dispatch; already active work is retained and reconciled, never killed. There is no timeout kill or automatic retry.

An existing approved retry record can be supplied explicitly:

```sh
python execution.py retry --lock EXECUTION_LOCK --lock-sha256 LOCK_SHA --bindings PRIVATE_BINDINGS --bindings-sha256 BINDINGS_SHA --job JOB_ID --retry-review APPROVED_REVIEW --retry-review-sha256 REVIEW_SHA
```

The controller verifies stopped predecessors, contiguous attempt counters and their actual FAILED/start hashes. It passes the unchanged review through the runners' existing preflight hook. It never writes a review reason or grants approval. A model retry repeats the whole five-method cell; the reader's exact repetition-agreement rule remains in force.

Missing start receipts, constructor remnants, partial or conflicting terminal markers, multiple complete attempts, and a valid COMPLETE accompanied by a failed service require external resolution. The retry command cannot erase those records or manufacture a predecessor. Source or numerical changes require a separately reviewed protocol/release.

## Validation scope

`test_execution.py` contains artificial metadata, fake service lifecycles and controller interruption fixtures. It imports no fitting or tensor API and uses no experimental observations. Run it with:

```sh
python -m unittest discover -s research/rosetta-execution -p 'test_execution.py' -v
```

Separate live systemd probe receipts must identify the exact backend revision tested. They establish those operational cases, not a real-input preflight, a complete execution trial, or scientific replication. No scientific runner, protocol, baseline or numerical code was changed by this launcher.
