# Five original runs, with differences retained

All five predetermined runs of the published DINOv2 B/14 mean × MPNet COCO configuration finished, including retrieval and three classification evaluations. The original supervisor completed on 11 October 2026 at 04:15:09 UTC, with unchanged source, inputs and release files. The baseline processes have exited.

The [untouched comparison](comparison.json) reports every seed and endpoint. Retrieval mean is 0.043206930600025444 versus the frozen reference 0.042989911199083145; sample SD is 0.04840590027316027 versus 0.0480892962510061. Both miss our fixed absolute tolerance of 1e-10. Both still print as 0.04 ± 0.05. The available named seed matches exactly.

CIFAR-10 top-1 is 70.55080056% ± 10.20101765 percentage points, which prints as **71 ± 10**, versus the paper’s **70 ± 10**. CIFAR-100 top-5 prints as 46 ± 14 and ImageNet-100 top-5 as 45 ± 11, matching the paper at its displayed precision. These are means and sample standard deviations over all five fixed seeds.

This is close agreement, with discrepancies that remain under investigation. It does not establish exact reproduction of every available reference or disprove Rosetta. Four individual author retrieval scores and unrounded classification references are unavailable in the frozen record. No seed or criterion has been changed.

The [status](status.json), [assessment](assessment.json) and [baseline evidence lock](baseline-results-lock.json) distinguish execution completion from reference agreement. The lock authorizes no new experiment. The original frozen release and earlier partial records remain unchanged. Raw host metadata and operational logs remain private; public records bind their hashes.

The model panel, full source-removal study, and prospective collection with fresh unpaired fitting and separate final evaluation remain required later stages.

The [discrepancy review](discrepancy-review.json) found no demonstrated transcription, aggregation, local source-integrity or explicit RNG-order error. The cause remains unknown. The existing model-panel protocol requires investigation, not exact agreement with the authors; this review satisfies that condition. Publication of complete final locks is still required before added inference, and all five reference replays must match our preserved baseline within 1e-10 before expanded fitting.
