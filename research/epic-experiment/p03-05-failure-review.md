# P03_05 acquisition failure: bounded diagnosis

On 2026-10-10 the live acquisition reported `InvalidDataError` after 39 of 88 requested frames for P03_05. It had reached UID 6787 slot 2 at 96.7800167 seconds. The next requested frame was slot 3 at 98.3175 seconds, inside the original action window of 92.91–99.09 seconds.

An isolated reproduction read only that same four-frame action, once with the current HTTP keepalive options and once with the original baseline options. Neither run touched production frames or reports. Both completed in about 11 seconds and reproduced the three already acquired frames byte-for-byte.

Both modes recovered slot 3 identically:

- Actual timestamp: `98.31488333333333` seconds.
- Source PTS: `5898893`.
- JPEG SHA256: `564d5ce9b1f272c3fbcd1c039bac45359dd60f3d6f74d9bf890cbef5cadbc47c`.
- RGB SHA256: `ff3c5164af08eeffc6e22899b3293a6abece1dfcd4204472d5adeef400f114b9`.

The missing target is therefore decodable. The keepalive reproduction's FFmpeg trace showed an HTTP offset mismatch (`expected 480, got 0`) and then a successful reconnection. The baseline trace had no HTTP error. This is evidence of transport instability, but the original failure did not retain a detailed trace, so its exact cause is unproven.

After the live acquisition finishes, retry the failed video using the unchanged current acquirer. A fresh connection recovered this action; the full 88-frame retry remains unverified. If the failure recurs, preserve its trace before choosing a transport fallback. The production report remained byte-identical throughout this diagnosis.

Evidence: `p03-05-failure-diagnostic.json` records the original report snapshot, hashes, both decoded-frame results and FFmpeg traces. HTTP cookie values were redacted for publication. Reproduction code: `diagnose_p03_05.py`, using `benchmark_http_seek.py` without changing acquisition code.
