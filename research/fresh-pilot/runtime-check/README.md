# Actual-model runtime check

This is a software check on the research VM, separate from the prospective
human experiment. It used an artificial pixel gradient held in memory and the
fixed text `This is a software test only.` No human packet was sealed, no
research photographs or captions were created, and no mapping was fitted.

The unchanged frozen extractor loaded the actual pinned DINO model on the H100,
produced finite normalized features, and gave byte-identical outputs on two
passes. It also checked the full GloVe archive and 300d member and produced a
finite normalized word feature. The run finished in 13.926 seconds.

[evidence.json](evidence.json) records the exact release, model checks and
runtime. [check_runtime.py](check_runtime.py) is the exact executed harness;
its machine-local paths describe the existing VM layout and its output uses
exclusive creation. It is not a collection tool or part of the six prescribed
competence fits. The four-file release lock remains unchanged.

This establishes runtime readiness for this fixture. It does not establish
competence on photographs, unpaired alignment, or freedom from model-choice
bias. The experiment still requires new human photographs and captions.
