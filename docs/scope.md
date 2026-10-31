# Analyses covered by this repository

The code supports the analyses reported in the paper:

- ISBC training and encoding from frozen text embeddings.
- AG News analysis of 127,600 documents: bit associations, conjunctions,
  profiles, embedding clusters, reconstruction and geometry.
- Code width/activity comparisons and learning-objective ablations.
- Correspondence across five embedding backbones.
- Alternative encoders, including native and top-6 ITQ and BinarySentEmb.
- Qwen-based analyses of annotated corpora.
- Mixed-domain fitting and interpretation of additional documents.
- PHEME log-outcome prediction, engagement tails and content inspection.
- Optimization timing and the ISBC Explorer dashboard.

IRA is a source component of the Mixed corpus; `download_ira_io.py` prepares
that component. It is not a standalone cross-corpus experiment.

`reproduction/workflows.json` defines the runnable workflows. Numerical recipes
are stored in `recipes/` and `runs/`; generated outputs, cached embeddings and
model weights are excluded from version control. Shared mathematical modules
provide the operations used by these workflows. No script edits or compiles
the manuscript.
