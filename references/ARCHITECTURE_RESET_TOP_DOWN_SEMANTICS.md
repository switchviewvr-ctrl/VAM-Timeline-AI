# Architecture Reset: Top-Down Semantics

Update: `Semantik_Master_Konsolidiert.docx` is now the canonical motion-ontology sourcebook. The top-down layer derived from it is `data/ontology/*_v2.yaml`. The older v1 ontology remains implementation history and should not be treated as the newest semantic source.

The project previously leaned too far bottom-up:

controller curves / motion windows -> features -> heuristic labels -> ML ranker -> maybe generation

That remains useful for analysis, retrieval, review, and calibration. It is not enough for generation.

The generation direction is now:

prompt / human semantic concept -> motion meaning -> pose + role + partner relation + contact/support -> motion grammar -> parameterized relative motion -> controller curves -> native Timeline export

Controller movement is the renderer/output layer. It is not the source of meaning.

Data, heuristics, ML, and VLM outputs are review-assist signals. They can calibrate numeric parameters, rank examples, and expose contradictions, but they do not define semantic truth.

Human review remains the final truth source.

Important VaM interpretation:

- In source material, "root", "root node", and "pelvis root" mean the body pelvis/hip/abdomen driver region.
- In VaM implementation this maps to `pelvisControl`, `hipControl`, and sometimes abdomen/abdomen2 controls.
- It never maps to the VaM `Person` atom root, world transform, or scene transform.

Existing clean_v2/clean_v3 heuristic labels are review aids. The Cowgirl ML ranker is review-assist only. The local VLM judge is review-assist only and currently low-trust unless live calibration improves. None of these outputs should be merged into `manual_labels.yaml` automatically.
