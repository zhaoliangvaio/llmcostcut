# llmcompiler

A framework for selectively invoking LLMs and distilling repeated workloads into smaller models.

This library provides infrastructure for:
- correctness-aware LLM fallback
- online distillation
- replay-buffer–based incremental training

Used in research work: [Distilling LLM Reasoning into Graph of Concept Predictors](https://arxiv.org/abs/2602.03006)


## Figures

### Fallback Mechanism
![Fallback](fallback.png)

The fallback ratio (teacher/LLM utilization) decreases over iterations as the student model becomes more capable. Early in training, the system relies entirely on the teacher LLM for correctness; as online distillation progresses, the student handles an increasing fraction of queries independently, reducing LLM calls to near zero by iteration 100.

### Accuracy Results
![Accuracy](acc.png)

Overall system accuracy remains close to the teacher's baseline (100%) throughout training. Despite the sharp reduction in LLM fallback, accuracy stabilizes around 95% after the initial adaptation phase, demonstrating that the distilled student model preserves predictive quality while significantly cutting inference cost.

## Acknowledgements

This framework was developed by the LLMCompiler team at Emory University.

We thank our collaborators and students for discussions and feedback.
Portions of this system were inspired by work in online learning,
knowledge distillation, and adaptive inference.

This repository will be released in conjunction with an upcoming
research publication.