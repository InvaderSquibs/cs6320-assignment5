# Assignment 5: Generalization and Trustworthy Evaluation

- **Due** Tuesday by 11:59pm
- **Points** 100
- **Submitting** a file upload
- **File Types** pdf, docx, md, zip, ipynb, py, csv, png, json, and txt

# CS 6320 Assignment 5: Generalization and Trustworthy Evaluation

## Purpose

This assignment asks whether a model can be trusted beyond a single aggregate score. You will use training and validation evidence, one generalization intervention, error analysis, subgroup or slice checks, and calibration or confidence evidence when appropriate to make an initial reliability judgment.

This is not a final portfolio-model assignment. The goal is to practice a trustworthy evaluation process and connect the evidence back to the dataset audit, risks, assumptions, and staged model plan you locked in Assignment 4.

By completing this assignment, you should be able to:

- Diagnose overfitting, underfitting, or plausible generalization from learning curves, validation results, and error examples.
- Select and justify a regularization, early stopping, dropout, or model-simplification intervention for a specific observed failure pattern.
- Explain why the chosen train/validation/test strategy is appropriate for the intended use and where it may still be weak.
- Analyze model behavior by examples, subgroups, slices, or conditions instead of relying only on an aggregate metric.
- Evaluate calibration or confidence behavior when it is meaningful for the task.
- Compare observed evidence against the risks and assumptions documented in the Assignment 4 portfolio charter and dataset audit.
- Make a bounded deployment or non-deployment recommendation that states what is supported, what is risky, and what remains untested.

## Expected Time

Estimated outside-class time: 5-7 hours.

Workload guidance:

- Part A: model evaluation and generalization evidence, about 3.5-4.5 hours.
- Part B: portfolio audit update and reliability judgment, about 1.5-2.5 hours.

## What You Will Do

Evaluate a portfolio baseline or initial model candidate if your portfolio project is ready for modeling. Use the dataset audit, evaluation strategy, leakage risks, success criteria, and staged model-improvement plan from Assignment 4 to decide what evidence to collect.

If your portfolio project is not yet ready for this level of evaluation, use the approved practice model based on the Week 4 NYC TLC trip-duration dataset, or instructor-provided sample results derived from that dataset. In that case, label Part A clearly as practice work, complete the same evaluation process, and explicitly explain how the process will transfer to your portfolio project. Part B must still discuss your portfolio project: name which evidence transfers directly and which portfolio-specific evidence remains missing.

The assignment should produce early evidence for the final portfolio presentation where possible, but the current model should be treated as a baseline or initial candidate, not as the final recommendation.

## Reuse From Assignment 4

Before starting new work, review your Assignment 4 portfolio charter and dataset audit.

Use the charter to identify:

- The intended stakeholder or use case.
- The prediction target and candidate inputs.
- The planned train/validation/test split or other evaluation strategy.
- The metrics and success criteria that fit the task.
- The leakage, prediction-time availability, missingness, imbalance, representativeness, or responsible-use risks that need evidence.
- The staged model-improvement plan and the next likely model revision.

If you change the split strategy, metric, baseline, model candidate, or scope from Assignment 4, briefly explain why the change is necessary. Do not redefine the core portfolio project unless the Assignment 4 scope has become infeasible and you have instructor approval.

## Required Work

### Part A: Model Evaluation and Generalization Evidence

Using your portfolio baseline, initial model candidate, approved practice model, or approved sample results:

- State the dataset, task, target, model being evaluated, and whether this is portfolio work or approved practice work.
- Use a clear train/validation/test strategy or other justified evaluation procedure.
- Explain why the split strategy is appropriate for the intended use and whether temporal, grouped, repeated-entity, or other leakage concerns affect the split.
- Provide evidence for the split choice, not only a rationale. At minimum, report split counts, how the split was created, and one check that the split is plausible for the task, such as target or label distributions by split, time ranges by split, group/entity overlap checks, category coverage checks, or another task-relevant split audit.
- Report at least one task-appropriate aggregate metric, using the metric plan from Assignment 4 where feasible.
- Compare training and validation behavior using a learning curve, metric table over epochs or settings, or another clear before/after evaluation trace.
- When feasible, include both training and validation metrics over epochs. A final metric table alone is not enough to diagnose overfitting or underfitting.
- Diagnose whether the evidence suggests overfitting, underfitting, plausible generalization, leakage risk, unstable validation behavior, or insufficient evidence.
- Apply at least one generalization technique such as weight decay, dropout, early stopping, data augmentation where appropriate, model simplification, reduced features, or another justified constraint.
- Compare behavior before and after the intervention and explain what changed.
- Analyze errors by examples, subgroup, slice, condition, time period, source, class, target range, or another task-relevant dimension.
- Evaluate calibration or confidence behavior when appropriate for the task. If calibration is not appropriate, briefly explain why.

**Scope boundary:** use one main model and one focused generalization intervention. You may include a small number of additional diagnostic runs if they clarify the main result, but identify one primary intervention and do not present the work as a broad hyperparameter search. Do not run an exhaustive hyperparameter search, try many architectures without a specific diagnostic purpose, or attempt to make the model final this week.

**Depth expectation:** this section should be evidence-first. A reader should be able to see what the model did, where it failed, what you changed, and whether the change improved trustworthiness or only changed the aggregate score.

### Part B: Portfolio Audit Update and Reliability Judgment

Connect the evaluation evidence back to your Assignment 4 charter and dataset audit.

Your writeup should include:

- A short trace from at least three Assignment 4 audit risks, assumptions, or success criteria to the Week 5 evidence you collected.
- Which risks or assumptions were confirmed, reduced, contradicted, newly discovered, or still untested.
- Whether the model is reliable enough for the intended use at this stage.
- What additional evidence would be needed before deployment or client-facing recommendation.
- How this evidence should guide the next staged portfolio model improvement.
- If you used a practice model, a short note explaining which parts of the evaluation process transfer directly to your portfolio project and which parts still need portfolio-specific evidence.

**Depth expectation:** this section should make a bounded reliability claim. Avoid both overclaiming and vague caution. State what the current evidence supports, what it does not support, and what the next model or evaluation step should test.

## Deliverable

Submit code or evaluation artifacts and one writeup document with two labeled parts: Part A and Part B.

Required structure:

- **Code or evaluation artifacts:** scripts, notebooks, configuration notes, command history, saved metrics, plots, logs, or provided sample-result references sufficient to understand what was evaluated.
- **Run evidence:** include the command used when you ran the model yourself, the data split description, random seed or nondeterminism note when relevant, and saved output, metric file, learning curve, or plot showing that the evaluation ran.
- **Part A writeup:** include dataset/task summary, model summary, split strategy, metric results, training-versus-validation evidence, generalization diagnosis, intervention attempted, before/after comparison, error analysis, subgroup or slice analysis, and calibration or confidence discussion when appropriate.
- **Part B writeup:** include the Assignment 4 audit-risk trace, confirmed or challenged assumptions, current reliability judgment, remaining evidence needed before deployment, and next staged model-improvement plan.

Useful artifacts include split-count tables, split-distribution plots, learning curves, metric summaries, confusion matrices or residual/error tables, and subgroup/slice summaries. Not every task needs every artifact, but the submitted evidence should make the split, model behavior, intervention effect, and reliability judgment inspectable.

What is not required in this assignment:

- No exhaustive hyperparameter search.
- No requirement that the intervention improve every metric.
- No requirement that the current model be deployment-ready.
- No final portfolio recommendation.
- No large compute escalation beyond the course workflow.
- No calibration analysis for tasks where confidence or probability interpretation is not meaningful.
- No project-scope change unless your Assignment 4 charter has become infeasible and you have instructor approval.

## Portfolio Connection

This assignment should produce the first major evidence section for your final presentation where your portfolio project is ready. Your goal is to move beyond accuracy and begin explaining when the current model works, when it fails, which audit risks appear in practice, and what should improve in the next staged model iteration.

If you use an approved practice model instead, the portfolio connection is methodological: you are practicing the evaluation structure that you will later apply to your portfolio model. Your writeup should still identify which Assignment 4 portfolio risks need analogous evidence.

## Success Criteria

A strong submission makes a defensible reliability judgment. It includes concrete evidence, compares training and validation behavior, applies one justified generalization intervention, analyzes errors beyond an aggregate metric, checks subgroup, slice, or condition behavior where possible, addresses calibration or confidence when meaningful, and explicitly updates the Assignment 4 audit assumptions.

A strong submission does not claim that a model is useful because of one score. It states what the evidence supports, what remains uncertain, and how the next staged model improvement should respond.