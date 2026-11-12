name: Newcomer onboarding
description: A guided first-contribution issue for someone new to the project.
title: "[onboarding] First contribution: "
labels: ["good first issue", "help wanted", "documentation", "status:triage"]
body:
  - type: markdown
    attributes:
      value: |
        Welcome. This issue is a guided path, not a bug report. Work through the reading
        list, pick one small task, and open a PR. You do not need a GPU for the tasks
        listed below.

  - type: markdown
    attributes:
      value: |
        ## 1. Read in this order
        1. `README.md` — claims and protocol summary.
        2. `AGENTS.md` — technical context.
        3. `src/engines/base.py` — the engine contract.
        4. `src/engines/triposr_engine.py` — simplest engine.
        5. `src/evaluation/benchmark_runner.py` — start at `BenchmarkRunner.run`.
        6. `docs/ISSUES.md` — pick one `good first issue`.
        7. `CONTRIBUTING.md` — workflow, verification and PR rules.

  - type: checkboxes
    id: reading
    attributes:
      label: Reading checklist
      options:
        - label: I read `README.md`.
        - label: I read `AGENTS.md`.
        - label: I read `src/engines/base.py` and `src/engines/triposr_engine.py`.
        - label: I skimmed `src/evaluation/benchmark_runner.py`.
        - label: I read the full entry for the issue I plan to claim.

  - type: dropdown
    id: task
    attributes:
      label: Which small task do you want?
      description: These are scoped for a first PR and do not require CUDA.
      options:
        - "DOC-01 — fix README ICP iteration count"
        - "DOC-06 — fix COMPILE.md numbering/backtick"
        - "EXP-03 — create parent directory in GLB export"
        - "EXP-05 — normalise float vertex colours in GLB export"
        - "EVAL-32 — rename gt_novel_Views to snake_case"
        - "EVAL-29 — remove duplicate Chamfer wrapper"
        - "TOOL-06 — fix merge_batches docstring"
        - "TOOL-07 — add --out to merge_batches"
        - "PKG-13 — add license metadata"
        - "DOC-11 — add CHANGELOG.md"
        - "DOC-12 — add CODE_OF_CONDUCT.md"
        - "CI-01 — add the first test"
    validations:
      required: true

  - type: textarea
    id: plan
    attributes:
      label: Your plan
      description: Which files will you change, and how will you verify it?
    validations:
      required: true

  - type: textarea
    id: questions
    attributes:
      label: Questions / blockers
    validations:
      required: false

  - type: markdown
    attributes:
      value: |
        ## Rules
        - One issue/branch per PR.
        - Do not change the locked protocol silently.
        - Do not commit outputs, weights or GSO assets.
        - Include the issue ID in the PR body.
        - Run the verification commands in `CONTRIBUTING.md` §7.