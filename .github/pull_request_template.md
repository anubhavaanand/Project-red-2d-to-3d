## Summary

<!-- One or two sentences: what changed and why. -->

## Related issue

<!-- e.g. EVAL-03. If this PR closes an issue, use "Closes #123". -->

## Component / labels

<!-- e.g. component:evaluation, bug, severity:medium -->

## What changed

<!-- Bullet the files and behaviour changes. Use file.py:line references. -->

## Protocol impact

- [ ] This PR does **not** change a measured number or the locked protocol.
- [ ] This PR **does** change a measured number or the locked protocol (explain below).

<!-- If protocol changed: which number(s), why, and how old runs remain reproducible. -->

## Verification

<!-- Commands run and their result. If no test framework exists, describe the check. -->

```bash
# e.g.
python -m py_compile src/evaluation/benchmark_runner.py
```

## Checklist

- [ ] I read `CONTRIBUTING.md`.
- [ ] I read the full entry in `docs/ISSUES.md` for the referenced ID.
- [ ] I did not commit generated outputs, model weights or GSO assets.
- [ ] I updated docs/labels when behaviour changed.
- [ ] I added or updated a focused test/script where feasible.