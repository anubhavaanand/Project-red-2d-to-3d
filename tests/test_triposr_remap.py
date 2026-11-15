"""Tests for the TripoSR checkpoint remap wrapper (ENG-02)."""

import ast
from pathlib import Path

ENGINE = Path(__file__).parent.parent / "src" / "engines" / "triposr_engine.py"


def _remap_source():
    src = ENGINE.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_remap_load_state_dict":
            return ast.unparse(node)
    raise AssertionError("_remap_load_state_dict not found")


def _build_remap(original):
    """Compile the wrapper with `_orig_load_state_dict` bound to `original`."""
    wrapper = _remap_source()
    # the wrapper calls the module-level helpers; stub them
    ns = {"_remap_vit_checkpoint_keys": lambda sd, model: sd, "_orig_load_state_dict": original}
    exec(wrapper, ns)
    return ns["_remap_load_state_dict"]


class TestStrictForwarding:
    def test_default_is_not_strict(self):
        calls = []

        def orig(model, sd, strict=False, **kw):
            calls.append({"strict": strict, "kw": kw})
            return [], []

        _build_remap(orig)(object(), {})
        assert calls[0]["strict"] is False

    def test_strict_true_is_forwarded(self):
        """ENG-02: a caller asking for strict loading must reach the original."""
        calls = []

        def orig(model, sd, strict=False, **kw):
            calls.append(strict)
            return [], []

        _build_remap(orig)(object(), {}, strict=True)
        assert calls == [True]

    def test_strict_is_never_hardcoded_false(self):
        """The bug was `strict=False` at the call site; assert it's forwarded."""
        wrapper = _remap_source()
        assert "strict=strict" in wrapper
        # the forwarded call must not pin strict to a literal
        call = next(
            line for line in wrapper.splitlines() if "_orig_load_state_dict(" in line
        )
        assert "strict=strict" in call
        assert "strict=False" not in call

    def test_extra_kwargs_are_forwarded(self):
        calls = []

        def orig(model, sd, strict=False, **kw):
            calls.append(kw)
            return [], []

        _build_remap(orig)(object(), {}, strict=True, assign=True)
        assert calls[0] == {"assign": True}


class TestSourceStructure:
    def test_wrapper_is_installed_once(self):
        src = ENGINE.read_text()
        assert "TSR._state_dict_wrapped" in src

    def test_remap_applied_before_loading(self):
        """The remap must actually be called by the wrapper."""
        wrapper = _remap_source()
        assert "_remap_vit_checkpoint_keys(state_dict, self_model)" in wrapper

    def test_engine_parses(self):
        ast.parse(ENGINE.read_text())