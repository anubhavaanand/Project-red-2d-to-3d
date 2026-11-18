"""Shared test helpers.

Several test modules must import an engine module on a machine with no real
torch, so they register a minimal ``torch`` stub in ``sys.modules``. Assigning to
``sys.modules`` directly leaks that stub into every test that runs afterwards in
the same session: with real torch installed, the stub clobbers the genuine module
and later tests fail with e.g.
``AttributeError: module 'torch' has no attribute 'from_numpy'``.

``TorchStub`` installs the stub only when real torch is absent, and restores
whatever was there before on ``restore()``. Use it from a module-scoped fixture::

    @pytest.fixture(scope="module")
    def lgm():
        stub = TorchStub()
        stub()
        try:
            from src.engines import lgm_engine
            return lgm_engine
        finally:
            stub.restore()
"""

import sys
import types

# Every sys.modules key a torch stub may reasonably touch.
TORCH_MODULE_KEYS = (
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.cuda",
    "torch.autograd",
    "torch.utils",
    "torch.utils.data",
    "torch.jit",
    "torch._C",
)


class TorchStub:
    """Installs a fake ``torch`` (only if real torch is absent) and restores it."""

    def __init__(self):
        self._saved = {}
        self.installed = False

    def __call__(self, **attrs) -> types.ModuleType:
        """Install the stub if needed; return the module ``torch`` resolves to.

        Extra keyword arguments are set as attributes on the stub, which lets a
        test that needs e.g. ``torch.OutOfMemoryError`` declare it inline.
        """
        if "torch" in sys.modules:
            # Real torch (or another module's stub) is already in place — leave it.
            return sys.modules["torch"]

        self._saved = {name: sys.modules.get(name, ...) for name in TORCH_MODULE_KEYS}
        self.installed = True

        torch = types.ModuleType("torch")
        torch.__version__ = "0.0-stub"
        torch.float32, torch.float16, torch.int64 = "float32", "float16", "int64"
        torch.device = lambda x: x
        torch.manual_seed = lambda s: None

        class _NoCtx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __call__(self, fn):
                # supports use as a decorator: @torch.no_grad()
                return fn

        torch.no_grad = _NoCtx
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
            device=lambda d: _NoCtx(),
        )
        torch.nn = types.ModuleType("torch.nn")
        torch.nn.functional = types.ModuleType("torch.nn.functional")

        for name, value in attrs.items():
            setattr(torch, name, value)

        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = torch.nn
        sys.modules["torch.nn.functional"] = torch.nn.functional

        # Optional heavy deps the engine modules import lazily; stub them too.
        for name in ("safetensors", "safetensors.torch", "huggingface_hub"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}
        sys.modules["huggingface_hub"].hf_hub_download = lambda *a, **k: ""
        sys.modules["huggingface_hub"].snapshot_download = lambda *a, **k: ""
        return torch

    def restore(self):
        """Put back whatever ``sys.modules`` held before the stub was installed."""
        for name, value in self._saved.items():
            if value is ...:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        self._saved = {}
        self.installed = False
