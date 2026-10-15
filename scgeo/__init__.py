__version__ = "0.1.0-dev"

from . import tl, pl, get, data, pp

__all__ = ["tl", "pl", "get", "data", "pp", "bench", "__version__"]


def __getattr__(name):
    # Benchmark resource profiling is Unix-specific; do not make it a
    # requirement for importing the cross-platform analysis package.
    if name == "bench":
        from importlib import import_module

        try:
            module = import_module(".bench", __name__)
        except ModuleNotFoundError as exc:
            if exc.name != "resource":
                raise
            raise ImportError(
                "scgeo.bench requires the Unix-only resource module for "
                "benchmark profiling. Run benchmarks on Linux (including WSL) "
                "or macOS; ScGeo analysis functions remain available on Windows."
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
