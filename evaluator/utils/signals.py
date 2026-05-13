"""Signal registry — pluggable response-feature detectors for rubric-style evaluators.

A signal extracts a deterministic, structured fact from the AI's reply text and
the raw target reaction (e.g. `has_chart: bool`, `char_count: int`). Signals are
consumed two ways by the rubric evaluator:
  1. `signal_check` criteria — a local assertion `signal op value → True/False`
  2. Inline facts surfaced to the LLM judge so free-form rubrics can reference
     them without the judge re-inferring the facts from the text.

Register via the `@register_signal("name")` decorator; no core changes needed
to add new signals. See `evaluator/plugin/signals/builtin.py` for built-ins.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

SignalFn = Callable[[str, Any], Any]

_REGISTRY: Dict[str, SignalFn] = {}


def register_signal(name: str) -> Callable[[SignalFn], SignalFn]:
    """Decorator: register `fn(response: str, reaction: Any) -> Any` as a signal.

    Later registrations override earlier ones under the same name (useful for
    tests and product-specific overrides).
    """

    def decorator(fn: SignalFn) -> SignalFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def compute_signal(name: str, response: str, reaction: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(
            f"signal {name!r} not registered. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](response or "", reaction)


def list_signals() -> List[str]:
    return sorted(_REGISTRY.keys())


def has_signal(name: str) -> bool:
    return name in _REGISTRY
