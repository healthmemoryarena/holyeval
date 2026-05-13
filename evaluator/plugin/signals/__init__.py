"""signals — response-feature detector plugins.

Importing this package triggers registration of the built-in signals.
New signals live in sibling modules that call `register_signal(...)`.
"""

from . import builtin  # noqa: F401 — side-effect: registers built-in signals
