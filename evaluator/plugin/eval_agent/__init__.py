"""eval_agent — Evaluation agent plugin package

Auto-imports all *_eval_agent.py modules in this directory to trigger
__init_subclass__ registration. Missing or broken plugins are silently skipped.
"""

import importlib
import pkgutil

for _info in pkgutil.iter_modules(__path__):
    if _info.name.endswith("_eval_agent"):
        try:
            importlib.import_module(f"{__name__}.{_info.name}")
        except ImportError:
            pass
