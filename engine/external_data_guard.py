"""External data provider import guard.

Importing this module permanently blocks any attempt to import external
market data providers or HTTP libraries.  The project runs exclusively
from local CSV data (data/nq_15m_data.csv).

Activate by importing this module before anything else:

    import engine.external_data_guard  # noqa: F401
"""

import builtins

_BLOCKED_MODULES = frozenset({
    "databento",
    "databento_dbn",
    "polygon",
    "alpaca",
    "ccxt",
    "yfinance",
    "requests",
    "aiohttp",
})

_original_import = builtins.__import__


def _guarded_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise RuntimeError(
            "External market data providers are disabled. "
            "This project runs only from local CSV data."
        )
    return _original_import(name, *args, **kwargs)


# Activate immediately on import
builtins.__import__ = _guarded_import
