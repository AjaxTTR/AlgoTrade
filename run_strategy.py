"""
Run any strategy module from the command line.

Usage:
    python run_strategy.py <strategy_name>

Example:
    python run_strategy.py strategy

The strategy name must correspond to a file in strategies/ that
exposes a generate_signals(df, **params) function.
"""

import sys


def cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_strategy.py <strategy_name>")
        print()
        print("Example:")
        print("  python run_strategy.py strategy")
        sys.exit(1)

    strategy_name = sys.argv[1]

    import main

    main.STRATEGY_NAME = strategy_name
    main.main()


if __name__ == "__main__":
    cli()
