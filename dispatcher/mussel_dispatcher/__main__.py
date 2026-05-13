#!/usr/bin/env python3
"""mussel-dispatcher — entry point.

Run as:
    python -m mussel_dispatcher config.yaml
    mussel-dispatcher config.yaml          # after pip install -e .
"""
from mussel_dispatcher.scheduler import main

if __name__ == "__main__":
    main()
