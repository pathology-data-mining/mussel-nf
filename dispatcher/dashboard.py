#!/usr/bin/env python3
"""Backward-compat shim. Use: python -m mussel_dispatcher.dashboard  or  mussel-dashboard CLI."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mussel_dispatcher.dashboard.server import main
if __name__ == "__main__":
    main()
