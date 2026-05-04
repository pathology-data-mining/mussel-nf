#!/usr/bin/env python3
"""Backward-compat shim. Use: python -m mussel_dispatcher  or  mussel-dispatcher CLI."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mussel_dispatcher.scheduler import main
from mussel_dispatcher import *  # noqa: re-export all public symbols

if __name__ == "__main__":
    main()
