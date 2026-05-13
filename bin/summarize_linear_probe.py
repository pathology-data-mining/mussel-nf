#!/usr/bin/env python3
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
from mussel_benchmark.linear_probe import main
if __name__ == "__main__":
    main()
