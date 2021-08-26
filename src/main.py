#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script to run.

Usage:
======
    python main.py

"""

import time
from datetime import datetime
from datetime import timedelta
import os
current_dir = os.getcwd()


def main():
    print("\n===========================================================", flush=True)
    print("====================== PREPROCESSING ======================", flush=True)
    print("===========================================================", flush=True)
    os.chdir("preprocessing")
    os.system("python formatting.py")
    os.system("python visualization.py")
    os.system("python transformation.py")
    os.chdir(current_dir)
    print("\n===========================================================", flush=True)
    print("==================== FEATURE SELECTION ====================", flush=True)
    print("===========================================================", flush=True)
    os.chdir("feature_selection")
    os.system("python fs.py")
    os.chdir(current_dir)
    print("\n===========================================================", flush=True)
    print("======================= EXPERIMENTS =======================", flush=True)
    print("===========================================================", flush=True)
    os.chdir("experiments")
    os.system("python exp.py")


if __name__ == "__main__":
    start_time = datetime.now()
    start_time_ = time.monotonic()
    print("\nStart time: ", start_time)

    main()

    end_time = datetime.now()
    end_time_ = time.monotonic()
    print('\nDuration: {}'.format(timedelta(seconds=end_time_ - start_time_)))
    print("(start \t {})".format(start_time))
    print("(end \t {})".format(end_time))
