#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiments."""

import os


def main():
    print("*********************** Preparation ***********************", flush=True)
    os.system("python reformatting.py")
    print("************** Balancing techniques results ***************", flush=True)
    os.system("python balancing_techniques.py")
    print("**************** Usual sources comparison *****************", flush=True)
    os.system("python usual_sources_compar.py")
    print("************** Additional sources comparison **************", flush=True)
    os.system("python additional_sources_compar.py")
    print("*************** Transfer learning experiment **************", flush=True)
    os.system("python transfer_learning.py")


if __name__ == "__main__":
    main()
