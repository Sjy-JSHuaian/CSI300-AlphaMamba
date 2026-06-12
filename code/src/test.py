"""Prediction entry point — Meta Ranker V2.

Loads Phase 6B models + submit1 scores → Ridge Meta Ranker → Portfolio Optimizer.
Output: output/result.csv (top-5 stocks with optimized weights)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

from predict_meta_ranker import main

if __name__ == "__main__":
    main()
