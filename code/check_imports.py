"""Lightweight import check - just verify imports work."""
import sys, os
LOG = 'check_imports.log'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

log('Step 1: starting')
log('Step 2: importing torch')
import torch
log('  torch OK, CUDA=' + str(torch.cuda.is_available()))

log('Step 3: importing numpy')
import numpy as np
log('  numpy OK')

log('Step 4: importing algorithms_collection')
log('  (this may take 30-60s due to sklearn/torch/mne imports)')
import algorithms_collection
log('  algorithms_collection OK')

log('Step 5: trying SSCSleepNet')
from algorithms_collection import SSCSleepNet, get_algorithm
log('  imports OK')

log('Step 6: instantiating SSCSleepNet')
m = SSCSleepNet(n_channels=2, n_times=3000, n_classes=5)
log('  Device=' + str(m.device))

log('=== IMPORT CHECK PASSED ===')
