"""Test SSC-SleepNet directly without mne import."""
import sys, os
LOG = 'test_ssc.log'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
if os.path.exists(LOG):
    os.remove(LOG)

log('[1] Python OK')
log('[2] importing torch...')
import torch
log('[2.1] torch ' + torch.__version__ + ' CUDA=' + str(torch.cuda.is_available()))

log('[3] importing numpy...')
import numpy as np
log('[3.1] numpy OK')

# Skip mne - directly import torch nn
import torch.nn as nn
log('[4] nn OK')

# Try importing braindecode (this is the suspected slow one)
log('[5] importing braindecode.util.set_random_seeds...')
try:
    from braindecode.util import set_random_seeds
    log('[5.1] braindecode.util OK')
except Exception as e:
    log('[5.1] braindecode.util FAILED: ' + str(e))

log('[6] importing skorch...')
try:
    from skorch.callbacks import EarlyStopping
    log('[6.1] skorch OK')
except Exception as e:
    log('[6.1] skorch FAILED: ' + str(e))

log('[7] importing braindecode EEGClassifier...')
try:
    from braindecode import EEGClassifier
    log('[7.1] EEGClassifier OK')
except Exception as e:
    log('[7.1] EEGClassifier FAILED: ' + str(e))

log('[DONE]')
