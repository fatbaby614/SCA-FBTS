"""Smoke pre-test for SSC-SleepNet GPU fix and SCA-FBTS."""
import sys, os
sys.path.insert(0, '.')

LOG = 'smoke_pretest.log'
def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

if os.path.exists(LOG):
    os.remove(LOG)

import numpy as np
import torch
from algorithms_collection import SSCSleepNet, get_algorithm

log('CUDA available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    log('Device: ' + torch.cuda.get_device_name(0))

log('\n[Test 1] SSC-SleepNet init + 3-epoch fit')
m = SSCSleepNet(n_channels=2, n_times=3000, n_classes=5)
log('  Device: ' + str(m.device))
X = np.random.randn(64, 2, 3000).astype(np.float32)
y = np.random.randint(0, 5, 64)
m.fit(X, y, epochs=3)
log('  [OK] SSC-SleepNet fit done')

log('\n[Test 2] SCA-FBTS fit + predict')
m2 = get_algorithm('SCA-FBTS', n_channels=2, n_times=3000, n_classes=5)
m2.fit(X, y)
pred = m2.predict(X[:8])
log('  [OK] SCA-FBTS predict shape: ' + str(pred.shape))

log('\n=== ALL PRE-TESTS PASSED ===')
