"""Test just the basic Python+CUDA without algorithms_collection."""
import sys, os
LOG = 'test_torch.log'
def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Truncate
if os.path.exists(LOG):
    os.remove(LOG)

log('[1] Python ' + sys.version)
log('[2] importing torch...')
import torch
log('[2.1] torch ' + torch.__version__ + ' CUDA=' + str(torch.cuda.is_available()))

log('[3] importing mne...')
import mne
log('[3.1] mne ' + mne.__version__)

log('[4] importing sklearn...')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
log('[4.1] sklearn OK')

log('[5] importing pyriemann...')
from pyriemann.estimation import Covariances
log('[5.1] pyriemann OK')

log('[6] importing skorch...')
from skorch.callbacks import EarlyStopping
log('[6.1] skorch OK')

log('[7] importing braindecode...')
from braindecode import EEGClassifier
log('[7.1] braindecode OK')

log('[ALL IMPORTS DONE]')
