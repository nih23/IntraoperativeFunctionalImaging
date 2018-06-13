import ThermalImagingAnalysis as tai
import ActivityPatterns as ap
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time

# load data
pPenalty = "Penalty_Gaussian_1024fr_2.5Hz_TruncatedWaveletBasis.mat"
pioData = 'io_data/seq_1074564_2.5Hz.mat'
f = h5py.File(pioData, "r")
S = np.array(f["seq"].value).transpose()
T = np.squeeze(np.array(f["T"].value))
f.close()
S2 = S[0:1024,]
T2 = T[0:1024,]
f_P = h5py.File(pPenalty, "r")
P = f_P["BPdir2"].value   # learned penalty matrix
print('[INFO] P is being transposed\n')
P = P.transpose() # P appears to be stored as transposed version of itself
B = f_P["B"].value # basis matrix
val = ap.computeBoxcarActivityPattern(T,sigma=30)
val_neg,vp = val.nonzero()

X = ap.computeGaussianActivityPattern(np.squeeze(T2)).transpose();
start_time = time.time()
F = tai.semiparamRegressio_VCM(S2,T2,B,P);
elapsed_time = time.time() - start_time
print('elapsed (CPU): ' + str(elapsed_time) + ' s')
plt.imshow(np.reshape(F,[640, 480]));plt.show()