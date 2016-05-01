import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb

Ls = sio.loadmat('Ls.mat')['x'][:,0]
ms = sio.loadmat('ms.mat')['x'][:,0]
gpns = sio.loadmat('gpns.mat')['x'][:,0]
losses = sio.loadmat('sgd_back.mat')['x'][:,0]

plt.subplot(221)
plt.plot(Ls, label='L')
plt.legend()

plt.subplot(222)
plt.plot(ms, label='m')
plt.legend()

plt.subplot(223)
plt.plot(gpns[50:], label='gpn')
plt.legend()

plt.subplot(224)
plt.plot(losses, label='loss')
plt.legend()
plt.show()
