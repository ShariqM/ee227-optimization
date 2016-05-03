import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb

sgd_loss  = sio.loadmat('mats/sgd_back.mat')['x'][:,0]
rms_loss  = sio.loadmat('mats/rmsprop.mat')['x'][:,0]
adam_loss = sio.loadmat('mats/adam.mat')['x'][:,0]

lw=6
plt.plot(sgd_loss,  label='SGD-Back', linestyle='-', lw=lw)
plt.plot(rms_loss,  label='RMSProp Loss', linestyle=':', lw=lw)
plt.plot(adam_loss, label='Adam Loss', linestyle='--', lw=lw)

plt.title("Convergence for various algorithms", fontdict={'fontsize':20})
plt.xlabel("Iteration # (k)", fontdict={'fontsize':18})
plt.ylabel("Loss", fontdict={'fontsize':18})
plt.axis([0, 200, 0, 10])

plt.legend()
plt.show()
