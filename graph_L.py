import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb

Ls = sio.loadmat('mats/Ls.mat')['x'][:,0]
plt.plot(Ls)
plt.title("L_k over time", fontdict={'fontsize':20})
plt.xlabel("Iteration # (k)", fontdict={'fontsize':18})
plt.ylabel("L_k", fontdict={'fontsize':18})
# plt.axis([0, 200, 0, 10])

plt.show()
