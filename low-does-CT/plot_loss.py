import numpy as np
import matplotlib.pyplot as plt

#path = '.Low_Dose/IOWM/Poisson_noise_1e4/nosie_1/loss_81200_iter.npy'
path = "./Low_Dose/IOWM/save_dcm_6/loss_82000_iter.npy"

losses = np.load(path)
x = np.arange(losses.shape[0])
epochs_loss = losses[::410]
print(epochs_loss)
plt.plot(np.arange(len(epochs_loss)), epochs_loss, 'r', label='train_loss')
plt.legend()
plt.savefig('./figure/loss_curve.jpg')
plt.close()


