import numpy as np
import matplotlib.pyplot as plt

filename = 'npy_dataset/train/SA1.npy'

data_tr = np.load(filename)
print(data_tr.shape)
# plt.imshow(data_tr[:,:28].T)

# plt.imshow(data_tr[:,28:].T)


filename = 'npy_dataset/test/SA1.npy'

data_ts = np.load(filename)
print(data_ts.shape)
# plt.imshow(data_ts[:,:28].T)

# plt.imshow(data_ts[:,28:].T)

filename = 'npy_dataset/valid/SA1.npy'

data_v = np.load(filename)
print(data_v.shape)
# plt.imshow(data_v[:,:28].T)

# plt.imshow(data_v[:,28:].T)

