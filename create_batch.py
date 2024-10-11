import numpy as np
import matplotlib.pyplot as plt

batch_shape = (100, 200)
batch = np.zeros(batch_shape)
N_lines = 5
batch_type = 'smooth'

# Generate random data by linear combination of sine waves
data = np.zeros((N_lines, batch_shape[1]))

for i in range(N_lines):
    t = np.linspace(0, 10, batch_shape[1])
    data[i] = np.sin(2 * np.pi * (i + 1) * t)

    if batch_type == 'smooth':
        x_coord = int(batch_shape[0] // N_lines * (i+0.5)) 
        print(x_coord)
        print(batch[x_coord, :].shape, data[i].shape)
        batch[x_coord, :] = data[i]

plt.imshow(batch, cmap='gray')
plt.show()