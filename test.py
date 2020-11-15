import h5py
import numpy as np

with h5py.File("data/cad/cad.h5", "r") as hf:
    train_images = np.array(hf.get(name="train_images"))
    val_images = np.array(hf.get(name="val_images"))
    test_images = np.array(hf.get(name="test_images"))

print(train_images.shape)
print((val_images == test_images).all())
