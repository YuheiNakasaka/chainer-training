import numpy as np
import matplotlib.pyplot as plt

def unpickle(image):
    import pickle
    f = open(image, 'rb')
    d = pickle.load(f, encoding='latin-1') # oh...
    f.close()
    return d

label_names = unpickle("./dataset/batches.meta")["label_names"]
d = unpickle("./dataset/data_batch_1")
data = d["data"]
labels = np.array([d["labels"]])

nclasses = 10
pos = 1
for i in range(nclasses):
    print("index: {} name: {}".format(i, label_names[i]))
    targets = np.where(labels == i)[1]
    np.random.shuffle(targets)
    for idx in targets[:10]:
        plt.subplot(10, 10, pos)
        img = data[idx]
        plt.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))
        plt.axis('off')
        pos += 1
plt.show()
