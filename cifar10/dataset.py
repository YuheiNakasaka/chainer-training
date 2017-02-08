import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    label_names = unpickle("./dataset/batches.meta")["label_names"]
    d = unpickle("./dataset/data_batch_1")
    data = d["data"]
    labels = np.array([d["labels"]])
    targets = np.where(labels == 3)[1] # cat
    data = np.asarray([data[idx] for idx in targets]).astype(dtype=np.float32)
    data = data.reshape(3, 32, 32)
    return data.astype(dtype=np.float32)


def unpickle(image):
    import pickle
    f = open(image, 'rb')
    d = pickle.load(f, encoding='latin-1') # oh...
    f.close()
    return d

if __name__ == '__main__':
    load_dataset()
