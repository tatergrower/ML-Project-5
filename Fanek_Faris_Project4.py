from LogisticRegression import SoftmaxRegression
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time


def load_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)

    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

X_train = load_images("train-images.idx3-ubyte")
y_train = load_labels("train-labels.idx1-ubyte")

X_test = load_images("t10k-images.idx3-ubyte")
y_test = load_labels("t10k-labels.idx1-ubyte")

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = SoftmaxRegression(lr=0.1, epochs=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

start = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start


cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Train:", train_time)
print("Test:", test_time)



filenames = [f"img_{i}" for i in range(len(y_pred))]

df = pd.DataFrame({
    "filename": filenames,
    "label": y_pred
})

df.to_excel("mnist_output.xlsx", index=False)

print(df["label"].value_counts())

#test
print(X_train.shape)  # should be (60000, 28, 28)
print(y_train.shape)  # should be (60000,)

