import numpy as np
from matplotlib import pyplot as plt

filepath = "./full_numpy_bitmap_book.npy"
img = np.load(filepath).astype(np.float32)
test_img = img[33]

categories = ['apple', 'book', 'door']
scores = []
weights = []

for item in categories:
    filepaths = f"./full_numpy_bitmap_{item}.npy"
    tmp_img = np.load(filepaths).astype(np.float32)
    avg_img = np.average(tmp_img, axis=0)
    weights.append(avg_img)
    tmp_score = np.dot(avg_img, test_img)
    scores.append(tmp_score)
print(scores)
print(f"The test image is most likely a {categories[scores.index(max(scores))]}")

plt.figure(figsize=(10, 4))
for i in range(len(weights)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weights[i].reshape(28, 28))
    plt.axis('off')
    plt.title(categories[i])
plt.show()