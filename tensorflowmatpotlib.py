import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ===============================
# 1. Load Dataset
# ===============================
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize Images
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ===============================
# 2. Show Sample Images
# ===============================
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# ===============================
# 3. Build CNN Model
# ===============================
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

# ===============================
# 4. Compile Model
# ===============================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 5. Train Model
# ===============================
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# ===============================
# 6. Evaluate Model
# ===============================
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy:", test_acc)

# ===============================
# 7. Save Model
# ===============================
model.save("cnn_model.h5")

# ===============================
# 8. Load Saved Model
# ===============================
new_model = tf.keras.models.load_model("cnn_model.h5")

# ===============================
# 9. Predict
# ===============================
predictions = new_model.predict(test_images)

print("Predicted Class:", class_names[predictions[0].argmax()])
print("Actual Class:", class_names[test_labels[0][0]])

# ===============================
# 10. Accuracy Graph
# ===============================
plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()

# ===============================
# 11. Loss Graph
# ===============================
plt.figure(figsize=(8,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()

# ===============================
# 12. Predict Custom Images
# ===============================
import numpy as np

sample = test_images[15]

prediction = new_model.predict(np.expand_dims(sample, axis=0))

predicted_class = class_names[np.argmax(prediction)]

plt.imshow(sample)
plt.title("Prediction : " + predicted_class)
plt.axis("off")
plt.show()

print("Prediction:", predicted_class)