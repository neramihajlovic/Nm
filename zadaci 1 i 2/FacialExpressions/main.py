import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.src.layers import Flatten
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.layers import BatchNormalization
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, RandomFlip, RandomRotation, RandomZoom

from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.utils import compute_class_weight

test = "C:/Users/PC/Desktop/archive/test"
train = "C:/Users/PC/Desktop/archive/train"

img_size = (48, 48)
color_mode = "grayscale"
batch_size = 32  #mozemo menjati

testData = image_dataset_from_directory(test,
                                        color_mode="grayscale",
                                        labels="inferred",
                                        label_mode="int",
                                        image_size=img_size,
                                        batch_size=batch_size,
                                        shuffle=False)

fullTrain = image_dataset_from_directory(train,
                                        color_mode="grayscale",
                                        labels="inferred",
                                        label_mode="int",
                                        image_size=img_size,
                                        batch_size=batch_size,
                                        shuffle=True)


classes = fullTrain.class_names
num_classes = len(classes)
print("Klase:", classes)

val_size = 0.2
total_batches = tf.data.experimental.cardinality(fullTrain).numpy()
train_batches = int((1 - val_size) * total_batches)
trainData = fullTrain.take(train_batches).map(lambda x, y: (x/255., tf.one_hot(y, num_classes)))
valData = fullTrain.skip(train_batches).take(total_batches - train_batches).map(lambda x, y: (x/255., tf.one_hot(y, num_classes)))
testData = testData.map(lambda x, y: (x/255., tf.one_hot(y, num_classes)))


klase = os.listdir(train)
counts = [len(os.listdir(os.path.join(train, c))) for c in klase]

plt.figure(figsize=(8,5))
plt.bar(klase, counts, color='skyblue')
plt.title("Broj slika po klasama - trening skup")
plt.xlabel("Klasa")
plt.ylabel("Broj slika")
plt.show()

one_per_class = {}
plt.figure(figsize=(14,3))

for images, labels in fullTrain.take(1):
    for img, lab in zip(images, labels):
        lab_int = int(lab.numpy())
        if lab_int not in one_per_class:
            one_per_class[lab_int] = img
    if len(one_per_class) == len(classes):
        break

for i, (lab_int, img) in enumerate(one_per_class.items()):
    plt.subplot(1, len(classes), i+1)
    plt.imshow(img.numpy().astype('uint8'))
    plt.title(classes[lab_int])
    plt.axis('off')

plt.show()


model = Sequential([
    # Data augmentation
    RandomFlip("horizontal", input_shape=(48, 48, 1)),
    RandomRotation(0.1),
    RandomZoom(0.1),

    # 1. blok
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # 2. blok
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 3. blok
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Fully connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)


y_train_classes = np.concatenate([y.numpy() for x, y in trainData], axis=0)
y_train_classes = np.argmax(y_train_classes, axis=1)

weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_classes)
class_weight = dict(enumerate(weights))
#class_weights = {i: 1.0 for i in range(num_classes)}


history = model.fit(
    trainData,
    epochs=15,
    validation_data=valData,
    callbacks=[early_stop],
    verbose=2,
    class_weight=class_weight
)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.show()

train_loss, train_acc = model.evaluate(trainData, verbose=0)
print(f"Trening tačnost: {train_acc:.4f}")

y_true = np.concatenate([y.numpy() for x, y in trainData], axis=0)
y_pred = model.predict(trainData, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Konfuziona matrica trening skupa')
plt.xlabel('Predikcija')
plt.ylabel('Prava klasa')
plt.show()


x_test_np = np.concatenate([x.numpy() for x, y in testData], axis=0)
y_test_np = np.concatenate([y.numpy() for x, y in testData], axis=0)
y_true_classes = np.argmax(y_test_np, axis=1)

y_pred_classes = np.argmax(model.predict(x_test_np, verbose=0), axis=1)

test_loss, test_acc = model.evaluate(x_test_np, tf.one_hot(y_true_classes, num_classes), verbose=0)
print(f"Test tačnost: {test_acc:.4f}")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))


cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Konfuziona matrica test skupa')
plt.xlabel('Predikcija')
plt.ylabel('Prava klasa')
plt.show()

# Prikaz tačno i netačno klasifikovanih primera

correct = np.where(y_pred_classes == y_true_classes)[0]
incorrect = np.where(y_pred_classes != y_true_classes)[0]

num_samples = 5
correct_sample = np.random.choice(correct, size=min(num_samples, len(correct)), replace=False)
incorrect_sample = np.random.choice(incorrect, size=min(num_samples, len(incorrect)), replace=False)

plt.figure(figsize=(12, 6))

for i, idx in enumerate(correct_sample):
    plt.subplot(2, 5, i+1)
    plt.imshow((x_test_np[idx] * 255).astype('uint8'), cmap='gray')
    plt.title(f"Pred: {classes[y_pred_classes[idx]]}\nPrava: {classes[y_true_classes[idx]]}")
    plt.axis('off')

for i, idx in enumerate(incorrect_sample):
    plt.subplot(2, 5, i+6)
    plt.imshow((x_test_np[idx] * 255).astype('uint8'), cmap='gray')
    plt.title(f"Pred: {classes[y_pred_classes[idx]]}\nPrava: {classes[y_true_classes[idx]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
