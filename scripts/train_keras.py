import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# === Yapılandırma ===
# Yeni organize ettiğimiz final veri seti yolu
DATA_PATH = r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier_final"
img_size = (128, 128) # Daha iyi detay için 64'ten 128'e çıkardım
batch_size = 32
epochs = 20 # Veri setimiz kaliteli olduğu için 20 epoch idealdir

# === Veri Yükleyiciler (Data Generators) ===
# Keras'ın bu özelliği küçük veri setlerini çeşitlendirmek için harikadır
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_PATH, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary' # 0: Adult, 1: Child
)

valid_gen = valid_datagen.flow_from_directory(
    os.path.join(DATA_PATH, "valid"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# === CNN Model Mimarisi ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5), # Ezberlemeyi (overfitting) önlemek için
    Dense(1, activation='sigmoid') # İhtimalleri 0-1 arasına sıkıştırır
])

# === Derleme ===
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Eğitim ===
print("🚀 Keras ile eğitim başlıyor...")
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=epochs
)

# === Modeli ve Grafikleri Kaydet ===
model.save("models/keras_child_adult_model.h5")
print("✅ Keras modeli 'models/keras_child_adult_model.h5' olarak kaydedildi.")

# Doğruluk Grafiği
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title('Doğruluk (Accuracy)')
plt.legend()

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Kayıp (Loss)')
plt.legend()

plt.savefig("models/keras_training_results.png")
plt.show()