import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

class DeepfakeDetector:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.Input(shape=(self.img_size[0], self.img_size[1], 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_images(self, folder, label):
        data = []
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            data.append([img, label])
        return data

    def train(self, real_path, fake_path, test_size=0.2, epochs=10):
        real_data = self.load_images(real_path, 0)
        fake_data = self.load_images(fake_path, 1)
        dataset = real_data + fake_data
        np.random.shuffle(dataset)

        print(f"Loaded {len(real_data)} real images and {len(fake_data)} fake images.")

        X = np.array([x[0] for x in dataset]).reshape(-1, self.img_size[0], self.img_size[1], 1)
        y = np.array([x[1] for x in dataset])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        early_stop = EarlyStopping(patience=2, restore_best_weights=True)
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=[early_stop])
        return self.model

    def predict(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Could not load image"
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = img.reshape(1, self.img_size[0], self.img_size[1], 1)
        prediction = self.model.predict(img, verbose=0)
        confidence = prediction[0][0]
        return f"Fake ({confidence:.2f})" if confidence > 0.3 else f"Real ({1 - confidence:.2f})"

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Cannot open video"

        frame_count = 0
        fake_count = 0
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0: 
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.img_size)
            gray = gray.astype('float32') / 255.0
            gray = gray.reshape(1, self.img_size[0], self.img_size[1], 1)

            prediction = self.model.predict(gray, verbose=0)
            if prediction[0][0] > 0.5:
                fake_count += 1
            total_frames += 1

        cap.release()
        fake_percentage = (fake_count / total_frames) * 100 if total_frames > 0 else 0
        return f"Fake Content Detected in {fake_percentage:.2f}% of sampled frames"

if __name__ == "__main__":
    detector = DeepfakeDetector()

    real_folder = r"E:\Deepfake_Detection\Test\Real"
    fake_folder = r"E:\Deepfake_Detection\Train\Fake"
    test_image_path = r"E:\Deepfake_Detection\archive(1)\Sample_fake_images\Sample_fake_images\fake\IMG-20250106-WA0009.jpg"
    test_video_path = r"E:\Deepfake_Detection\Dataset1\DFDCDFDC\test_video\VID-20250129-WA0001.mp4"

    print("Training started...")
    detector.train(real_folder, fake_folder, epochs=10)
    print("Training completed.")

    detector.model.save("deepfake_model.h5")
    print("Model saved as deepfake_model.h5")

    print("Image Prediction:", detector.predict(test_image_path))
    print("Video Prediction:", detector.predict_video(test_video_path))
