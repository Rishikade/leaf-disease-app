import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/leaf_model.h5")

classes = ['early_blight', 'healthy', 'late_blight']

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.reshape(img, (1,224,224,3))

prediction = model.predict(img)

result = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", result)
print("Confidence:", confidence)
