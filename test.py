import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("fire_severity_classifier.h5")

# Load and preprocess a test image
image_path = 'resources/test/abc008.jpg'  # Replace with the path to your test image
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make a prediction
predictions = model.predict(img_array)
class_labels = ['Low', 'Moderate', 'Severe', 'No Fire']  # Updated with "No Fire"
severity = class_labels[np.argmax(predictions)]
print(f"Predicted fire severity: {severity}")
