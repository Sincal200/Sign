# Create a file named convert_to_tfjs.py with this content:
import tensorflow as tf
import tensorflowjs as tfjs

# Load the model
model = tf.keras.models.load_model('test.h5')

# Convert and save as TensorFlow.js format
tfjs.converters.save_keras_model(model, 'salida')
print("Model successfully converted to TensorFlow.js format in 'salida' directory")