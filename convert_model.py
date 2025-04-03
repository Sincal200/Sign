import tensorflow as tf
import os

# Path to your model
model_path = "test.h5"  # Your H5 model file

# Check if file exists
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    print("Available files:")
    for file in os.listdir():
        print(f"- {file}")
    exit(1)

# Load the model
model = tf.keras.models.load_model(model_path)

# Export as SavedModel format (new way in Keras 3)
saved_model_dir = "tf_saved_model"
model.export(saved_model_dir)  # Use export instead of save

print(f"Model exported as SavedModel to {saved_model_dir}")
print("Now run this command:")
print("tensorflowjs_converter --input_format=tf_saved_model tf_saved_model salida")