from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
from collections import defaultdict, deque

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
model = load_model('action (1).h5')

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions your model can predict
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Store sequences for each client session
client_sequences = defaultdict(lambda: deque(maxlen=30))
client_predictions = defaultdict(list)

# Reuse your existing functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.json:
        return jsonify({'error': 'No frame data found'}), 400

    client_id = request.json.get('client_id', 'default')
    frame_data = request.json['frame']

    # Decode base64 image
    img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Process the frame with MediaPipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)

        # Extract keypoints
        keypoints = extract_keypoints(results)

        # Add to this client's sequence
        client_sequences[client_id].append(keypoints)

        response = {
            'sequence_length': len(client_sequences[client_id]),
            'prediction': None,
            'confidence': None,
        }

        # If we have enough frames, make a prediction
        if len(client_sequences[client_id]) == 30:
            sequence = list(client_sequences[client_id])
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            confidence = float(res[np.argmax(res)])

            # Track predictions for this client
            client_predictions[client_id].append(np.argmax(res))
            if len(client_predictions[client_id]) > 10:
                client_predictions[client_id].pop(0)

            # Only return confident predictions with consensus
            if (len(client_predictions[client_id]) >= 5 and
                np.unique(client_predictions[client_id][-5:])[0] == np.argmax(res) and
                confidence > 0.7):
                response['prediction'] = predicted_action
                response['confidence'] = confidence

        return jsonify(response)

@app.route('/reset', methods=['POST'])
def reset_sequence():
    client_id = request.json.get('client_id', 'default')
    if client_id in client_sequences:
        client_sequences[client_id].clear()
    if client_id in client_predictions:
        client_predictions[client_id].clear()
    return jsonify({'status': 'reset successful'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)