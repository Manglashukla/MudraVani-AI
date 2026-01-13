import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React to communicate

# --- CONFIGURATION ---
MODEL_PATH = "../../3_Models/isl_model_final.pkl"
ENCODER_PATH = "../../3_Models/label_encoder.pkl"

print("Loading AI Brain...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

current_prediction = "Waiting..."

# --- HELPER: EXACT FEATURE EXTRACTION (Matches Training Logic) ---
def extract_features(landmarks):
    """
    Converts MediaPipe landmarks into the exact feature vector
    our XGBoost model expects (Normalized Coords + Distances + Angles).
    """
    # 1. Convert to Numpy (21 points x 3 coords)
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).reshape(21, 3)
    
    # 2. Normalization (Center Wrist at 0,0,0)
    wrist = landmarks_np[0]
    centered = landmarks_np - wrist
    
    # 3. Scale (Make hand size consistent)
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / (max_dist if max_dist > 0 else 1)
    
    # 4. Calculate Distances (Pinch checks)
    # Tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    d1 = np.linalg.norm(normalized[4] - normalized[8])  # Thumb-Index
    d2 = np.linalg.norm(normalized[4] - normalized[12]) # Thumb-Middle
    d3 = np.linalg.norm(normalized[4] - normalized[16]) # Thumb-Ring
    d4 = np.linalg.norm(normalized[4] - normalized[20]) # Thumb-Pinky
    
    # 5. Calculate "Simple Angles" (Is Finger Bent?)
    # Distance from Wrist(0) to Tips
    a1 = np.linalg.norm(normalized[0] - normalized[8])
    a2 = np.linalg.norm(normalized[0] - normalized[12])
    a3 = np.linalg.norm(normalized[0] - normalized[16])
    a4 = np.linalg.norm(normalized[0] - normalized[20])
    
    # 6. Flatten & Combine
    features = np.concatenate([normalized.flatten(), [d1, d2, d3, d4, a1, a2, a3, a4]])
    return features.reshape(1, -1)  # Reshape for model input

# --- VIDEO GENERATOR ---
def generate_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                try:
                    # 1. Extract Features
                    features = extract_features(hand_landmarks.landmark)
                    
                    # 2. Predict
                    prediction_index = model.predict(features)[0]
                    prediction_text = label_encoder.inverse_transform([prediction_index])[0]
                    
                    current_prediction = prediction_text
                    
                    # Draw on screen for debug
                    cv2.putText(frame, prediction_text, (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                except Exception as e:
                    print(f"Prediction Error: {e}")

        else:
            current_prediction = "..."

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': current_prediction})

if __name__ == '__main__':
    app.run(port=5000, debug=True)