import os
import cv2
import mediapipe as mp
import csv

# --- CONFIGURATION ---
DATA_DIR = "../1_Raw_Dataset"
OUTPUT_FILE = "isl_data.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
# static_image_mode=False is better for videos, but True is safer for mixed data.
# We will use False (Stream mode) for videos manually.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Setup CSV File
header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

def process_frame(image, label, writer):
    """Helper function to extract landmarks from a single frame/image"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row = [label]
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            writer.writerow(row)

print("Creating Dataset... This might take a while.")

with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # LOOP 1: PROCESS STATIC IMAGE FOLDERS (A-Z)
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        
        # Skip the 'videos' folder for now, we handle it later
        if folder_name == "videos" or not os.path.isdir(folder_path):
            continue
            
        print(f"Processing Image Folder: {folder_name}...")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            process_frame(img, folder_name, writer)

    # LOOP 2: PROCESS VIDEO FILES
    video_folder = os.path.join(DATA_DIR, "videos")
    if os.path.exists(video_folder):
        print(f"Processing Videos in: {video_folder}...")
        
        for video_file in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video_file)
            
            # Use filename as label (e.g., "Hello.mp4" -> "Hello")
            label = os.path.splitext(video_file)[0]
            
            # Open Video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                # OPTIMIZATION: Only take every 5th frame to avoid duplicates
                if frame_count % 5 == 0:
                    process_frame(frame, label, writer)
                
                frame_count += 1
            
            cap.release()
            print(f"  -> Processed video: {label} ({frame_count} frames scanned)")

print(f"Success! All data saved to {OUTPUT_FILE}")