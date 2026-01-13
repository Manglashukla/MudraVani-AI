import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
CSV_FILE = "isl_data.csv"
MODEL_DIR = "../3_Models"
MODEL_PATH = os.path.join(MODEL_DIR, "isl_model.pkl")

# 1. Load Data
if not os.path.exists(CSV_FILE):
    print("Error: CSV file not found!")
    exit()

print("Loading Dataset...")
# MEMORY FIX 1: Load as float32 to save 50% RAM
data = pd.read_csv(CSV_FILE, dtype={'label': str}, low_memory=False)
data.dropna(inplace=True)

# 2. Separate Features (X) and Labels (y)
# MEMORY FIX 2: Convert coordinates to float32 explicitly
X = data.drop('label', axis=1).astype('float32')
y = data['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train with Memory Limits
print("Training the Brain (Lite Version)...")

# MEMORY FIX 3: Limit the size of the model
model = RandomForestClassifier(
    n_estimators=30,      # Reduced from 100 to 30
    max_depth=10,         # Limit depth to prevent 12GB RAM explosion
    n_jobs=1,             # use 1 core to save memory overhead
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Success! Model saved at: {MODEL_PATH}")