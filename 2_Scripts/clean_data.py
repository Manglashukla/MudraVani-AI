import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = "isl_data.csv"
OUTPUT_FILE = "isl_data_CLEANED.csv"

print("Loading your massive dataset...")
df = pd.read_csv(INPUT_FILE, dtype={'label': str}, low_memory=False)

print(f"Original Row Count: {len(df)}")
print(f"Original Class Count: {df['label'].nunique()} (Too many!)")

# --- THE FILTERING LOGIC ---
# We only want labels that are:
# 1. Alphabets (A-Z) OR
# 2. Real words (not numbers like '03156')

def is_valid_label(label):
    # If the label is a number (e.g., "03156"), TRASH IT.
    if label.isdigit():
        return False
    # If the label is a mix like "User_10", TRASH IT.
    if "User" in label or "Sample" in label:
        return False
    # Keep everything else (A, B, Hello, Thanks...)
    return True

# Apply the filter
print("Filtering out garbage numeric labels...")
df_clean = df[df['label'].apply(is_valid_label)]

print(f"\n--- RESULTS ---")
print(f"Cleaned Row Count: {len(df_clean)}")
print(f"Cleaned Class Count: {df_clean['label'].nunique()}")
print("Classes kept:", df_clean['label'].unique())

# Save the new clean file
df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Success! Saved cleaned data to: {OUTPUT_FILE}")
print("Now use THIS file for training.")