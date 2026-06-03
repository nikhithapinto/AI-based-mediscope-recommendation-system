import os
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("="*60)
print("CREATING NEW DISEASE PREDICTOR PROJECT")
print("="*60)

# Create project directory on desktop
desktop = r"C:\Users\lenovo\Desktop"
project_dir = os.path.join(desktop, "DiseasePredictorNew")

# Remove old project if exists
if os.path.exists(project_dir):
    shutil.rmtree(project_dir)
    print(f"✓ Removed old project")

# Create directories
os.makedirs(project_dir, exist_ok=True)
os.makedirs(os.path.join(project_dir, "datasets"), exist_ok=True)
os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)

print(f"✓ Created project at: {project_dir}")

# Copy existing datasets
print("\n📁 Copying datasets...")
datasets_to_copy = [
    'datasets/disease_precautions.csv',
    'datasets/hospitals_with_names.csv',
    'datasets/Training.csv'
]

for dataset in datasets_to_copy:
    if os.path.exists(dataset):
        shutil.copy(dataset, os.path.join(project_dir, dataset))
        print(f"  ✓ Copied {dataset}")

# Create clean training data
print("\n📊 Creating clean training dataset...")

disease_patterns = {
    'Heart attack': {
        'primary': ['chest_pain', 'breathlessness', 'sweating', 'vomiting'],
        'secondary': ['neck_pain', 'weakness_in_limbs', 'fast_heart_rate']
    },
    'Diabetes': {
        'primary': ['increased_appetite', 'polyuria', 'fatigue', 'weight_loss'],
        'secondary': ['excessive_hunger', 'blurred_and_distorted_vision', 'irregular_sugar_level']
    },
    'Migraine': {
        'primary': ['headache', 'acidity', 'indigestion', 'blurred_and_distorted_vision'],
        'secondary': ['nausea', 'visual_disturbances', 'irritability']
    },
    'Pneumonia': {
        'primary': ['chest_pain', 'fast_heart_rate', 'rusty_sputum', 'chills'],
        'secondary': ['high_fever', 'breathlessness', 'cough', 'phlegm']
    },
    'GERD': {
        'primary': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'cough'],
        'secondary': ['vomiting', 'chest_pain', 'indigestion']
    },
    'Common Cold': {
        'primary': ['phlegm', 'throat_irritation', 'sinus_pressure', 'runny_nose'],
        'secondary': ['congestion', 'redness_of_eyes', 'continuous_sneezing']
    },
    'Allergy': {
        'primary': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
        'secondary': ['runny_nose', 'congestion', 'itching']
    },
    'Fungal infection': {
        'primary': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic _patches'],
        'secondary': ['skin_peeling', 'red_sore_around_nose']
    },
    'Malaria': {
        'primary': ['high_fever', 'chills', 'vomiting', 'muscle_pain'],
        'secondary': ['sweating', 'headache', 'nausea', 'fatigue']
    },
    'Dengue': {
        'primary': ['headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes'],
        'secondary': ['high_fever', 'muscle_pain', 'joint_pain', 'vomiting']
    },
    'Typhoid': {
        'primary': ['chills', 'fatigue', 'high_fever', 'vomiting'],
        'secondary': ['headache', 'constipation', 'abdominal_pain', 'diarrhoea']
    },
    'Gastroenteritis': {
        'primary': ['diarrhoea', 'vomiting', 'sunken_eyes', 'dehydration'],
        'secondary': ['abdominal_pain', 'nausea', 'stomach_pain']
    },
    'Jaundice': {
        'primary': ['itching', 'vomiting', 'yellowish_skin', 'yellowing_of_eyes'],
        'secondary': ['fatigue', 'weight_loss', 'dark_urine', 'loss_of_appetite']
    },
    'Hepatitis C': {
        'primary': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite'],
        'secondary': ['yellowing_of_eyes', 'abdominal_pain', 'dark_urine']
    },
    'Urinary tract infection': {
        'primary': ['bladder_discomfort', 'continuous_feel_of_urine', 'burning_micturition', 'foul_smell_of urine'],
        'secondary': ['abdominal_pain']
    },
    'Hypertension': {
        'primary': ['headache', 'chest_pain', 'loss_of_balance', 'lack_of_concentration'],
        'secondary': ['dizziness', 'blurred_and_distorted_vision']
    },
    'Arthritis': {
        'primary': ['muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness'],
        'secondary': ['joint_pain', 'knee_pain', 'hip_joint_pain']
    },
    'Osteoarthritis': {
        'primary': ['joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain'],
        'secondary': ['swelling_joints', 'movement_stiffness', 'back_pain']
    },
    'Acne': {
        'primary': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
        'secondary': ['skin_peeling', 'inflammatory_nails']
    },
    'Psoriasis': {
        'primary': ['skin_rash', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails'],
        'secondary': ['joint_pain', 'inflammatory_nails', 'itching']
    }
}

# Load original to get all symptoms
original_df = pd.read_csv('datasets/Training.csv')
all_symptoms = [col for col in original_df.columns if col != 'prognosis']

# Create training data
training_data = []
for disease, patterns in disease_patterns.items():
    for i in range(150):
        row = {symptom: 0 for symptom in all_symptoms}
        
        # Add all primary symptoms
        for symptom in patterns['primary']:
            if symptom in all_symptoms:
                row[symptom] = 1
        
        # Add 1-2 random secondary symptoms
        if patterns.get('secondary'):
            num_secondary = np.random.randint(1, min(3, len(patterns['secondary']) + 1))
            selected = np.random.choice(patterns['secondary'], size=num_secondary, replace=False)
            for symptom in selected:
                if symptom in all_symptoms:
                    row[symptom] = 1
        
        row['prognosis'] = disease
        training_data.append(row)

df = pd.DataFrame(training_data)
df = df[all_symptoms + ['prognosis']]
df.to_csv(os.path.join(project_dir, 'datasets', 'Training_clean.csv'), index=False)
print(f"  ✓ Created Training_clean.csv ({len(df)} samples, {df['prognosis'].nunique()} diseases)")

# Train model
print("\n🤖 Training model...")
X = df.drop('prognosis', axis=1)
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)
model.fit(X, y_encoded)

# Save model
with open(os.path.join(project_dir, 'models', 'disease_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(project_dir, 'models', 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print(f"  ✓ Model trained and saved")

# Test model
test_symptoms = ['chest_pain', 'breathlessness', 'sweating', 'vomiting']
test_vec = np.zeros(len(X.columns))
for s in test_symptoms:
    if s in X.columns:
        test_vec[X.columns.get_loc(s)] = 1

pred = model.predict(test_vec.reshape(1, -1))[0]
probs = model.predict_proba(test_vec.reshape(1, -1))[0]
predicted = le.inverse_transform([pred])[0]
conf = probs[pred] * 100

print(f"\n✅ Model Test:")
print(f"  Symptoms: {test_symptoms}")
print(f"  Predicted: {predicted} ({conf:.1f}%)")

print("\n" + "="*60)
print(f"✅ PROJECT CREATED SUCCESSFULLY!")
print(f"📁 Location: {project_dir}")
print("="*60)
