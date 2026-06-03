import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import numpy as np

# --- Configuration ---
DATA_PATH = 'datasets/Training_augmented.csv'  # Use augmented dataset for better predictions
MODEL_DIR = 'models'

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

try:
    # 1. Load Data
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} samples, {len(df.columns)-1} symptoms, {df['prognosis'].nunique()} diseases")

    # 2. Separate Features (X) and Target (y)
    X = df.iloc[:, :-1] 
    y = df.iloc[:, -1]  

    # 3. Encode the Target Variable (Disease names to numbers)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Split Data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Train the Model with symptom-aware confidence parameters
    print("\nTraining Random Forest model with symptom-aware confidence...")
    model = RandomForestClassifier(
        n_estimators=200,           # Balanced number of trees
        max_depth=12,               # Moderate depth for good generalization
        min_samples_split=4,        # Require reasonable samples to split
        min_samples_leaf=3,         # Require multiple samples in leaves
        max_features='sqrt',        # Square root of features
        random_state=42,
        n_jobs=-1,                  # Use all CPU cores
        class_weight='balanced',    # Handle class imbalance
        bootstrap=True,             # Enable bootstrap sampling
        oob_score=True,             # Out-of-bag scoring
        criterion='gini'            # Gini impurity for better probability estimates
    )
    model.fit(X_train, y_train)

    # 6. Evaluate the Model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Testing accuracy
    test_accuracy = model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
    
    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, X, y_encoded, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Out-of-bag score
    print(f"Out-of-Bag Score: {model.oob_score_*100:.2f}%")
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    # Feature importance (top 10 symptoms)
    feature_importance = pd.DataFrame({
        'symptom': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Symptoms:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['symptom']}: {row['importance']:.4f}")
    
    # Per-disease accuracy
    print("\nPer-Disease Prediction Summary:")
    diseases_unique = le.inverse_transform(np.unique(y_test))
    for disease in diseases_unique[:5]:  # Show first 5
        disease_idx = le.transform([disease])[0]
        disease_mask = y_test == disease_idx
        if disease_mask.sum() > 0:
            disease_acc = (y_pred[disease_mask] == y_test[disease_mask]).mean()
            print(f"  {disease}: {disease_acc*100:.1f}%")
    print("  ...")

    # 7. Save the Model and Encoder
    pickle.dump(model, open(os.path.join(MODEL_DIR, 'disease_model.pkl'), 'wb'))
    pickle.dump(le, open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb'))
    
    print("\n" + "="*60)
    print("✓ Model and Label Encoder saved successfully!")
    print(f"✓ Model file: {MODEL_DIR}/disease_model.pkl")
    print(f"✓ Encoder file: {MODEL_DIR}/label_encoder.pkl")
    print("="*60)

except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_PATH}. Check your file paths.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()