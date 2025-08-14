import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import keras_tuner as kt
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load Dataset

df = pd.read_csv("physical_drill_string_dataset_with_risk.csv")

df = df.drop(columns=["Risk_Level"])

# Features and target
X = df.drop(columns=["Failure"])
y = df["Failure"]

# Ensure target is int
y = y.astype(int)

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# Handle Class Imbalance

smote = SMOTE(random_state=42)
under = RandomUnderSampler(random_state=42)
imb_pipeline = ImbPipeline(steps=[('smote', smote), ('under', under)])

X_train_res, y_train_res = imb_pipeline.fit_resample(
    preprocessor.fit_transform(X_train), y_train
)

X_test_proc = preprocessor.transform(X_test)


# Model Building Function

def build_model(hp):
    model = keras.Sequential()
    input_shape = X_train_res.shape[1]

    # First layer
    model.add(layers.Input(shape=(input_shape,)))

    # Hidden layers
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice(f'act_{i}', values=['relu', 'tanh', 'selu']),
            kernel_regularizer=regularizers.l2(
                hp.Float(f'l2_{i}', 1e-5, 1e-2, sampling='log')
            )
        ))
        model.add(layers.Dropout(
            hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        ))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner Setup

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_dir',
    project_name='drill_string_failure'
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)

# Search

tuner.search(
    X_train_res, y_train_res,
    validation_split=0.15,
    epochs=100,
    batch_size=None,  # Let tuner decide
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Get Best Model

best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate
y_pred_probs = best_model.predict(X_test_proc).ravel()
y_pred = (y_pred_probs >= 0.5).astype(int)

print("\nClassification report (test):")
print(classification_report(y_test, y_pred))
print("ROC AUC (test):", roc_auc_score(y_test, y_pred_probs))


# Save Model & Artifacts

model_path = "ann_failure_prediction.keras"
best_model.save(model_path)

artifacts = {
    'model_path': model_path,
    'preprocessor': preprocessor,
    'features': numeric_features + categorical_features
}

with open('failure_prediction_ann.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\nSaved best Keras model to: {model_path}")
print("Saved artifacts to: failure_prediction_ann.pkl")


