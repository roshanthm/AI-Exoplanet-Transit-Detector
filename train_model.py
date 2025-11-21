import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_POINTS = 500
TRAIN_SIZE = 6000
VAL_SIZE = 1000

def generate_light_curve(num_points, has_transit):
    curve = np.ones(num_points)

    # Random baseline noise
    noise = np.random.uniform(0.01, 0.10)
    curve += np.random.normal(0, noise, num_points)

    if has_transit:
        depth = np.random.uniform(0.01, 0.10)
        duration = np.random.randint(30, 120)
        center = np.random.randint(num_points//4, 3*num_points//4)

        # Transit dip with sloped edges
        for i in range(num_points):
            d = abs(i - center)

            if d < duration // 2:
                curve[i] -= depth
            elif d < duration // 2 + np.random.randint(10, 20):
                slope = 1 - ((d - duration // 2) / np.random.uniform(10, 20))
                curve[i] -= depth * slope

        # Random stellar activity bumps
        for _ in range(np.random.randint(2, 6)):
            bump_pos = np.random.randint(0, num_points)
            bump_size = np.random.uniform(-0.01, 0.01)
            curve[bump_pos:bump_pos+5] += bump_size

    # Normalize
    curve = (curve - np.mean(curve)) / (np.std(curve) + 1e-6)
    return curve

# Create dataset
X, y = [], []
for _ in range(TRAIN_SIZE + VAL_SIZE):
    label = np.random.choice([0, 1])
    X.append(generate_light_curve(NUM_POINTS, label))
    y.append(label)

X = np.array(X)[..., np.newaxis]  # CNN expects channel dimension
y = np.array(y)

X_train, X_val = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
y_train, y_val = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

# Build CNN model
model = keras.Sequential([
    layers.Input(shape=(NUM_POINTS, 1)),
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 5, activation="relu", padding="same"),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 5, activation="relu", padding="same"),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=32,
    verbose=1
)

model.save("exoplanet_transit_model.keras")
print("\n✔ Realistic CNN model saved: exoplanet_transit_model.keras")
