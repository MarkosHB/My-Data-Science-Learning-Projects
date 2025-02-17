# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Load the diabetes dataset.
ds = datasets.load_diabetes(as_frame=True)
print(f"The dataset contains the following information: {ds.keys()}")

# Check total number of rows and columns in the dataset.
print(f"Total number of samples inside data is '{ds.data.shape[0]}' and there are '{ds.data.shape[1]}' attributes to predict the '{ds.target.name}' column.", end="\n\n")

# Obtain a brief look of the dataset.
print(f"Visualize the first few samples: \n {ds.frame.head()}")

# (Optional) See dataset description.
# print(f"Dataset contextual information:\n {df.DESCR}")

# %%
# Efectuate the partition of the dataset into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# %%
# Define the model using the Tensorflow library.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Regression output.
])

# Compile the model.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

# Train the model.
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model.
loss, mae, mse = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {loss} \nTest MAE: {mae}")

# Save the trained model.
model.save('diabetes_model.keras')
# model.summary()

# %%
# Show the metrics obtained during traning process.
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([40,180])
  plt.legend()

plot_history(history)

# %%
# Valorate the model performance.
std_dev = np.std(y_test)
print(f"Standard deviation of the real values: {std_dev}")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"R² (the model is better when closer to 1): {r2}")

# Visualize the Gaussian error.
error = y_pred.flatten() - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


