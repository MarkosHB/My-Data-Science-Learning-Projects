# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import autokeras as ak
import tensorflow as tf
import warnings

# (Optional) Hide warnings messages.
warnings.simplefilter("ignore")
tf.get_logger().setLevel('ERROR') 

# %%
ds = datasets.load_breast_cancer(as_frame=True)
print(f"The dataset contains the following information: {ds.keys()}")

# Check total number of rows and columns in the dataset.
print(f"Total number of samples inside data is '{ds.data.shape[0]}' and there are '{ds.data.shape[1]}' attributes to predict the '{ds.target.name}' column.", end="\n\n")

# Obtain a brief look of the dataset.
print(f"Visualize the first few samples: \n {ds.frame.head()}")

# %%
# Efectuate the partition of the dataset into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# %%
# Define layer structure.
input_node = ak.Input()
layer1 = ak.DenseBlock()(input_node)
output_node = ak.ClassificationHead()(layer1)

# Create the model itself and train it. 
model = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
model.fit(x_train.values, y_train.values, epochs=10)

# %%
# Make some predictions.
predictions = model.predict(x_test.values)

# Visualize results in a confusion matrix.
conf_matrix = confusion_matrix(y_test, predictions)
print(f"Confusion matrix:\n{conf_matrix}")

# Calcular la precisión
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy:\n{accuracy:.4f}")


