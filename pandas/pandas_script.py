# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

dataframe_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'

# %%
df = pd.read_csv(dataframe_url)
print(f"The dataframe contains the following information:\n {df.columns}")

# Check total number of rows and columns in the dataframe.
print(f"Total number of samples is '{df.shape[0]}' and there are '{len(df.columns[:-1])}' attributes to inference the '{df.columns[-1]}' column.", end="\n\n")

# Obtain a brief look of the dataframe.
print(f"Visualize the first few samples:\n{df.head()}", end="\n\n")

# Check if any entry contains a NULL value.
print(f"Does any entry contain a NULL value?\n{df.isnull().sum()}")
# Eliminate row if that is the case.
df.dropna(how='all', axis=0, inplace=True)

# %%
# Visualize the data in a histogram ...
df.hist(backend='matplotlib')
plt.show()

# ... or in Pie Chart (last column)
df[df.columns[-1]].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title(f"{df.columns[-1]}")
plt.ylabel("") # Hide vertical label.
plt.show()

# %%
# Convert label into numeric values.
label = df.columns[-1]
df[label] = df[label].astype('category').cat.codes

# Divide dataframe.
data = df.drop(label, axis=1).values
target = df[label].values

# Efectuate the partition of the dataset into training and testing data.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=42)
print(f"Training data shape: {x_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing target shape: {y_test.shape}")

# %%
# Convert data into Pytorch tensors.
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create datasets and loaders. 
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Network internal structure.
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# %%
# Instantiate the model itself.
model = Classifier(input_dim=x_train.shape[1], 
                   hidden_dim=64, 
                   output_dim=len(np.unique(target))
                   )
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING MODE.
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward pass.
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# EVALUATION MODE.
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

torch.save(model, 'iris_model.pth')


