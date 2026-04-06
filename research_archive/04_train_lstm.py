import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load Time-Series Data
df = pd.read_csv('data/aqi_timeseries_dataset.csv')
X = df.drop('AQI_Target', axis=1).values
y = df['AQI_Target'].values

# 2. Scaling (Essential for Neural Networks)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 3. Reshape for PyTorch [Batch, Sequence, Features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 4. Define the LSTM Brain Structure
class AQI_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AQI_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the Model
model = AQI_LSTM(input_size=X_train.shape[2], hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. The Training Loop
epochs = 100
train_losses = []

print("🧠 Starting PyTorch Deep Learning Training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward Pass (The Learning part)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# 6. Evaluation and Saving
print("✅ Training Complete!")
torch.save(model.state_dict(), 'models/lstm_aqi_model.pth')
print("💾 Model saved as 'lstm_aqi_model.pth'")

# Plot the learning curve
plt.plot(train_losses)
plt.title('PyTorch Training Loss (Learning Progress)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

import joblib
# Save the scalers so the App can use them
joblib.dump(scaler_x, 'models/scaler_x.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
print("💾 Scalers saved to models folder!")