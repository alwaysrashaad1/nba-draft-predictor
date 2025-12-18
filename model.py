import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1. Load dataset
# ---------------------------
# Load dataset containing player information and draft pick.
# Expected columns include 'draft_pick', 'player_name', and other features.
df = pd.read_csv("data/dataset.csv")

# Replace undrafted players (-1) with 61 to handle them numerically as the last pick.
df['draft_pick_cont'] = df['draft_pick'].replace(-1, 61).astype(float)

# ---------------------------
# 2. Prepare features
# ---------------------------
# Drop target column and identifier column
X = df.drop(columns=['draft_pick', 'player_name'])

# Convert categorical variables to one-hot encoding
X = pd.get_dummies(X)

# ---------------------------
# 3. Train/test split
# ---------------------------
# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, df['draft_pick_cont'].values, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Scale features
# ---------------------------
# Standardize features to have mean=0 and std=1 for better neural network training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# 5. Scale target to 0–1
# ---------------------------
# Since the model output uses a sigmoid activation, scale target to 0–1.
# 1 maps to 0, 61 maps to 1
y_train_scaled = (y_train - 1) / 60
y_test_scaled  = (y_test - 1) / 60

# Convert features and target to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_scaled  = torch.tensor(y_test_scaled, dtype=torch.float32)

# ---------------------------
# 6. Define the model
# ---------------------------
class ContinuousDraftModel(nn.Module):
    """
    Fully-connected neural network predicting draft pick as a continuous value.
    Architecture: 4 hidden layers with batch normalization, dropout, and ReLU activations.
    Output layer uses sigmoid to constrain predictions to the range [0, 1].
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()  # constrain output to 0–1
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # remove extra dimension for single output

# Initialize the model
model = ContinuousDraftModel(X_train.shape[1])

# ---------------------------
# 7. Optimizer and loss function
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Mean squared error for regression task

# ---------------------------
# 8. Training loop
# ---------------------------
epochs = 150
for epoch in range(epochs):
    model.train()           # set model to training mode
    optimizer.zero_grad()   # reset gradients

    outputs = model(X_train)             # forward pass
    loss = loss_fn(outputs, y_train_scaled)  # compute loss

    loss.backward()        # backpropagation
    optimizer.step()       # update weights

    if (epoch+1) % 10 == 0:  # print every 10 epochs
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ---------------------------
# 9. Evaluation
# ---------------------------
model.eval()  # set model to evaluation mode
with torch.no_grad():
    preds_scaled = model(X_test)

# Rescale predictions to original draft pick scale (1–61)
preds = preds_scaled * 60 + 1
preds_rounded = torch.round(preds).long()  # round to nearest integer pick

# Clamp predictions greater than 60 as undrafted (61)
preds_clamped = preds_rounded.clone()
preds_clamped[preds_clamped > 60] = 61

# Clamp true values similarly
y_true = torch.tensor(y_test, dtype=torch.long)
y_true_clamped = y_true.clone()
y_true_clamped[y_true_clamped > 60] = 61

# Compute exact and ±n pick accuracy
exact_acc = (preds_clamped == y_true_clamped).float().mean().item()
within_1 = (torch.abs(preds_clamped - y_true_clamped) <= 1).float().mean().item()
within_2 = (torch.abs(preds_clamped - y_true_clamped) <= 2).float().mean().item()
within_5 = (torch.abs(preds_clamped - y_true_clamped) <= 5).float().mean().item()

print(f"\nExact Pick Accuracy: {exact_acc*100:.2f}%")
print(f"±1 Pick Accuracy: {within_1*100:.2f}%")
print(f"±2 Pick Accuracy: {within_2*100:.2f}%")
print(f"±5 Pick Accuracy: {within_5*100:.2f}%")

# ---------------------------
# 10. Save model and scaler
# ---------------------------
torch.save(model.state_dict(), "continuous_draft_model.pt")
joblib.dump(scaler, "continuous_scaler.pkl")
print("\nModel and scaler saved.")

# ---------------------------
# 11. Permutation feature importance
# ---------------------------
def permutation_importance(model, X, y, metric=mean_squared_error, n_repeats=10, random_seed=42):
    """
    Compute feature importance using permutation importance:
    1. Measure baseline model performance.
    2. Shuffle each feature column multiple times and evaluate performance.
    3. Importance is the increase in error caused by shuffling the feature.
    
    Args:
        model: trained PyTorch model
        X: feature matrix as NumPy array
        y: true target values
        metric: function to evaluate model error (default: MSE)
        n_repeats: number of shuffles per feature
        random_seed: random seed for reproducibility
    
    Returns:
        importances: NumPy array of mean importance per feature
    """
    np.random.seed(random_seed)
    baseline_preds = model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    baseline_score = metric(y, baseline_preds)
    
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)
    
    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])  # shuffle only this feature
            preds_shuffled = model(torch.tensor(X_shuffled, dtype=torch.float32)).detach().numpy()
            score = metric(y, preds_shuffled)
            scores.append(score)
        importances[i] = np.mean(scores) - baseline_score  # importance = increase in error
    
    return importances

# Convert test set to NumPy arrays
X_test_np = X_test.numpy()
y_test_np = y_test_scaled.numpy()

# Compute permutation importance for all features
feature_importances = permutation_importance(model, X_test_np, y_test_np, mean_squared_error)

# Map feature importances to feature names
feature_names = X.columns
feat_imp_dict = dict(zip(feature_names, feature_importances))

# Sort features by descending importance
feat_imp_sorted = sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True)

# Print top 20 features
print("\nTop 20 Features by Permutation Importance (higher = more impact on draft pick):")
for name, imp in feat_imp_sorted[:20]:
    print(f"{name:30} | {imp:.6f}")
