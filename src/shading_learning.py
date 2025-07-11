import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class ShadingMLP(nn.Module):
    def __init__(self, input_size):
        super(ShadingMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


def train_shading_model(cluster_results, model_path):
    """Train shading detection model and save it"""
    
    # Fixed test plants for consistent evaluation
    test_plants = ["gw-0002", "gw-0019", "gw-0025", "gw-0035", "gw-0040", 
                   "gw-0048", "gw-0053", "gw-0057", "gw-0065", "gw-0097"]
    
    # Prepare data with plant-based split
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    for plant_id, data in cluster_results.items():
        cluster_labels = data['cluster_labels']
        features = data['features']
        
        for i, label in enumerate(cluster_labels):
            # Label as shaded if cluster is 1 or higher (representing shading)
            is_shaded = 1 if label >= 1 else 0
            
            if plant_id in test_plants:
                X_test.append(features[i])
                y_test.append(is_shaded)
            else:
                X_train.append(features[i])
                y_train.append(is_shaded)
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize model
    model = ShadingMLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs > 0.5).float()
        test_accuracy = (test_predictions == y_test_tensor).float().mean()
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model and scaler
    model_data = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_size': X_train.shape[1]
    }
    torch.save(model_data, model_path)
    print(f"Model saved to: {model_path}")


def load_shading_model(model_path):
    """Load trained shading detection model"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = ShadingMLP(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {
        'model': model,
        'scaler': checkpoint['scaler']
    }


def predict_shading(model_dict, feature_vectors):
    """Predict shading levels for given feature vectors"""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    predictions = {}
    
    for plant_id, features in feature_vectors.items():
        daily_predictions = {}
        
        for date, row in features.iterrows():
            # Prepare features
            feature_array = row.values.reshape(1, -1)
            feature_scaled = scaler.transform(feature_array)
            feature_tensor = torch.FloatTensor(feature_scaled)
            
            # Predict
            with torch.no_grad():
                prediction = model(feature_tensor)
                shading_class = int(prediction.item() > 0.5)  # Binary classification
            
            daily_predictions[str(date)] = shading_class
        
        predictions[plant_id] = daily_predictions
    
    return predictions
