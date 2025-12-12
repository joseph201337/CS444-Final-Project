import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocess import get_data_loaders
from models import RainANN, RainLSTM, RainGRU
import numpy as np
import time

# SETTINGS
DATA_PATH = 'data/weatherAUS.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
BATCH_SIZE = 64  

# Define Regions for regional models
COASTAL_CITIES = ['Sydney', 'CoffsHarbour', 'Wollongong', 'GoldCoast', 'Cairns']
INLAND_CITIES = ['AliceSprings', 'Moree', 'Woomera', 'Uluru']

def train_model(model, train_loader, test_loader, model_name="Model"):
    """Train model with AdamW optimizer and track all metrics"""
    pos_weight = torch.tensor([3.3]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    model.to(DEVICE)
    print(f"\n--- Training {model_name} on {DEVICE} ---")

    best_f1 = 0.0
    best_acc = 0.0
    best_auc = 0.0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation every 2 epochs
        if (epoch+1) % 2 == 0:
            val_acc, val_f1, val_auc = evaluate_model(model, test_loader)
            print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
            
            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                best_auc = val_auc
                torch.save(model.state_dict(), f"{model_name}_best.pth")
                print(f"  >>> New Best F1: {best_f1:.4f} - Model Saved")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time, best_acc, best_f1, best_auc

def evaluate_model(model, loader):
    """Evaluate model with optimal threshold search"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # Calculate ROC-AUC
    auc = roc_auc_score(all_labels, all_preds)
    
    # Find optimal threshold for F1
    best_f1 = 0
    best_thresh = 0.5
    for threshold in np.arange(0.3, 0.7, 0.05):
        preds = all_preds > threshold
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold
    
    # Calculate accuracy with best threshold
    final_preds = all_preds > best_thresh
    acc = accuracy_score(all_labels, final_preds)
    
    return acc, best_f1, auc

def train_and_evaluate_all_models(region_name=None, region_filter=None):
    """Train ANN, LSTM, GRU and return results table"""
    print(f"\n{'='*70}")
    if region_name:
        print(f"Training Models for {region_name} Region")
    else:
        print("Training Global Models (All Locations)")
    print(f"{'='*70}")
    
    # Load data
    train_dl, test_dl, _, features = get_data_loaders(
        DATA_PATH, 
        batch_size=BATCH_SIZE,
        region_filter=region_filter
    )
    
    input_dim = len(features)
    print(f"Input dimension: {input_dim} features")
    
    results = []
    
    # Train ANN
    print("\n" + "="*70)
    ann = RainANN(input_dim, seq_length=14)  # Fixed seq_length
    ann, ann_time, ann_acc, ann_f1, ann_auc = train_model(ann, train_dl, test_dl, 
                                 f"{'Global' if not region_name else region_name}_ANN")
    results.append(["ANN", ann_acc, ann_f1, ann_auc, ann_time])
    torch.save(ann.state_dict(), f"{'global' if not region_name else region_name.lower()}_ann.pth")
    
    # Train LSTM
    # print("\n" + "="*70)
    # lstm = RainLSTM(input_dim)
    # lstm, lstm_time, lstm_acc, lstm_f1, lstm_auc = train_model(lstm, train_dl, test_dl, 
    #                                f"{'Global' if not region_name else region_name}_LSTM")
    # results.append(["LSTM", lstm_acc, lstm_f1, lstm_auc, lstm_time])
    # torch.save(lstm.state_dict(), f"{'global' if not region_name else region_name.lower()}_lstm.pth")
    
    # # Train GRU
    # print("\n" + "="*70)
    # gru = RainGRU(input_dim)
    # gru, gru_time, gru_acc, gru_f1, gru_auc = train_model(gru, train_dl, test_dl, 
    #                              f"{'Global' if not region_name else region_name}_GRU")
    # results.append(["GRU", gru_acc, gru_f1, gru_auc, gru_time])
    # torch.save(gru.state_dict(), f"{'global' if not region_name else region_name.lower()}_gru.pth")
    
    # Print results table
    print(f"\n{'='*70}")
    print(f"Results Summary - {region_name if region_name else 'Global'}")
    print(f"{'='*70}")
    print(f"{'Model':<10} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Time (s)':>12}")
    print("-" * 70)
    for row in results:
        print(f"{row[0]:<10} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]:>10.4f} {row[4]:>12.2f}")
    print("="*70)
    
    return results

if __name__ == "__main__":
    # Option 1: Train ONLY global models (for global SHAP analysis)
    global_results = train_and_evaluate_all_models()
    
    # Option 2: Train regional models (for regional comparison)
    # Uncomment these if you want region-specific models:
    coastal_results = train_and_evaluate_all_models("Coastal", COASTAL_CITIES)
    inland_results = train_and_evaluate_all_models("Inland", INLAND_CITIES)
    
    print("\n" + "="*70)
    print("Training Complete! Models saved.")
    print("="*70)