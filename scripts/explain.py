import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import RainLSTM, RainGRU, RainANN
from preprocess import get_data_loaders

# Configuration
DATA_PATH = 'data/weatherAUS.csv'
LSTM_MODEL_PATH = 'Global_LSTM_best.pth'
GRU_MODEL_PATH = 'Global_GRU_best.pth'
ANN_MODEL_PATH = 'Global_ANN_best.pth'

COASTAL_CITIES = ['Sydney', 'CoffsHarbour', 'Wollongong', 'GoldCoast', 'Cairns']
INLAND_CITIES = ['AliceSprings', 'Moree', 'Woomera', 'Uluru']

def run_shap_analysis(region_name, cities, model_type='LSTM'):
    print(f"\n--- Analyzing {region_name} ({model_type}) ---")
    
    # 1. Get Data
    train_dl, test_dl, _, features = get_data_loaders(DATA_PATH, region_filter=cities)
    input_dim = len(features)
    
    # 2. Load Model
    if model_type == 'LSTM':
        model = RainLSTM(input_dim=input_dim)
        model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    elif model_type == 'GRU':
        model = RainGRU(input_dim=input_dim)
        model.load_state_dict(torch.load(GRU_MODEL_PATH))
    elif model_type == 'ANN':
        model = RainANN(input_dim=input_dim, seq_length=14)
        model.load_state_dict(torch.load(ANN_MODEL_PATH))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.eval()
    
    # 3. Prepare Data
    background_data = next(iter(train_dl))[0][:100]
    test_data = next(iter(test_dl))[0][:50]
    
    # 4. Run SHAP
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data, check_additivity=False)
    
    # 5. Process Shapes
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Remove extra dimension if (Samples, Time, Feat, 1)
    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
        
    # Sum over time (Samples, Time, Feat) -> (Samples, Feat)
    if shap_values.ndim == 3:
        shap_values_2d = np.sum(shap_values, axis=1)
        test_data_2d = np.mean(test_data.numpy(), axis=1)
    else:
        # Handle ANN flat input
        if shap_values.shape[1] == input_dim * 14:
            shap_values = shap_values.reshape(shap_values.shape[0], 14, input_dim)
            shap_values_2d = np.sum(shap_values, axis=1)
            test_data_reshaped = test_data.numpy().reshape(test_data.shape[0], 14, input_dim)
            test_data_2d = np.mean(test_data_reshaped, axis=1)
        else:
            shap_values_2d = shap_values
            test_data_2d = test_data.numpy()
            
    return shap_values_2d, test_data_2d, features

def compare_models_side_by_side(region_name, cities, models=['ANN', 'LSTM', 'GRU']):
    print(f"\n=== Generating Comparison for {region_name} ===")
    
    # Collect data for each model
    results = []
    for model_type in models:
        try:
            sv, td, feats = run_shap_analysis(region_name, cities, model_type)
            results.append({'type': model_type, 'shap': sv, 'data': td, 'feats': feats})
        except Exception as e:
            print(f"Skipping {model_type}: {e}")

    if not results:
        print(f"No successful analyses for {region_name}")
        return
        
    # Create individual plots for each model
    for res in results:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(res['shap'], res['data'], feature_names=res['feats'], show=False)
        plt.title(f'{region_name} - {res["type"]}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"shap_{region_name}_{res['type']}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: shap_{region_name}_{res['type']}.png")

def run_full_analysis():
    for region_name, cities in [("Coastal", COASTAL_CITIES), ("Inland", INLAND_CITIES)]:
        compare_models_side_by_side(region_name, cities)

if __name__ == "__main__":
    run_full_analysis()