import pandas as pd
import datetime
import os
import sys
import codecarbon
from sklearn.datasets import make_classification

# cuML imports
from cuml.model_selection import train_test_split
from cuml import RandomForestClassifier as cuRF
from cuml import LogisticRegression as cuLR
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml import MultinomialNB as cuNB
from cuml.metrics import accuracy_score
import xgboost as xgb
import cupy as cp # Usado para conversão de tipos

# -----------------------
# 1. Argumentos e datasets
# -----------------------
if len(sys.argv) < 4:
    print("Erro: Forneça o cenário ('comm_bound' ou 'compute_bound'), run_id e execution_mode ('cc_on' ou 'cc_off').")
    sys.exit(1)

cost_scenario = sys.argv[1]
run_id = int(sys.argv[2])
EXECUTION_MODE = sys.argv[3]

print(f"ID: {run_id}, Cenário: {cost_scenario}, Modo: {EXECUTION_MODE}")

# -----------------------
# Geração de Dados
# -----------------------
if cost_scenario == "comm_bound":
    X, y = make_classification(n_samples=1_500_000, n_features=100, n_informative=50, n_redundant=20, random_state=42)
elif cost_scenario == "compute_bound":
    X, y = make_classification(n_samples=500_000, n_features=80, n_informative=60, n_redundant=10, random_state=42)
else:
    print(f"Cenário desconhecido: {cost_scenario}")
    sys.exit(1)

# ### CORREÇÃO ###: Adiciona divisão treino/teste para validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 2. Parâmetros e Modelos
# -----------------------
compute_bound_params = {
    "cuLR": {"max_iter": 2000},
    "cuKNN": {"n_neighbors": 100},
    "cuRF": {"n_estimators": 3000, "max_depth": 25},
    "xgb": {"n_estimators": 2000, "max_depth": 12, "tree_method": "gpu_hist"},
}
models_to_test = {"cuLR": cuLR, "cuKNN": cuKNN, "cuRF": cuRF, "cuNB": cuNB, "xgb": "xgb_custom"}

# -----------------------
# 3. Execução do benchmark
# -----------------------
output_dir = os.path.join("results", EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)
all_results = []

for model_id, model_cls in models_to_test.items():
    print(f"--- Processando {model_id} ({cost_scenario}) ---")
    tracker = codecarbon.EmissionsTracker(project_name=f"{model_id}_{cost_scenario}_{run_id}")
    
    try:
        tracker.start()
        
        # ### CORREÇÃO ###: Lógica para aplicar parâmetros apenas no cenário correto
        params = {}
        if cost_scenario == 'compute_bound':
            params = compute_bound_params.get(model_id, {})
        
        # Treinamento e Predição
        accuracy = None
        if model_id == "xgb":
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            model = xgb.train(params, dtrain)
            preds = model.predict(dtest)
            # XGBoost predict retorna probabilidades, precisamos converter para 0/1
            accuracy = accuracy_score(y_test, cp.round(preds))
        else:
            model = model_cls(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)

    except Exception as e:
        print(f"Erro ao treinar {model_id}: {e}")
        duration_real, accuracy = None, None
    finally:
        emissions_data = tracker.stop()

    # Coleta de métricas
    result = {
        "model_id": model_id, "run_id": run_id, "accuracy": float(accuracy) if accuracy is not None else None,
        "real_duration_seconds": duration_real, "cc_duration_seconds": emissions_data,
        "cpu_energy_kWh": tracker.final_emissions_data.cpu_energy, "gpu_energy_kWh": tracker.final_emissions_data.gpu_energy,
        "ram_energy_kWh": tracker.final_emissions_data.ram_energy, "total_energy_kWh": tracker.final_emissions_data.energy_consumed,
        "emissions_kg_CO2eq": tracker.final_emissions_data.emissions
    }
    all_results.append(result)

# -----------------------
# 4. Salvar resultados
# -----------------------
if all_results:
    run_df = pd.DataFrame(all_results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"{cost_scenario}_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"\nResultados salvos em: {run_filename}")
