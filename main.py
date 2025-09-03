import pandas as pd
import datetime
import os
import sys
import time
import warnings
import logging
import codecarbon
from sklearn.datasets import make_classification

# cuML imports (GPU-ready)
from cuml import RandomForestClassifier as cuRF
from cuml import LogisticRegression as cuLR
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml import MultinomialNB as cuNB
import xgboost as xgb

# -----------------------
# 0. Configurações gerais
# -----------------------
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ["NUMBA_WARNINGS"] = "0"

# -----------------------
# 1. Argumentos e datasets
# -----------------------
if len(sys.argv) < 4:
    print("Erro: Forneça o cenário ('comm_bound' ou 'compute_bound'), run_id e execution_mode ('cc_on' ou 'cc_off').")
    print("Exemplo: python main.py comm_bound 1 cc_on")
    sys.exit(1)

cost_scenario = sys.argv[1]
run_id = int(sys.argv[2])
EXECUTION_MODE = sys.argv[3]  # cc_on ou cc_off

print(f"Execução ID: {run_id}")
print(f"Cenário: {cost_scenario}")
print(f"Execution Mode: {EXECUTION_MODE}")

# -----------------------
# Dataset conforme gargalo
# -----------------------
if cost_scenario == "comm_bound":
    # Dataset grande (~50-60 GB) para usar bem a GPU
    X, y = make_classification(
        n_samples=1_500_000,  # grande número de amostras
        n_features=100,       # 100 features
        n_informative=50,
        n_redundant=20,
        random_state=42
    )
elif cost_scenario == "compute_bound":
    # Dataset denso para computação intensiva
    X, y = make_classification(
        n_samples=500_000,
        n_features=80,
        n_informative=60,
        n_redundant=10,
        random_state=42
    )
else:
    print(f"Cenário desconhecido: {cost_scenario}")
    sys.exit(1)

X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_s = pd.Series(y, name="target")
df = pd.concat([X_df, y_s], axis=1)

# -----------------------
# 2. Parâmetros de modelos
# -----------------------
compute_bound_params = {
    "cuLR": {"max_iter": 2000},
    "cuKNN": {"n_neighbors": 100},
    "cuRF": {"n_estimators": 3000, "max_depth": 25},  # mais estimators
    "xgb": {"n_estimators": 2000, "max_depth": 12, "tree_method": "gpu_hist"},
}

# -----------------------
# 3. Lista de modelos
# -----------------------
models_to_test = {
    "cuLR": cuLR,
    "cuKNN": cuKNN,
    "cuRF": cuRF,
    "cuNB": cuNB,
    "xgb": xgb
}

# -----------------------
# 4. Pasta de saída
# -----------------------
output_dir = os.path.join("results", f"{EXECUTION_MODE}")
os.makedirs(output_dir, exist_ok=True)

# -----------------------
# 5. Execução do benchmark
# -----------------------
all_results_for_this_run = []

for model_id, model_cls in models_to_test.items():
    print(f"--- Processando {model_id} ({cost_scenario}) ---")
    
    tracker = codecarbon.EmissionsTracker(project_name=f"{model_id}_{cost_scenario}_{run_id}")
    
    try:
        tracker.start()
        
        # Seleção de parâmetros
        params = compute_bound_params.get(model_id, {})

        # Medir tempo real
        start_time = time.time()

        # Cria e treina o modelo
        if model_id == "xgb":
            dtrain = xgb.DMatrix(X, label=y)
            bst = xgb.train(params, dtrain)
        else:
            model = model_cls(**params)
            model.fit(X, y)

        duration_real = time.time() - start_time
        time.sleep(1)  # pequeno delay para coleta de métricas
    except Exception as e:
        print(f"Erro ao treinar {model_id}: {e}")
        duration_real = None
    finally:
        emissions_data = tracker.stop()

    # Coleta de métricas
    result = {
        "model_id": model_id,
        "run_id": run_id,
        "real_duration_seconds": duration_real,
        "duration_seconds": emissions_data,
        "cpu_energy_kWh": tracker.final_emissions_data.cpu_energy,
        "gpu_energy_kWh": tracker.final_emissions_data.gpu_energy,
        "ram_energy_kWh": tracker.final_emissions_data.ram_energy,
        "total_energy_kWh": tracker.final_emissions_data.energy_consumed,
        "emissions_kg_CO2eq": tracker.final_emissions_data.emissions
    }
    all_results_for_this_run.append(result)

# -----------------------
# 6. Salvar resultados
# -----------------------
if all_results_for_this_run:
    run_df = pd.DataFrame(all_results_for_this_run)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"{cost_scenario}_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"\nResultados salvos em: {run_filename}")

