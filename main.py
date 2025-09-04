import pandas as pd
import datetime
import os
import sys
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

# cuML imports
from cuml.model_selection import train_test_split
from cuml import RandomForestClassifier as cuRF
from cuml import LogisticRegression as cuLR
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml import MultinomialNB as cuNB
from cuml.metrics import accuracy_score
from cuml.svm import SVC as cuSVC

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
    X, y = make_classification(
        n_samples=1_500_000,
        n_features=100,
        n_informative=50,
        n_redundant=20,
        random_state=42
    )
elif cost_scenario == "compute_bound":
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 2. Parâmetros e Modelos
# -----------------------
model_classes = {
    "log_reg": cuLR,
    "knn": cuKNN,
    "rf": cuRF,
    "nb": cuNB,
    "svc": cuSVC,
}

params_by_scenario = {
    "compute_bound": {
        "log_reg": {"max_iter": 1000, "solver": "qn"},
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "max_depth": 16},
        "nb": {},
        "svc": {"kernel": "rbf", "C": 1.0, "max_iter": 1000},
    },
    "comm_bound": {
        "log_reg": {"max_iter": 100, "solver": "qn"},
        "knn": {"n_neighbors": 3},
        "rf": {"n_estimators": 50, "max_depth": 8},
        "nb": {},
        "svc": {"kernel": "linear", "C": 1.0, "max_iter": 500},
    }
}

# -----------------------
# 3. Execução do benchmark
# -----------------------
output_dir = os.path.join("results", EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)
all_results = []

for model_id, model_cls in model_classes.items():
    print(f"--- Processando {model_id} ({cost_scenario}) ---")
    tracker = EmissionsTracker(project_name=f"{model_id}_{cost_scenario}_{run_id}")

    duration_real, accuracy = None, None
    try:
        tracker.start()

        params = params_by_scenario[cost_scenario].get(model_id, {})

        # Treinamento e Predição
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

    except Exception as e:
        print(f"Erro ao treinar {model_id}: {e}")
    finally:
        tracker.stop()

    # Coleta de métricas
    result = {
        "model_id": model_id,
        "run_id": run_id,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "duration_seconds": tracker.final_emissions_data.duration,
        "cpu_energy_kWh": tracker.final_emissions_data.cpu_energy,
        "gpu_energy_kWh": tracker.final_emissions_data.gpu_energy,
        "ram_energy_kWh": tracker.final_emissions_data.ram_energy,
        "total_energy_kWh": tracker.final_emissions_data.energy_consumed,
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
