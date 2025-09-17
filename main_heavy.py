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
# 1. Argumentos
# -----------------------
if len(sys.argv) < 3:
    print("Erro: Forneça o run_id e o execution_mode ('cc_on' ou 'cc_off').")
    sys.exit(1)

run_id = int(sys.argv[1])
EXECUTION_MODE = sys.argv[2]

print(f"ID: {run_id}, Modo: {EXECUTION_MODE}")

# -----------------------
# 2. Geração de Dados (versão pesada)
# -----------------------
# Dataset bem maior para forçar GPU
X, y = make_classification(
    n_samples=5_000_000,   # 5 milhões de amostras
    n_features=300,        # 300 features
    n_informative=200,
    n_redundant=50,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 3. Modelos e Parâmetros (pesados)
# -----------------------
model_classes = {
    "log_reg": cuLR,
    "knn": cuKNN,
    "rf": cuRF,
    "nb": cuNB,
    "svc": cuSVC,
}

params_heavy = {
    "log_reg": {"max_iter": 5000, "solver": "qn"},
    "knn": {"n_neighbors": 10},
    "rf": {"n_estimators": 1000, "max_depth": 32},
    "nb": {},
    "svc": {"kernel": "rbf", "C": 10.0, "max_iter": 5000},
}

# -----------------------
# 4. Execução do benchmark
# -----------------------
output_dir = os.path.join("results", EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)
all_results = []

for model_id, model_cls in model_classes.items():
    print(f"--- Processando {model_id} ---")
    tracker = EmissionsTracker(project_name=f"{model_id}_heavy_{run_id}")

    accuracy = None
    try:
        tracker.start()

        params = params_heavy.get(model_id, {})
        model = model_cls(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

    except Exception as e:
        print(f"Erro ao treinar {model_id}: {e}")
    finally:
        tracker.stop()

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
# 5. Salvar resultados
# -----------------------
if all_results:
    run_df = pd.DataFrame(all_results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"heavy_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"\nResultados salvos em: {run_filename}")

