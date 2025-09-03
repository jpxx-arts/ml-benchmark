import pandas as pd
import datetime
import os
import time
import sys
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

# Importando modelos GPU
from cuml.linear_model import LogisticRegression as cuLR
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.naive_bayes import GaussianNB as cuNB
from cuml.tree import DecisionTreeClassifier as cuDT
from cuml.ensemble import RandomForestClassifier as cuRF
from xgboost import XGBClassifier

EXECUTION_MODE = "PLACEHOLDER"

# --- 1. LEITURA DOS ARGUMENTOS E GERAÇÃO DE DADOS ---
if len(sys.argv) < 3:
    print("Erro: Forneça o cenário ('comm_bound' ou 'compute_bound') e o run_id como argumentos.")
    print("Exemplo: python main.py comm_bound 1")
    sys.exit(1)

cost_scenario = sys.argv[1]
run_id = int(sys.argv[2])
print(f"Execução ID: {run_id}")
print(f"Gerando o conjunto de dados para o cenário: {cost_scenario}")

if cost_scenario == "comm_bound":
    # Cenário de comunicação: MUITAS LINHAS E MUITAS COLUNAS.
    # O volume total de dados é o gargalo (transferência CPU->GPU).
    print("Gerando dataset LARGO e LONGO para gargalo de comunicação...")
    X, y = make_classification(
        n_samples=2_000_000,    # Muitas linhas
        n_features=200,         # Muitas colunas
        n_informative=60,
        n_redundant=40,
        random_state=42
    )
elif cost_scenario == "compute_bound":
    # Cenário de computação: MENOS LINHAS e MENOS COLUNAS.
    # Mas a complexidade do modelo é aumentada para forçar compute-bound.
    print("Gerando dataset MENOR e mais DENSO para gargalo computacional...")
    X, y = make_classification(
        n_samples=50_000,       # Menos linhas
        n_features=40,          # Menos colunas
        n_informative=30,
        n_redundant=5,
        random_state=42
    )
else:
    print(f"Erro: Cenário '{cost_scenario}' desconhecido.")
    sys.exit(1)

# --- 2. MODELOS GPU ---
models = {
    "lr": cuLR(max_iter=2000, solver="qn"),  # logistic regression GPU
    "knn": cuKNN(n_neighbors=100),
    "nb": cuNB(),
    "dt": cuDT(max_depth=30),
    "rf": cuRF(n_estimators=1500, max_depth=20, n_streams=8),
    "xgboost": XGBClassifier(
        n_estimators=1500,
        max_depth=12,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# --- 3. EXECUÇÃO DO BENCHMARK ---
output_dir = os.path.join('results', EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)

all_results_for_this_run = []

for model_id, model in models.items():
    print(f"\n--- Processando modelo: {model_id} (Cenário: {cost_scenario}) ---")

    tracker = EmissionsTracker(
        project_name=f"{EXECUTION_MODE}_{cost_scenario}_{model_id}",
    )

    try:
        tracker.start()

        start = time.time()
        model.fit(X, y)  # treino direto no GPU
        end = time.time()

        duration = end - start
        print(f"    -> Tempo de treino: {duration:.2f} segundos")

    except Exception as e:
        print(f"Falha ao treinar o modelo {model_id}. Erro: {e}")
        duration = None
    finally:
        emissions_data = tracker.stop()

    if emissions_data and duration:
        result = {
            "model_id": model_id,
            "run_id": run_id,
            "scenario": cost_scenario,
            "duration_seconds": duration,
            "energy_consumed_kWh": tracker.final_emissions_data.energy_consumed,
            "emissions_kg_CO2eq": tracker.final_emissions_data.emissions,
            "cpu_energy_kWh": tracker.final_emissions_data.cpu_energy,
            "gpu_energy_kWh": tracker.final_emissions_data.gpu_energy,
            "ram_energy_kWh": tracker.final_emissions_data.ram_energy,
            "dataset_size_MB": (X.nbytes + y.nbytes) / (1024 ** 2)
        }
        # Throughput só faz sentido no cenário comm_bound
        if cost_scenario == "comm_bound":
            result["throughput_MBps"] = result["dataset_size_MB"] / duration

        all_results_for_this_run.append(result)

# --- 4. SALVAR RESULTADOS ---
if all_results_for_this_run:
    run_df = pd.DataFrame(all_results_for_this_run)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"{EXECUTION_MODE}_{cost_scenario}_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"\nResultados salvos em: {run_filename}")

