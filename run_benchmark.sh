#!/bin/bash

# --- CONFIGURAÇÕES GERAIS ---
TOTAL_RUNS=1
PYTHON_SCRIPT="main.py"  # script Python a ser executado

# --- VALIDAÇÃO DO INPUT ---
if [ "$1" != "on" ] && [ "$1" != "off" ]; then
  echo "Erro: Forneça um argumento ('on' ou 'off') para definir o CC Mode."
  echo "Uso: ./run_benchmark.sh on"
  echo "  ou: ./run_benchmark.sh off"
  exit 1
fi

# Define o modo CC com base no argumento
if [ "$1" == "on" ]; then
  EXEC_MODE="cc_on"
else
  EXEC_MODE="cc_off"
fi

echo "========================================================"
echo "    INICIANDO EXPERIMENTO COMPLETO - MODO: $EXEC_MODE"
echo "========================================================"

# Cenários alinhados com o script Python
COST_SCENARIOS=("comm_bound" "compute_bound")

# --- LOOP SOBRE OS CENÁRIOS ---
for scenario in "${COST_SCENARIOS[@]}"; do

  echo ""
  echo "--------------------------------------------------------"
  echo "    INICIANDO CENÁRIO: $scenario"
  echo "--------------------------------------------------------"

  # Arquivo de progresso único para cada cenário e modo
  PROGRESS_FILE="progress_${scenario}_${EXEC_MODE}.log"

  # Define de qual execução começar
  if [ -f "$PROGRESS_FILE" ]; then
    LAST_COMPLETED_RUN=$(cat "$PROGRESS_FILE")
    START_RUN=$((LAST_COMPLETED_RUN + 1))
    echo "Arquivo de progresso ($PROGRESS_FILE) encontrado. Retomando da execução #$START_RUN."
  else
    START_RUN=1
    echo "Nenhum arquivo de progresso encontrado. Iniciando da execução #1."
  fi

  if [ "$START_RUN" -gt "$TOTAL_RUNS" ]; then
    echo "Cenário '$scenario' já concluído com $TOTAL_RUNS execuções. Pulando."
    continue
  fi

  # Loop principal das execuções
  for i in $(seq $START_RUN $TOTAL_RUNS); do
    echo ""
    echo "--- [$(date)] Iniciando Execução #$i de $TOTAL_RUNS (Cenário: $scenario, Modo: $EXEC_MODE) ---"

    # Executa o script Python passando os 3 argumentos: cenário, run_id, execution_mode
    python "$PYTHON_SCRIPT" "$scenario" "$i" "$EXEC_MODE"
    
    # Checa o código de saída
    if [ $? -eq 0 ]; then
      echo "$i" > "$PROGRESS_FILE"
      echo "--- [$(date)] Execução #$i concluída com sucesso. Progresso salvo. ---"
    else
      echo "--- [$(date)] ERRO: A execução #$i falhou. O progresso NÃO foi salvo. ---"
      echo "Saindo do script. Na próxima vez, o script tentará executar o passo #$i novamente."
      exit 1
    fi
  done
done

echo ""
echo "========================================================"
echo "    TODOS OS CENÁRIOS CONCLUÍDOS PARA O MODO $EXEC_MODE"
echo "========================================================"

