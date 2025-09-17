import pandas as pd
from pathlib import Path

def load_all_data(base_path, modes_to_analyze):
    """
    Carrega todos os arquivos CSV, adiciona colunas 'cc_mode' e 'scenario'
    e concatena tudo em um único DataFrame.
    """
    all_dfs = []
    for mode, folder_name in modes_to_analyze.items():
        mode_path = base_path / folder_name
        all_files = list(mode_path.glob('*.csv'))
        
        if not all_files:
            print(f"Aviso: Nenhum arquivo CSV encontrado em '{mode_path}'. Pulando este modo.")
            continue
        
        for f in all_files:
            df = pd.read_csv(f)
            df['cc_mode'] = mode
            if 'comm_bound' in f.name:
                df['scenario'] = 'comm_bound'
            elif 'compute_bound' in f.name:
                df['scenario'] = 'compute_bound'
            else:
                df['scenario'] = 'unknown'
            all_dfs.append(df)

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)

def main():
    """
    Orquestra a consolidação dos dados brutos e a criação do relatório de resumo.
    """
    base_path = Path('results')
    modes_to_analyze = {'OFF': 'cc_off', 'ON': 'cc_on'}

    print("Carregando e consolidando todos os dados brutos...")
    all_data = load_all_data(base_path, modes_to_analyze)

    if all_data is None:
        print("Erro: Nenhuma pasta de resultados encontrada ou todas estão vazias.")
        return

    # ### MUDANÇA CHAVE 1 ###
    # Salva o arquivo consolidado com todos os dados brutos. Essencial para os boxplots.
    raw_output_filename = 'raw_results_all.csv'
    all_data.to_csv(raw_output_filename, index=False)
    print(f"Dados brutos consolidados salvos em: {raw_output_filename}")

    # --- Agregação para o relatório de resumo ---
    print("Gerando relatório de resumo (médias, desvios padrão e overheads)...")
    numeric_cols = all_data.select_dtypes(include='number').columns.tolist()
    if 'run_id' in numeric_cols:
        numeric_cols.remove('run_id')

    stats_df = all_data.groupby(['model_id', 'scenario', 'cc_mode'])[numeric_cols].agg(['mean', 'std'])
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    
    # --- Cálculo do Overhead ---
    comparison_df = stats_df.unstack(level='cc_mode')
    
    metrics_for_overhead = ['duration_seconds', 'total_energy_kWh', 'gpu_energy_kWh', 'cpu_energy_kWh', 'emissions_kg_CO2eq']

    for metric in metrics_for_overhead:
        col_mean_on = (f'{metric}_mean', 'ON')
        col_mean_off = (f'{metric}_mean', 'OFF')
        overhead_col_name = f'{metric}_overhead_%'

        if col_mean_on in comparison_df.columns and col_mean_off in comparison_df.columns:
            comparison_df[overhead_col_name] = (
                (comparison_df[col_mean_on] - comparison_df[col_mean_off]) / 
                 comparison_df[col_mean_off]
            ) * 100
    
    # --- Organização e Salvamento do Relatório de Resumo ---
    new_columns = []
    for col in comparison_df.columns:
        if isinstance(col, tuple):
            new_columns.append('_'.join(part for part in col if part))
        else:
            new_columns.append(col)
    comparison_df.columns = new_columns
    
    comparison_df = comparison_df.reset_index()

    summary_output_filename = 'final_summary_report.csv'
    comparison_df.to_csv(summary_output_filename, index=False)
    print(f"Relatório de resumo salvo em: {summary_output_filename}")
    print("\nAnálise concluída!")


if __name__ == '__main__':
    main()

