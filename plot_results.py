import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_bars_with_ci(summary_df, metric_prefix, ylabel, title, filename):
    """
    Plots a bar chart with 95% Confidence Interval from the summary dataframe.
    """
    n_runs = 30
    z_score = 1.96  # For 95% confidence

    mean_off, mean_on = f"{metric_prefix}_mean_OFF", f"{metric_prefix}_mean_ON"
    std_off, std_on = f"{metric_prefix}_std_OFF", f"{metric_prefix}_std_ON"

    for scenario in summary_df["scenario"].unique():
        subset = summary_df[summary_df["scenario"] == scenario].sort_values("model_id")
        models = subset["model_id"]
        
        # Check for NaN in std, which can happen if n_runs < 2 or if a metric was 0 consistently.
        # If std is NaN, the error bar cannot be calculated.
        if subset[std_off].isnull().any() or subset[std_on].isnull().any():
            print(f"Warning: Missing standard deviation data for '{scenario}' scenario and '{metric_prefix}' metric. Skipping CI bars for this plot.")
            error_off = 0 # No error bar
            error_on = 0  # No error bar
        else:
            error_off = z_score * (subset[std_off] / np.sqrt(n_runs))
            error_on = z_score * (subset[std_on] / np.sqrt(n_runs))

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(x - width/2, subset[mean_off], width, label="CC Mode OFF", yerr=error_off, capsize=5, color='skyblue')
        ax.bar(x + width/2, subset[mean_on], width, label="CC Mode ON", yerr=error_on, capsize=5, color='coral')

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title}\nScenario: {scenario}", fontsize=14, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        plt.savefig(f"{filename}_{scenario}.png", dpi=300)
        plt.close(fig)

def plot_boxplots(raw_df, metric_col, ylabel, title, filename):
    """
    Plots boxplots from the raw data dataframe.
    """
    for scenario in raw_df["scenario"].unique():
        plt.figure(figsize=(12, 7))
        subset = raw_df[raw_df["scenario"] == scenario]
        
        sns.boxplot(data=subset, x="model_id", y=metric_col, hue="cc_mode", 
                    order=sorted(subset["model_id"].unique()),
                    palette={"OFF": "skyblue", "ON": "coral"})
        
        plt.xlabel("Model", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title}\nScenario: {scenario}", fontsize=14, weight='bold')
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{filename}_{scenario}.png", dpi=300)
        plt.close()


def plot_overhead_bars(summary_df, metric_overhead_col, ylabel, title, filename):
    """
    Plots a bar chart showing the percentage overhead for each model and scenario.
    """
    for scenario in summary_df["scenario"].unique():
        subset = summary_df[summary_df["scenario"] == scenario].sort_values("model_id")
        models = subset["model_id"]
        overhead_values = subset[metric_overhead_col]

        fig, ax = plt.subplots(figsize=(12, 7))
        # Usar cores diferentes para indicar overhead positivo ou negativo, se relevante
        colors = ['red' if x > 0 else 'green' for x in overhead_values] 
        ax.bar(models, overhead_values, color=colors)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title}\nScenario: {scenario}", fontsize=14, weight='bold')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Linha de base zero
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.tight_layout()
        plt.savefig(f"{filename}_{scenario}.png", dpi=300)
        plt.close(fig)


def main():
    """
    Loads processed files and generates all visualizations.
    """
    try:
        summary_df = pd.read_csv("final_summary_report.csv")
        raw_df = pd.read_csv("raw_results_all.csv")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Run 'analysis.py' first.")
        return

    models_to_exclude = ['rf']
    print(f"Deleting: {models_to_exclude}")

    summary_df = summary_df[~summary_df['model_id'].isin(models_to_exclude)]

    if raw_df is not None:
        raw_df = raw_df[~raw_df['model_id'].isin(models_to_exclude)]

    print("Generating bar charts with confidence intervals...")
    plot_bars_with_ci(summary_df, "duration_seconds", "Average Time (s)", 
                      "Comparison of Execution Time (Mean and 95% CI)", "duration_comparison")
    plot_bars_with_ci(summary_df, "total_energy_kWh", "Average Energy (kWh)", 
                      "Comparison of Total Energy Consumption (Mean and 95% CI)", "energy_comparison")
    plot_bars_with_ci(summary_df, "emissions_kg_CO2eq", "Average Emissions (kg CO2eq)", 
                      "Comparison of Carbon Footprint (Mean and 95% CI)", "emissions_comparison")
    print("Bar charts saved.")

    print("\nGenerating boxplots...")
    plot_boxplots(raw_df, "duration_seconds", "Execution Time (s)", 
                  "Distribution of Execution Time", "duration_boxplot")
    plot_boxplots(raw_df, "total_energy_kWh", "Total Energy Consumed (kWh)", 
                  "Distribution of Total Energy Consumption", "energy_boxplot")
    plot_boxplots(raw_df, "emissions_kg_CO2eq", "Emissions (kg CO2eq)", 
                  "Distribution of Carbon Footprint", "emissions_boxplot")
    print("Boxplots saved.")
    print("\nVisualization process complete!")

    print("\nGenerating Overhead bar charts...")
    plot_overhead_bars(summary_df, "duration_seconds_overhead_%", "Time Overhead (%)", 
                       "Execution Time Overhead by CC Mode", "duration_overhead_bars")
    plot_overhead_bars(summary_df, "total_energy_kWh_overhead_%", "Energy Overhead (%)", 
                       "Total Energy Consumption Overhead by CC Mode", "energy_overhead_bars")
    plot_overhead_bars(summary_df, "emissions_kg_CO2eq_overhead_%", "Emissions Overhead (%)",
                       "Carbon Footprint Overhead by CC Mode", "emissions_overhead_bars")
    print("Overhead bar charts saved.")


if __name__ == '__main__':
    main()

