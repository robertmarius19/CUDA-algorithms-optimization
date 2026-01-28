import pandas as pd
import matplotlib.pyplot as plt
import io

FILE_NAME = "linear_results.csv"

def load_and_clean_data(filename):
    """
    Reads the CSV file while filtering out the separator lines ('----')
    """
    try:
        cleaned_lines = []
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip().startswith('-'):
                    cleaned_lines.append(line)
        
        df = pd.read_csv(io.StringIO(''.join(cleaned_lines)))
        
        df['N'] = pd.to_numeric(df['N'])
        df['time_ms'] = pd.to_numeric(df['time_ms'])
        
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'.")
        exit(1)

def plot_algorithm(df, algo_name, ax, label_overrides=None):
    """
    Plots a single algorithm.
    label_overrides: A dictionary to rename keys like 'omp' -> 'cuBLAS'
    """
    if label_overrides is None:
        label_overrides = {}

    subset = df[df['algo'] == algo_name]
    versions = subset['version'].unique()
    
    base_styles = {
        'seq':  {'color': 'red', 'marker': 'o', 'label': 'Sequential'},
        'omp':  {'color': 'blue', 'marker': 's', 'label': 'Parallel (CPU)'},
        'cuda': {'color': 'green', 'marker': '^', 'label': 'CUDA (Kernel)'}
    }

    for v in versions:
        data = subset[subset['version'] == v].sort_values(by='N')
        
        style = base_styles.get(v, {'color': 'black', 'marker': 'x', 'label': v})
        
        final_label = label_overrides.get(v, style['label'])

        ax.plot(data['N'], data['time_ms'], 
                marker=style['marker'], 
                color=style['color'], 
                label=final_label,
                linewidth=2)

    ax.set_title(f"{algo_name} Performance", fontsize=14, fontweight='bold')
    ax.set_xlabel("Problem Size (N)", fontsize=12)
    ax.set_ylabel("Time (ms) - Log Scale", fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

df = load_and_clean_data(FILE_NAME)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_algorithm(df, "DFT", ax1, label_overrides={
    'omp': 'OpenMP (CPU)'
})

plot_algorithm(df, "GEMM", ax2, label_overrides={
    'omp': 'cuBLAS / Library', 
    'cuda': 'Custom CUDA Kernel'
})

plt.tight_layout()
plt.savefig("benchmark_results.png")
print("Graph saved as 'benchmark_results.png'")
plt.show()