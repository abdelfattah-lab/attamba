import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cycler
# name_dict = {
#     "AttentiveSSMNoProjCyc8L16_Long": "Attamba\u00A0\u00A0(512, 8, 16, \u00A0\u00A0\u00A0\u00A032, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
#     "xmer_long": "Xmer\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00, \u00A0\u00A0\u00A00, \u00A0\u00A08)",
#     "mamba_long": "Mamba\u00A0\u00A0\u00A0\u00A0(640, 0, \u00A0\u00A00, \u00A0\u00A0\u00A0\u00A064, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
    
#     "mamba_long_v2": "Mamba\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, 1024, \u00A0\u00A0\u00A02, \u00A0\u00A08)",
#     "AttentiveSSMNoProjCyc4L32_Long": "Attamba\u00A0\u00A0(512, 4, 32, \u00A0\u00A0\u00A0\u00A032, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
#     "mamba_longrun": "Mamba\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, \u00A0\u00A0128, \u00A016, 16)",
# }


name_dict = {
    # "AttentiveSSMNoProjCyc8L16_Long_NoRes": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",

    "AttentiveSSMNoProjCyc4L32_Long": "Attamba\u00A0\u00A0(512\u00A0 4\u00A0 32\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    "xmer_long": "Xmer\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
    "mamba_long": "Mamba\u00A0\u00A0\u00A0\u00A0(640\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A064\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    
    "mamba_long_v2": "Mamba\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 1024\u00A0 \u00A0\u00A0\u00A02\u00A0 \u00A0\u00A08)",
    # "AttentiveSSMNoProjCyc8L16_Long_NoRes": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",

    "AttentiveSSMNoProjCyc8L16_Long": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    "mamba_longrun": "Mamba\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0128\u00A0 \u00A016\u00A0 16)",
}

skip_expts = [""] # These are lower params, so skip them -- actually log differs so its ok.


expt_order = ["AttentiveSSMNoProjCyc8L16_Long", "AttentiveSSMNoProjCyc4L32_Long", "mamba_long", "mamba_longrun", "mamba_long_v2", "xmer_long"]
colors = plt.cm.tab20.colors  # Use a colormap for more distinct colors
markers = ['o', '<', '*', 'h', 'X', 'P', '|', '+', 's', 'D', '^', 'v',  '>', 'p', 'x']

# Ensure the cycler uses equal-length cycles
num_combinations = min(len(colors), len(markers))
plt.gca().set_prop_cycle(cycler('color', colors[:num_combinations]))

def load_metrics_from_folder(folder_path):
    """Loads global_step and perplexity data from metrics.eval.jsonl"""
    metrics_path = os.path.join(folder_path, "metrics.eval.jsonl")
    steps, perplexities = [], []

    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return steps, perplexities

    with open(metrics_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            steps.append(data["global_step"])
            perplexities.append(data["wikitext"]["word_perplexity,none"])
    
    return steps, perplexities

def plot_results(directory, selected_experiments=None):
    """Plots the perplexity values for all or selected experiments"""
    experiments = os.listdir(directory)
    if selected_experiments:
        experiments = [exp for exp in experiments if exp in selected_experiments]

    experiments = [exp for exp in expt_order if exp in experiments]

    plt.figure(figsize=(10, 6))
    # plt.figure(figsize=(18, 6))
    plt.title("Comparing Transformers, Mamba and Attamba", fontsize=24)
    plt.xlabel("Global Step", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    marker_index = 0  # Initialize marker index

    for experiment in experiments:
        if experiment in skip_expts:
            continue
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, perplexities = load_metrics_from_folder(folder_path)
            if steps and perplexities:
                # Use name_dict to change experiment name if it exists in the dictionary
                plot_label = name_dict.get(experiment, experiment)
                
                plt.plot(steps, perplexities, label=plot_label, linestyle="-", 
                         color=colors[marker_index % num_combinations], markersize=12, marker=markers[marker_index % len(markers)])
                marker_index += 1  # Increment marker index

    plt.legend(title = "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0E\u00A0\u00A0\u00A0 C \u00A0\u00A0\u00A0L \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0Ds \u00A0\u00A0\u00A0G \u00A0\u00A0\u00A0H\u00A0", title_fontsize=16, fontsize=16, ncol=1, loc="upper right")
    plt.tight_layout()
    # plt.xlim(70000, 100000)
    plt.xlim(60000, 100000)
    plt.ylim(20, 32)
    plt.savefig("collated_results.pdf")
    print("Plot saved as 'collated_results.pdf'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
    parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
    args = parser.parse_args()

    # Specify a subset of experiments here if needed, e.g., ["AttentiveSSMNoProjUnif", "AttentiveSSMWProjUnif"]
    selected_experiments = None  # Change to a list of experiment names to filter

    plot_results(args.directory, selected_experiments)
