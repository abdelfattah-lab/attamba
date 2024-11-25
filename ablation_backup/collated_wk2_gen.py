import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.ticker import ScalarFormatter
# name_dict = {
#     "AttentiveSSMNoProjCyc8L16_Long": "Attamba\u00A0\u00A0(512, 8, 16, \u00A0\u00A0\u00A0\u00A032, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
#     "xmer_long": "Xmer\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00, \u00A0\u00A0\u00A00, \u00A0\u00A08)",
#     "mamba_long": "Mamba\u00A0\u00A0\u00A0\u00A0(640, 0, \u00A0\u00A00, \u00A0\u00A0\u00A0\u00A064, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
    
#     "mamba_long_v2": "Mamba\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, 1024, \u00A0\u00A0\u00A02, \u00A0\u00A08)",
#     "AttentiveSSMNoProjCyc4L32_Long": "Attamba\u00A0\u00A0(512, 4, 32, \u00A0\u00A0\u00A0\u00A032, \u00A0\u00A0\u00A01, \u00A0\u00A08)",
#     "mamba_longrun": "Mamba\u00A0\u00A0\u00A0\u00A0(512, 0, \u00A0\u00A00, \u00A0\u00A0128, \u00A016, 16)",
# }

result_new = {"isoxmer_attnlong": 
{
    5000: 57.826293141496095,
    10000: 44.70515256557447,
    20000: 40.13716909576324,
    60000: 34.03071519671879,
    80000: 30.422002624813825,
    90000: 29.114242625880298,
    95000: 28.698214184307858,
    100000:28.661845903556138
}}

# Smaller space than \u00A0
# smspace = "\u2003"

name_dict = {
    # "AttentiveSSMNoProjCyc8L16_Long_NoRes": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",

    "AttentiveSSMNoProjCyc4L32_Long": "Attamba\u00A0\u00A0(512\u00A0 4\u00A0 32\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    "xmer_long": "Xmer\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
    "isoxmer_long": "\u00A0\u00A0(+KVC)\u00A0\u00A0(128\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
    "isoxmer_attnlong": "\u00A0\u00A0(+SWA)\u00A0\u200A(128\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
    "mamba_long": "Mamba\u00A0\u00A0\u00A0\u00A0(640\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A064\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    
    "mamba_long_v2": "Mamba\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 1024\u00A0 \u00A0\u00A0\u00A02\u00A0 \u00A0\u00A08)",
    # "AttentiveSSMNoProjCyc8L16_Long_NoRes": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",

    "AttentiveSSMNoProjCyc8L16_Long": "Attamba\u00A0\u00A0(512\u00A0 8\u00A0 16\u00A0 \u00A0\u00A0\u00A0\u00A032\u00A0 \u00A0\u00A0\u00A01\u00A0 \u00A0\u00A08)",
    "mamba_longrun": "Mamba\u00A0\u00A0\u00A0\u00A0(512\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0128\u00A0 \u00A016\u00A0 16)",
    "hawk": "Hawk\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0(480\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
    "minGRU": "minGRU\u00A0\u00A0\u00A0(576\u00A0 0\u00A0 \u00A0\u00A00\u00A0 \u00A0\u00A0\u00A0\u00A0\u00A0\u00A00\u00A0 \u00A0\u00A0\u00A00\u00A0 \u00A0\u00A08)",
}

# skip_expts = ["AttentiveSSMNoProjCyc8L16_Long", "mamba_long_v2", "mamba_longrun"]
skip_expts = ["AttentiveSSMNoProjCyc8L16_Long", "mamba_long_v2", "mamba_longrun"]

# expt_order = ["AttentiveSSMNoProjCyc8L16_Long", "AttentiveSSMNoProjCyc4L32_Long", "mamba_long", "mamba_longrun", "mamba_long_v2", "minGRU", "hawk", "xmer_long"]
expt_order = ["minGRU", "hawk", "mamba_longrun", "AttentiveSSMNoProjCyc8L16_Long", "mamba_long", "mamba_long_v2", "AttentiveSSMNoProjCyc4L32_Long", "xmer_long", "isoxmer_long"]
colors = plt.cm.Dark2.colors  # Use a colormap for more distinct colors
# colors = colors[::-1]
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
    final_perplexity = perplexities[-1] if perplexities else None

    return steps, perplexities, final_perplexity

def plot_results(directory, selected_experiments=None):
    """Plots the perplexity values for all or selected experiments"""
    experiments = os.listdir(directory)
    # if selected_experiments:
    #     experiments = [exp for exp in experiments if exp in selected_experiments]
    print(experiments)
    # experiments = [exp for exp in expt_order if exp in experiments]
    experiments = [exp for exp in experiments if exp in expt_order]
    experiments = sorted(experiments, key=lambda exp: expt_order.index(exp))

    # remove from experiments if not in expt_order
    print(experiments)

    plt.figure(figsize=(16, 6))
    # plt.figure(figsize=(18, 6))
    plt.title("Transformer, Mamba, minGRU, Hawk, Attamba", fontsize=24)
    plt.xlabel("Global Step", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    marker_index = 0  # Initialize marker index

    for experiment in experiments:
        if experiment in skip_expts:
            print("SKIP ", experiment)
            continue
        print(experiment)
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, perplexities, fpplx = load_metrics_from_folder(folder_path)
            if steps and perplexities:
                # Use name_dict to change experiment name if it exists in the dictionary
                plot_label = name_dict.get(experiment, experiment)
                if fpplx is not None:
                    plot_label += f"\u00A0\u00A0\u00A0\u00A0{fpplx:.2f}"
                
                plt.plot(steps, perplexities, label=plot_label, linestyle="-", 
                         color=colors[marker_index % num_combinations], markersize=16, marker=markers[marker_index % len(markers)])
                marker_index += 1  # Increment marker index

    # Plot 'result_new' data
    for run_name, data in result_new.items():
        steps = list(data.keys())
        perplexities = list(data.values())
        plot_label = name_dict.get(run_name, run_name)
        if perplexities:
            fpplx = perplexities[-1]
            plot_label += f"\u00A0\u00A0\u00A0\u00A0{fpplx:.2f}"
        
        plt.plot(steps, perplexities, label=plot_label, linestyle="--",
                 color=colors[marker_index % num_combinations], markersize=16, marker=markers[marker_index % len(markers)])
        marker_index += 1  # Increment marker index

    legend = plt.legend(title = "\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0E\u00A0\u00A0\u00A0 C \u00A0\u00A0\u00A0L \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0Ds \u00A0\u00A0\u00A0G \u00A0\u00A0\u00A0H\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u2003\u2002\u2009\u200A", \
                        title_fontsize=20, fontsize=20, ncol=1, loc="lower left")

    # for text in legend.get_texts():
    #     if "Attamba" in text.get_text():
    #         text.set_weight("bold")
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.gca().set_xticks([20000, 40000, 60000, 80000, 100000])  # Multiples of 10,000
    plt.gca().set_yticks([22, 24, 26, 28])  # Multiples of 10,000
    # plt.xlim(40000, 100000)
    # plt.ylim(20.5, 34)
    plt.xlim(20000, 100000)
    plt.ylim(20.5, 30)
    plt.savefig("collated_results.pdf")
    print("Plot saved as 'collated_results.pdf'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
    parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
    args = parser.parse_args()

    # Specify a subset of experiments here if needed, e.g., ["AttentiveSSMNoProjUnif", "AttentiveSSMWProjUnif"]
    selected_experiments = None  # Change to a list of experiment names to filter

    plot_results(args.directory, selected_experiments)
