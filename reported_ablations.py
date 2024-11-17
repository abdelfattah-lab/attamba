import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cycler

# Define colors and markers
colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '|', '+', 'x']
num_combinations = min(len(colors), len(markers))

# Set up color and marker cycler
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

def get_label(experiment_name, experiment_group):
    """Maps experiment names to their formatted labels, specific to each group"""
    if experiment_group == 1:
        mapping = {
            "AttentiveSSMWProjUnif": "Attamba + Proj",
            "AttentiveSSMNoProjUnif": "Attamba - Proj"
        }
    elif experiment_group == 2:
        mapping = {
            "AttentiveSSM_Small_NoProjCyc": "Attamba SSM 2M",
            "AttentiveSSMNoProjCyc": "Attamba SSM 4M",
            "AttentiveSSM_Large_NoProjCyc": "Attamba SSM 16M"
        }
    elif experiment_group == 3:
        mapping = {
            "AttentiveSSMNoProjRand": "Random",
            "AttentiveSSMNoProjCyc": "Cyclic",
            "AttentiveSSMNoProjUnif": "Uniform",
            "AttentiveSSMNoProjFSSM": "FSSM",
            "AttentiveSSMNoProjFAttn": "FAttn"
        }
    elif experiment_group == 4:
        mapping = {
            "AttentiveSSMNoProjCyc4": "Attamba Chunk 4",
            "AttentiveSSMNoProjCyc": "Attamba Chunk 8",
            "AttentiveSSMNoProjCyc64": "Attamba Chunk 64",
            "AttentiveSSMNoProjCyc128": "Attamba Chunk 128"
        }
    elif experiment_group == 5:
        mapping = {
            "AttentiveSSMNoProjCyc": "Attamba C: 8",
            "xmer_100k_dclm": "Transformer",
            "AttentiveSSMNoProjCycPseudo": "Pseudo-Attamba C: 8",
            "AttentiveSSMNoProjCycPseudo128": "Pseudo-Attamba C: 128"
        }
    else:
        mapping = {}

    return mapping.get(experiment_name, experiment_name)

def plot_experiment(directory, experiment_names, file_name, experiment_group):
    """Plots and saves the results for a given experiment"""
    plt.figure(figsize=(10, 6))
    plt.xlabel("Global Step", fontsize=18)
    plt.ylabel("WK2 Perplexity", fontsize=18)
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    
    marker_index = 0  # Initialize marker index

    for experiment in experiment_names:
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, perplexities = load_metrics_from_folder(folder_path)
            if steps and perplexities:
                label = get_label(experiment, experiment_group)
                plt.plot(steps, perplexities, label=label, linestyle="-", 
                         color=colors[marker_index % num_combinations], marker=markers[marker_index % len(markers)],
                         markersize=12)
                marker_index += 1  # Increment marker index

    plt.legend(fontsize=24)
    if "expt_1" in file_name:
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Impact of KV-Projection before SSM", fontsize=24)
    if "expt_2" in file_name:
        plt.title("Impact of SSM Size (Model Size: 60M)", fontsize=24)
        plt.yscale("log")
        plt.xscale("log")
    if "expt_3" in file_name:
        plt.title("Impact of Chunking Boundary Strategy", fontsize=24)
        plt.yscale("log")
        plt.xscale("log")
    if "expt_4" in file_name:
        plt.xlim(10000)
        plt.title("Impact of Chunk Size", fontsize=24)
        plt.yscale("log")
        plt.xscale("log")
    if "expt_5" in file_name:
        plt.title("Impact of Pseudo-Chunking", fontsize=24)
        plt.xlim(10000)
        # plt.ylim(25, 45)
        plt.yscale("log")
        plt.xscale("log")
    plt.tight_layout()
    os.makedirs("reported_graphs", exist_ok=True)
    plt.savefig(os.path.join("reported_graphs", file_name))
    plt.close()
    print(f"Plot saved as '{file_name}'")

def main(directory):
    # Define experiment groups, names, and their file names
    experiments = {
        "expt_1.pdf": (["AttentiveSSMWProjUnif", "AttentiveSSMNoProjUnif"], 1),
        "expt_2.pdf": (["AttentiveSSM_Small_NoProjCyc", "AttentiveSSMNoProjCyc", "AttentiveSSM_Large_NoProjCyc"], 2),
        "expt_3.pdf": (["AttentiveSSMNoProjRand", "AttentiveSSMNoProjCyc", "AttentiveSSMNoProjUnif", "AttentiveSSMNoProjFSSM", "AttentiveSSMNoProjFAttn"], 3),
        "expt_4.pdf": (["AttentiveSSMNoProjCyc4", "AttentiveSSMNoProjCyc", "AttentiveSSMNoProjCyc64", "AttentiveSSMNoProjCyc128"], 4),
        "expt_5.pdf": (["AttentiveSSMNoProjCyc", "xmer_100k_dclm", "AttentiveSSMNoProjCycPseudo", "AttentiveSSMNoProjCycPseudo128"], 5)
    }

    # Plot each experiment
    for file_name, (experiment_names, experiment_group) in experiments.items():
        plot_experiment(directory, experiment_names, file_name, experiment_group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
    parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
    args = parser.parse_args()

    main(args.directory)
