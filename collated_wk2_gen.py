import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cycler

colors = plt.cm.tab20.colors  # Use a colormap for more distinct colors
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '|', '+', 'x']

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

    plt.figure(figsize=(10, 6))
    plt.title("Perplexity vs. Global Step", fontsize=14)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Word Perplexity", fontsize=12)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    marker_index = 0  # Initialize marker index

    for experiment in experiments:
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, perplexities = load_metrics_from_folder(folder_path)
            if steps and perplexities:
                plt.plot(steps, perplexities, label=experiment, linestyle="-", 
                         color=colors[marker_index % num_combinations], marker=markers[marker_index % len(markers)])
                marker_index += 1  # Increment marker index

    plt.legend(title="Experiments", fontsize=10)
    plt.tight_layout()
    plt.savefig("collated_results.pdf")
    print("Plot saved as 'collated_results.pdf'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
    parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
    args = parser.parse_args()

    # Specify a subset of experiments here if needed, e.g., ["AttentiveSSMNoProjUnif", "AttentiveSSMWProjUnif"]
    selected_experiments = None  # Change to a list of experiment names to filter

    plot_results(args.directory, selected_experiments)
