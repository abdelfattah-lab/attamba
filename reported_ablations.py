import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.ticker import ScalarFormatter

# Define colors and markers
colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
# markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', '|', '+', 'x']
markers = ['o', '<', '*', 'h', 'X', 'P', '|', '+', 's', 'D', '^', 'v',  '>', 'p', 'x']
glabs = 20

num_combinations = min(len(colors), len(markers))

# Set up color and marker cycler
plt.gca().set_prop_cycle(cycler('color', colors[:num_combinations]))


def plot_experiment_group_7(directory, experiment_names, file_name):
    """Plots and saves the structured results for experiment_group 7"""
    plt.figure(figsize=(10, 6))
    plt.xlabel("L (Leading Uncompressed Tokens)", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    # Use the mapping to simplify experiment classification
    layer_sizes = {"Attamba C4": [], "Attamba C8": [], "Attamba C64": [], "Attamba C128": []}
    perplexities = {"Attamba C4": [], "Attamba C8": [], "Attamba C64": [], "Attamba C128": []}

    for experiment in experiment_names:
        label = get_label(experiment, 7)
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, wk2_perplexity, fpplx = load_metrics_from_folder(folder_path)
            if steps and wk2_perplexity:
                final_perplexity = wk2_perplexity[-1]

                # Determine layer size (L) from the label
                if "C4" in label:
                    lsize = int(label.split("L")[-1])
                    layer_sizes["Attamba C4"].append(lsize)
                    perplexities["Attamba C4"].append(final_perplexity)
                elif "C8" in label:
                    lsize = int(label.split("L")[-1])
                    layer_sizes["Attamba C8"].append(lsize)
                    perplexities["Attamba C8"].append(final_perplexity)
                elif "C64" in label:
                    lsize = int(label.split("L")[-1])
                    layer_sizes["Attamba C64"].append(lsize)
                    perplexities["Attamba C64"].append(final_perplexity)
                elif "C128" in label:
                    lsize = int(label.split("L")[-1])
                    layer_sizes["Attamba C128"].append(lsize)
                    perplexities["Attamba C128"].append(final_perplexity)

    # Plot each variant
    for label in layer_sizes:
        # ensure it is sorted by 'L'
        layer_sizes[label], perplexities[label] = zip(*sorted(zip(layer_sizes[label], perplexities[label])))
        plt.plot(layer_sizes[label], perplexities[label], label=label, marker="o", markersize=12, linestyle="-")

    # Add the Transformer baseline as a horizontal dotted line
    transformer_final_perplexity = load_metrics_from_folder(os.path.join(directory, "xmer_100k_dclm"))[1][-1]
    plt.axhline(y=transformer_final_perplexity, color="black", linestyle="--", label="Transformer")

    plt.legend(fontsize=24, loc="upper right")
    plt.title("Impact of Full-Attention on Leading L Tokens", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=glabs)

    plt.tight_layout()
    # plt.yscale("log")
    plt.xscale("log")
    os.makedirs("reported_graphs", exist_ok=True)
    plt.savefig(os.path.join("reported_graphs", file_name))
    plt.close()
    print(f"Plot saved as '{file_name}'")


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


def get_label(experiment_name, experiment_group):
    """Maps experiment names to their formatted labels, specific to each group"""
    if experiment_group == 1:
        mapping = {
            "AttentiveSSMWProjUnif": "AttambaProj",
            "AttentiveSSMNoProjUnif": "Attamba\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0"
        }
    elif experiment_group == 2:
        mapping = {
            "AttentiveSSM_Small_NoProjCyc": "Attamba Ds \u00A0\u00A032 (62M)",
            "AttentiveSSMNoProjCyc": "Attamba Ds 128 (64M)",
            "AttentiveSSM_Large_NoProjCyc": "Attamba Ds 512 (76M)"
        }
    elif experiment_group == 3:
        mapping = {
            "AttentiveSSMNoProjRand": "Random\u2006",
            "AttentiveSSMNoProjUnif": "Uniform\u00A0",
            "AttentiveSSMNoProjFSSM": "FSSM\u00A0\u2003\u00A0",
            "AttentiveSSMNoProjFAttn": "FAttn\u00A0\u2003\u2002",
            "AttentiveSSMNoProjCyc": "Cyclic\u00A0\u2003",
        }
    elif experiment_group == 4:
        mapping = {
            "AttentiveSSMNoProjCyc4": "Attamba C4\u00A0\u00A0\u00A0\u00A0\u00A0L4\u2007\u2007",
            "AttentiveSSMNoProjCyc": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L8\u2007\u2007",
            "AttentiveSSMNoProjCyc64": "Attamba C64\u00A0\u00A0\u00A0L64\u2007",
            "AttentiveSSMNoProjCyc128": "Attamba C128 L128"
        }
    elif experiment_group == 5:
        mapping = {
            "xmer_100k_dclm": "Transformer\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u2007\u2007\u2007\u2002\u2002\u2005",
            "AttentiveSSMNoProjCyc": "Attamba\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0C8\u00A0\u00A0\u00A0\u00A0 L8\u2007\u2007\u2007\u2006",
            "AttentiveSSMNoProjCycPseudo": "Pseudo Attamba C8\u00A0\u00A0\u00A0\u00A0 L1024",
            "AttentiveSSMNoProjCycPseudo128": "Pseudo Attamba C128 L1024"
        }
    elif experiment_group in [6, 7]:
        mapping = {
            "xmer_100k_dclm": "Transformer",
            "AttentiveSSMNoProjCyc4_L1": "Attamba C4\u00A0\u00A0\u00A0\u00A0\u00A0L1",
            "AttentiveSSMNoProjCyc4_L128": "Attamba C4\u00A0\u00A0\u00A0\u00A0\u00A0L128",
            "AttentiveSSMNoProjCyc4_L64": "Attamba C4\u00A0\u00A0\u00A0\u00A0\u00A0L64",
            "AttentiveSSMNoProjCyc4": "Attamba C4\u00A0\u00A0\u00A0\u00A0\u00A0L4",
            "AttentiveSSMNoProjCyc8_L128": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L128",
            "AttentiveSSMNoProjCyc8_L64": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L64",
            "AttentiveSSMNoProjCyc8_L16": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L16",
            "AttentiveSSMNoProjCyc": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L8",
            "AttentiveSSMNoProjCyc8_L1": "Attamba C8\u00A0\u00A0\u00A0\u00A0\u00A0L1",
            "AttentiveSSMNoProjCyc64": "Attamba C64\u00A0\u00A0\u00A0L64",
            "AttentiveSSMNoProjCyc64_L16": "Attamba C64\u00A0\u00A0\u00A0L16",
            "AttentiveSSMNoProjCyc64_L128": "Attamba C64\u00A0\u00A0\u00A0L128",
            "AttentiveSSMNoProjCyc64_L1": "Attamba C64\u00A0\u00A0\u00A0L1",
            "AttentiveSSMNoProjCyc128": "Attamba C128\u00A0L128",
            "AttentiveSSMNoProjCyc128_L16": "Attamba C128\u00A0L16",
            "AttentiveSSMNoProjCyc128_L1": "Attamba C128\u00A0L1",
        }
    else:
        mapping = {}

    return mapping.get(experiment_name, experiment_name)

def plot_experiment(directory, experiment_names, file_name, experiment_group):
    """Plots and saves the results for a given experiment"""
    plt.figure(figsize=(10, 6))
    plt.xlabel("Global Step", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)
    plt.grid(visible=True, linestyle="--", alpha=0.5)
    
    marker_index = 0  # Initialize marker index
    expt5_ordering = ["xmer_100k_dclm", "AttentiveSSMNoProjCyc", "AttentiveSSMNoProjCycPseudo", "AttentiveSSMNoProjCycPseudo128"]
    if experiment_group == 5:
        experiment_names = expt5_ordering
    for experiment in experiment_names:
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            steps, perplexities, fpplx = load_metrics_from_folder(folder_path)
            if steps and perplexities:
                label = get_label(experiment, experiment_group)
                if fpplx is not None:
                    label += f"\u00A0\u00A0\u00A0\u00A0{fpplx:.1f}"
                plt.plot(steps, perplexities, label=label, linestyle="-", 
                         color=colors[marker_index % num_combinations], marker=markers[marker_index % len(markers)],
                         markersize=12)
                marker_index += 1  # Increment marker index

    plt.legend(fontsize=24)
    if "expt_1" in file_name:
        plt.xlim(40000, 60000)
        # Set upper ylim to 60 without setting lower limit
        plt.ylim(35, 45)
        # plt.yscale("log")
        plt.xscale("log")
        # plt.yscale("log")
        # plt.xscale("log")
        plt.tick_params(axis='x', which='major', labelsize=glabs)
        plt.tick_params(axis='x', which='both', labelsize=glabs)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.title("Impact of KV-Projection before SSM", fontsize=24)
    if "expt_2" in file_name:
        plt.xlim(40000, 60000)
        # Set upper ylim to 60 without setting lower limit
        plt.ylim(34, 44)
        plt.gca().set_xticks([40000, 50000, 60000])  # Multiples of 10,000
        plt.tick_params(axis='x', which='major', labelsize=glabs)
        plt.title("Impact of Increasing SSM State-Dimension", fontsize=24)
    if "expt_3" in file_name:
        plt.title("Impact of Chunking Boundary Strategy", fontsize=24)
        plt.yscale("log")
        plt.xscale("log")
        # set y ticks as [40, 60, 100]
        plt.gca().set_yticks([40, 60, 100])
        plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.tick_params(axis='y', which='major', labelsize=glabs)
        plt.tick_params(axis='y', which='both', labelsize=glabs)
    if "expt_4" in file_name:
        plt.xlim(10000)
        plt.title("Impact of Chunk Size", fontsize=24)
        plt.yscale("log")
        plt.gca().set_yticks([40, 60, 100])
        plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter(useOffset=False))
        plt.tick_params(axis='both', which='both', labelsize=glabs)
        plt.xscale("log")
    if "expt_5" in file_name:
        plt.title("Impact of Pseudo-Chunking", fontsize=24)
        plt.xlim(10000)
        # Set upper ylim to 60 without setting lower limit
        plt.ylim(top=50)
        plt.legend(fontsize=20, loc="lower left")
        plt.tick_params(axis='x', which='major', labelsize=glabs)
        plt.tick_params(axis='x', which='both', labelsize=glabs)
        # plt.yscale("log")
        plt.xscale("log")
    if "expt_6" in file_name:
        plt.title("Impact of leading-chunk preservation", fontsize=24)
        # plt.ylim(25, 80)
        plt.xscale("log")
        # change legend location to top left
        plt.legend(fontsize=14, loc="lower left")
        plt.yscale("log")
    plt.tick_params(axis='both', which='major', labelsize=glabs)
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
        "expt_5.pdf": (["AttentiveSSMNoProjCyc", "xmer_100k_dclm", "AttentiveSSMNoProjCycPseudo", "AttentiveSSMNoProjCycPseudo128"], 5),
        "expt_6.pdf": (["xmer_100k_dclm", "AttentiveSSMNoProjCyc", "AttentiveSSMNoProjCyc8_L16", "AttentiveSSMNoProjCyc8_L1", "AttentiveSSMNoProjCyc64", "AttentiveSSMNoProjCyc64_L16", "AttentiveSSMNoProjCyc64_L1", "AttentiveSSMNoProjCyc128", "AttentiveSSMNoProjCyc128_L16", "AttentiveSSMNoProjCyc128_L1"], 6),
        "expt_7.pdf": (["xmer_100k_dclm", "AttentiveSSMNoProjCyc", "AttentiveSSMNoProjCyc8_L16", "AttentiveSSMNoProjCyc8_L64", "AttentiveSSMNoProjCyc8_L128", "AttentiveSSMNoProjCyc8_L1", "AttentiveSSMNoProjCyc64", "AttentiveSSMNoProjCyc64_L16", "AttentiveSSMNoProjCyc64_L1", "AttentiveSSMNoProjCyc128", "AttentiveSSMNoProjCyc128_L16", "AttentiveSSMNoProjCyc128_L1", "AttentiveSSMNoProjCyc4_L128", "AttentiveSSMNoProjCyc4_L64", "AttentiveSSMNoProjCyc4", "AttentiveSSMNoProjCyc64_L128", "AttentiveSSMNoProjCyc4_L1"], 7)

    }

    for file_name, (experiment_names, experiment_group) in experiments.items():
        # plot_experiment(directory, experiment_names, file_name, experiment_group)
        if experiment_group == 7:
            plot_experiment_group_7(directory, experiment_names, file_name)
        else:
            plot_experiment(directory, experiment_names, file_name, experiment_group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
    parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
    args = parser.parse_args()

    main(args.directory)
