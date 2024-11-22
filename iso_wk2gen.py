import os
import json
import matplotlib.pyplot as plt

# Updated dictionary to link experiments to 'P' values
pmap = {
    "isoAttentiveSSM4": 4,
    "isoAttentiveSSM8": 8,
    "isoAttentiveSSM16": 16,
    "isoKV4": 4,
    "isoKV8": 8,
    "isoKV16": 16,
    "isoFLOP160": 4,
    "isoFLOP104": 8,
    "isoFLOP72": 16,
}

# Label mapping for updated experiment labels
label_map = {
    "isoAttentiveSSM": "Attamba",
    "isoKV": "iso-KV-Cache",
    "isoFLOP": "iso-FLOPs",
    "isoParam": "iso-Parameter"
}

def load_final_perplexity(folder_path):
    """Loads the final perplexity data from metrics.eval.jsonl."""
    metrics_path = os.path.join(folder_path, "metrics.eval.jsonl")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, 'r') as file:
        perplexities = []
        for line in file:
            data = json.loads(line)
            perplexities.append(data["wikitext"]["word_perplexity,none"])
    # return perplexities[-1] if perplexities else None
    try:
        ix = -1
        return perplexities[ix] if perplexities else None
    except:
        return perplexities[-1] if perplexities else None

def plot_bar_chart(directory):
    """Creates a bar chart of final perplexities for experiments grouped by 'P'."""
    experiments = os.listdir(directory)
    grouped_results = {}
    iso_param_perplexity = None

    for experiment in experiments:
        folder_path = os.path.join(directory, experiment)
        if os.path.isdir(folder_path):
            final_perplexity = load_final_perplexity(folder_path)
            if final_perplexity is None:
                continue
            if experiment == "isoParam":
                iso_param_perplexity = final_perplexity
                continue
            if experiment in pmap:
                p_value = pmap[experiment]
                if p_value not in grouped_results:
                    grouped_results[p_value] = []
                grouped_results[p_value].append((experiment, final_perplexity))
            print("analyzed experiment:", experiment)

    # Sorting grouped results by 'P' values
    grouped_results = dict(sorted(grouped_results.items()))

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.title("Iso-Setting WK2 Perplexity", fontsize=24)
    plt.xlabel("Token Chunk Size", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)

    # Bar chart settings
    bar_width = 0.2
    colors = plt.cm.tab20.colors
    x_positions = []
    x_labels = []
    current_x = 0
    print(grouped_results)
    # Adding unique labels to the legend
    added_labels = set()

    # Adding isoParam as a dashed horizontal line
    if iso_param_perplexity is not None:
        plt.axhline(y=iso_param_perplexity, color='red', linestyle='--', label=label_map["isoParam"])

    for i, (p_value, results) in enumerate(grouped_results.items()):
        for j, (experiment, perplexity) in enumerate(results):
            # Update label
            base_label = next((key for key in label_map if experiment.startswith(key)), experiment)
            # Remove numbers from base label
            base_label = ''.join([i for i in base_label if not i.isdigit()])
            label = label_map[base_label]

            # Only add label to legend if it hasn't been added yet
            if label not in added_labels:
                plt.bar(
                    current_x, perplexity, bar_width, color=colors[j % len(colors)],
                    label=label
                )
                added_labels.add(label)
            else:
                plt.bar(current_x, perplexity, bar_width, color=colors[j % len(colors)])

            current_x += bar_width
        x_positions.append(current_x - bar_width)  # Add center of group
        x_labels.append(str(p_value))
        current_x += bar_width  # Add spacing between groups
    plt.ylim(25, 45)
    # Adding labels and legend
    plt.yticks(fontsize=24)
    plt.xticks(x_positions, x_labels, fontsize=24)
    plt.legend(loc='upper right', fontsize=24)
    plt.tight_layout()
    plt.savefig("collated_bar.pdf")
    print("Bar chart saved as 'collated_bar.pdf'")

# Main execution
import argparse
parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
args = parser.parse_args()
plot_bar_chart(args.directory)
