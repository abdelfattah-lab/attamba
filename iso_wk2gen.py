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
    "isoKV": "\u2002\u2002(+ Smaller KV Cache (KVC))",
    "isoFLOP": "iso-FLOPs",
    "isoParam": "Transformer"
}

# add space
space_id = '\u2002'
new_study = {
    4: [("\u2002\u2002(+ Sliding Window (SWA))", 37.128316028657046),
        ("\u2002\u2002(+ Smaller KVC + SWA)", 40.50273552251267),],     # 260, isoKV4],          # 260, isoParam
    8: [("\u2002\u2002(+ Sliding Window (SWA))", 43.91095101819114),      # 136, isoKV8
        ("\u2002\u2002(+ Smaller KVC + SWA)", 50.94767982061267)],         # 136, isoParam
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
            print("Analyzed experiment:", experiment)

    # Add `new_study` data to `grouped_results`
    for p_value, studies in new_study.items():
        if p_value not in grouped_results:
            grouped_results[p_value] = []
        for label, perplexity in studies:
            grouped_results[p_value].append((label, perplexity))
    # Sorting grouped results by 'P' values
    grouped_results = dict(sorted(grouped_results.items()))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Impact Of Optimizing Transformers vs. Attamba", fontsize=24)
    plt.xlabel("Token Chunk Size (P)", fontsize=24)
    plt.ylabel("WK2 Perplexity", fontsize=24)

    # Bar chart settings
    bar_width = 0.2
    # colors = plt.cm.tab20.colors
    # set colors as 4 shades of blue
        
    colors = [
        plt.cm.tab20.colors[2],  # Green from tab20
        plt.cm.Blues(0.5),       # Light blue
        plt.cm.Blues(0.7),       # Medium blue
        plt.cm.Blues(0.9)        # Darker blue
    ]
    line_color = plt.cm.Blues(0.3)

    # get a new color, as green
    x_positions = []
    x_labels = []
    current_x = 0
    print(grouped_results)
    # Adding unique labels to the legend
    added_labels = set()

    for i, (p_value, results) in enumerate(grouped_results.items()):
        ctrk = 0
        if p_value == 16:
            continue
        cluster_start = current_x  # Start of the cluster
        for j, (experiment, perplexity) in enumerate(results):
            # Update label
            base_label = next((key for key in label_map if experiment.startswith(key)), experiment)
            if experiment in [v[0] for v in new_study.values()]:
                label = experiment  # Use the label from `new_study`
            else:
                label = label_map.get(base_label, base_label)
            
            if "FLOP" in label:
                continue

            # Plot bar and add label to legend if needed
            if label not in added_labels:
                plt.bar(current_x, perplexity, bar_width, color=colors[ctrk % len(colors)], label=label)
                added_labels.add(label)
            else:
                plt.bar(current_x, perplexity, bar_width, color=colors[ctrk % len(colors)])
            ctrk += 1
            
            current_x += bar_width
        cluster_mid = (cluster_start + current_x - bar_width) / 2  # Mid-point of cluster
        x_positions.append(cluster_mid)
        x_labels.append(str(p_value))
        current_x += bar_width  # Add spacing between clusters

    plt.ylim(28, 55)  # Adjusted to accommodate new perplexity range

    # Adding isoParam as a dashed horizontal line
    if iso_param_perplexity is not None:
        plt.axhline(y=iso_param_perplexity, color=line_color, linestyle='-', label=label_map["isoParam"])


    # Adding labels and legend
    plt.yticks(fontsize=24)
    plt.xticks(x_positions, x_labels, fontsize=24)

    # Custom legend ordering
    handles, labels = plt.gca().get_legend_handles_labels()
    iso_param_index = labels.index(label_map["isoParam"])  # Index of the isoParam label
    # swap first and second elements
    labels[0], labels[1] = labels[1], labels[0]
    handles[0], handles[1] = handles[1], handles[0]
    plt.legend(handles, labels, loc='upper left', fontsize=20, ncol=1)
    # plt.legend(loc='upper left', fontsize=20, ncol=1)
    plt.tight_layout()
    plt.savefig("collated_bar.pdf")
    print("Bar chart saved as 'collated_bar.pdf'")



# Main execution
import argparse
parser = argparse.ArgumentParser(description="Plot Perplexity vs Global Step for Experiments")
parser.add_argument("directory", nargs="?", default="reported_runs", help="Directory containing experiment folders")
args = parser.parse_args()
plot_bar_chart(args.directory)
