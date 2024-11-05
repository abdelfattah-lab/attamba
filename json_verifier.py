import json
import glob

files = glob.glob("/scratch/ya255/lingua/setup/data/dclm_baseline_1.0_shuffled/*.jsonl")

error_counts = {}

for file in files:
    count = 0
    with open(file, 'r') as f:
        for line in f:
            try:
                json.loads(line)
            except json.JSONDecodeError:
                count += 1
    error_counts[file] = count
    print(f"Finished processing {file} with {count} errors.")

# Display total errors per file
print("\nSummary of errors per file:")
for file, count in error_counts.items():
    print(f"{file}: {count} errors")


