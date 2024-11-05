import json
import glob
import re
import os
from tqdm import tqdm

input_dir = "/scratch/ya255/lingua/setup/data/dclm_baseline_1.0_shuffled"
output_dir = "/scratch/ya255/lingua/setup/data/cleaned"
log_file = "/scratch/ya255/lingua/setup/data/error_log.txt"
os.makedirs(output_dir, exist_ok=True)

# Helper function to fix JSON formatting
def repair_json(line):
    line = line.strip()

    # Ensure line starts and ends with braces
    if not line.startswith("{"):
        line = "{" + line
    if not line.endswith("}"):
        line += "}"

    # Fix trailing commas before closing braces and in arrays
    line = re.sub(r",(\s*[}\]])", r"\1", line)
    
    # Replace invalid JSON characters and escape backslashes
    line = re.sub(r"\\text|\{|\}", "", line)
    line = line.replace("\\", "\\\\")

    # Truncate unterminated strings at typical end points (common issue with text fields)
    line = re.sub(r'"([^"]*)$', r'"\1"', line)
    return line

files = glob.glob(f"{input_dir}/*.jsonl")

with open(log_file, 'w') as log:
    for file in tqdm(files, desc="Processing files"):
        output_file = f"{output_dir}/{os.path.basename(file)}"
        with open(file, 'r') as f, open(output_file, 'w') as out_f:
            num_lines = sum(1 for _ in open(file, 'r'))
            f.seek(0)

            total_cleaned = 0
            total_repaired = 0
            total_failed = 0

            # Display statistics in the tqdm progress bar
            with tqdm(f, total=num_lines, desc=f"Cleaning {os.path.basename(file)}", leave=False) as pbar:
                for line in pbar:
                    # Split line by '}{' to separate JSON objects
                    objects = line.strip().split('}{')
                    for i, obj in enumerate(objects):
                        if i > 0:
                            obj = '{' + obj
                        if i < len(objects) - 1:
                            obj += '}'
                        
                        # Attempt to repair common issues
                        fixed_obj = repair_json(obj)

                        # Try parsing the repaired JSON
                        try:
                            json_data = json.loads(fixed_obj)
                            out_f.write(json.dumps(json_data) + '\n')
                            total_cleaned += 1
                        except json.JSONDecodeError as e:
                            # Log failed attempts for manual review
                            total_failed += 1
                            log.write(f"Failed to parse in {file} on line: {line[:50]}...\n")
                            log.write(f"Error: {e}\n")
                            log.write(f"Tried fix: {fixed_obj}\n\n")
                            continue

                        # Track lines that required repair
                        if fixed_obj != obj:
                            total_repaired += 1

                    # Update tqdm description with statistics
                    pbar.set_postfix({
                        'Cleaned': total_cleaned, 
                        'Repaired': total_repaired, 
                        'Failed': total_failed
                    })

        print(f"Finished cleaning {file}, saved to {output_file}")

print("\nAll files processed and saved to the 'cleaned' directory. Check error_log.txt for unresolved issues.")
