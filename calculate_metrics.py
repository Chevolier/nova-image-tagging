import pandas as pd
import sys
from collections import defaultdict

# Get xlsx file from command line argument
if len(sys.argv) != 2:
    print("Usage: python calculate_metrics.py <xlsx_file>")
    sys.exit(1)

xlsx_file = sys.argv[1]

# Read Excel file
df = pd.read_excel(xlsx_file)

# Get ground truth and predictions using column names
# You can modify these column names to match your Excel file
ground_truth_col = 'tag_gt'  # Change this to your actual column name
predictions_col = 'inference_result'    # Change this to your actual column name

try:
    ground_truth = df[ground_truth_col].astype(str)
    predictions = df[predictions_col].astype(str)
except KeyError as e:
    print(f"Error: Column {e} not found in Excel file.")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

ground_truth_set = set(ground_truth)

# Calculate metrics for each label
label_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

for gt, pred in zip(ground_truth, predictions):
    pred_list = [p.strip() for p in pred.split(',') if p.strip()]
    
    # True positive: ground truth appears in predictions
    if gt in pred_list:
        label_stats[gt]['tp'] += 1
    else:
        label_stats[gt]['fn'] += 1
    
    # False positives: predicted labels that don't match ground truth
    for p in pred_list:
        if p != gt and p in ground_truth_set:
            label_stats[p]['fp'] += 1

# Calculate and save results
results = []
for label in sorted(label_stats.keys()):
    stats = label_stats[label]
    
    precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
    recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
    
    results.append({
        'Label': label,
        'Precision': precision,
        'Recall': recall,
        'TP': stats['tp'],
        'FP': stats['fp'],
        'FN': stats['fn']
    })

# Save to CSV
results_df = pd.DataFrame(results)
metric_file= xlsx_file.replace('.xlsx', '_metric.csv')

results_df.to_csv(metric_file, index=False)