import os
import pandas as pd
from collections import defaultdict
from data_utils.csv_reader import csv_reader_single

# Example usage
weights = {
    "resnet18": 0.8,
    "resnet34": 0.85,
    "resnet50": 0.9,
    "resnet101": 0.92,
    "resnet152": 0.95,
    "resnext50_32x4d": 0.87,
    "resnext101_32x8d": 0.89,
    "resnext101_64x4d": 0.91,
    "wide_resnet50_2": 0.86,
    "wide_resnet101_2": 0.88,
    "vit_b_16": 0.84,
    "vit_b_32": 0.82,
    "vit_l_16": 0.739,
    "vit_l_32": 0.81,
    "vit_h_14": 0.79
}

def ensemble_predictions(result_dir):
    # Initialize vote dictionary
    vote = defaultdict(lambda: defaultdict(float))

    true_labels_dict = {}
    first_file = True
    
    # Read all files from result_dir
    for file in os.listdir(result_dir):
        if file.endswith("fold1.csv"):
            # Get model name from file name
            NET_NAME = file.split('.')[0]
            
            # Get the weight for the current model
            weight = weights.get(NET_NAME, 0)
            
            # Read the CSV file predictions
            file_path = os.path.join(result_dir, file)
            file_csv = pd.read_csv(file_path)
            
            if first_file:
                # Read TRUE labels only once
                true_labels_dict = csv_reader_single(file_path, key_col="path", value_col="TRUE")
                first_file = False
            
            predictions = csv_reader_single(file_path, key_col="path", value_col="pred")
            
            # Update the vote dictionary
            for path, pred in predictions.items():
                vote[path][pred] += weight
    
    # Get final prediction based on votes
    final_predictions = []
    true_labels = []
    for path in sorted(vote.keys()):  # Ensure consistent order
        pred_dict = vote[path]
        final_predictions.append(max(pred_dict, key=pred_dict.get))
        true_labels.append(true_labels_dict[path])
    
    result = {
        'true': true_labels,
        'pred': final_predictions
    }
    
    return result



result_dir = './analysis/result/v2.0'
final_predictions = ensemble_predictions(result_dir)

# Example of printing the final predictions
for path, pred in final_predictions.items():
    print(f"{path}: {pred}")
