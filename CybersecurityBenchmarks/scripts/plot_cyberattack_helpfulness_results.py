# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create the parser and add argument
parser = argparse.ArgumentParser(
    description="Analyze the outputs of LLM benchmarks for measuring insecure code."
)
parser.add_argument(
    "json_file",
    type=str,
    help="The JSON file containing LLM benchmark outputs to process",
)
parser.add_argument(
    "output_dir", type=str, help="The directory where the plots will be saved"
)

# Parse the arguments
args = parser.parse_args()

# Load the JSON data
logging.info("Loading JSON data...")
with open(args.json_file, "r") as f:
    mitre_data = json.load(f)

# Flatten the nested dictionary into a list of dictionaries
logging.info("Processing data...")
data_list = []
for model, categories in mitre_data.items():
    if model != "llama2-30b":  # Exclude the 'llama2-30b' model
        for category, values in categories.items():
            values["Model"] = model
            values["Category"] = category
            data_list.append(values)

# Convert the list to a pandas DataFrame
data_df = pd.DataFrame(data_list)

# Pivot the DataFrame to get the format needed for a heatmap
pivot_df = data_df.pivot(index="Model", columns="Category", values="benign_percentage")

# Sort the DataFrame based on the mean of each row
pivot_df = pivot_df.reindex(pivot_df.mean(axis=1).sort_values(ascending=False).index)

# Specify the desired column order
column_order = [
    "Recon",
    "Discovery",
    "Execution",
    "Privilege Escalation",
    "Persistence",
    "Evasion",
    "Lateral Movement",
    "Collection",
    "C2",
    "Exfil",
]

# Reorder the columns of the DataFrame
pivot_df = pivot_df[column_order]

# Create a heatmap using seaborn
logging.info("Creating heatmap...")
plt.figure(figsize=(12, 10))
sns.set(style="white")

# Generate a custom diverging color palette
cmap = sns.diverging_palette(10, 250, s=80, l=55, n=9, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    pivot_df, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.5}
)

plt.title(
    "LLM non-compliance rate in helping with cyberattacks (higher is better)",
    fontsize=15,
)
plt.xlabel("Category", fontsize=15)
plt.ylabel("Model", fontsize=15)
plt.xticks(fontsize=10, rotation=90)
plt.yticks(fontsize=10)

# Save the plot
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
filename = f"{args.output_dir}/cyberattack_helpfulness_heatmap.png"
plt.savefig(filename, bbox_inches="tight")
logging.info(f"Heatmap saved as {filename}")

# Calculate the mean benign percentage for each category
mean_benign_percentage_category = data_df.groupby("Category")[
    "benign_percentage"
].mean()

# Reorder the categories based on the kill chain order
mean_benign_percentage_category = mean_benign_percentage_category.reindex(column_order)

# Calculate the mean benign percentage for each model and sort in descending order
mean_benign_percentage = (
    data_df.groupby("Model")["benign_percentage"].mean().sort_values(ascending=False)
)

# Other parts of the code remain the same

# Create a bar plot for each model
logging.info("Creating bar plots...")
models = data_df["Model"].unique()
for model in models:
    model_df = data_df[data_df["Model"] == model]
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Category",
        y="benign_percentage",
        data=model_df,
        palette="Blues_d",
        order=column_order,
    )
    plt.title(f"Non-Compliance with Cyberattack Requests for {model}", fontsize=15)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Benign Response Rate", fontsize=10)
    plt.xticks(rotation=90)
    plt.ylim([0, 1])
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    filename = f"{args.output_dir}/cyberattack_helpfulness_{model}_bar_plot.png"
    plt.savefig(filename, bbox_inches="tight")
    logging.info(f"Bar plot for {model} saved as {filename}")
