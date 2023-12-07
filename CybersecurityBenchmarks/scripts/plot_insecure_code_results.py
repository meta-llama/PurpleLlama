# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Import necessary libraries
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up argument parser
parser = argparse.ArgumentParser()

# Add arguments for autocomplete and instruct JSON files, and output directory
parser.add_argument("autocomplete_json_file", type=str)
parser.add_argument("instruct_json_file", type=str)
parser.add_argument("output_dir", type=str)

# Parse the arguments
args = parser.parse_args()

# Load the JSON data from the autocomplete and instruct files
with open(args.autocomplete_json_file, "r") as f:
    autocomplete_data = json.load(f)
with open(args.instruct_json_file, "r") as f:
    instruct_data = json.load(f)

# Process the autocomplete data into a list of dictionaries
data = []
for model, languages in autocomplete_data.items():
    for language, metrics in languages.items():
        metrics["model"] = model
        metrics["language"] = language
        metrics["autocomplete_pass_rate"] = metrics["pass_rate"]
        metrics["autocomplete_bleu"] = metrics["bleu"]
        data.append(metrics)

# Convert the list of dictionaries into a DataFrame
df_autocomplete = pd.DataFrame(data)

# Process the instruct data into a list of dictionaries
data = []
for model, languages in instruct_data.items():
    for language, metrics in languages.items():
        metrics["model"] = model
        metrics["language"] = language
        metrics["instruct_pass_rate"] = metrics["pass_rate"]
        metrics["instruct_bleu"] = metrics["bleu"]
        data.append(metrics)

# Convert the list of dictionaries into a DataFrame
df_instruct = pd.DataFrame(data)

# Merge the autocomplete and instruct DataFrames
df = pd.merge(df_autocomplete, df_instruct, on=["model", "language"], how="inner")

# Group the DataFrame by model and calculate the average for each metric
df_grouped = df.groupby("model").mean().reset_index()

# Create a figure with 2x2 subplots for bar plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

# Create bar plots for each metric
for i, metric in enumerate(
    [
        "autocomplete_pass_rate",
        "instruct_pass_rate",
        "autocomplete_bleu",
        "instruct_bleu",
    ]
):
    df_sorted = df_grouped.sort_values(by=metric, ascending=False)
    sns.barplot(
        data=df_sorted, x="model", y=metric, ax=axes[i // 2, i % 2], palette="Blues_d"
    )
    axes[i // 2, i % 2].set_title(f"Average {metric} by model", fontsize=16)
    axes[i // 2, i % 2].tick_params(axis="x", rotation=90, labelsize=14)
    axes[i // 2, i % 2].xaxis.label.set_size(10)

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "bar_plots.png"))

# Create a figure with 1 row and 2 columns for scatter plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# Create scatter plots for each metric
for i, metric in enumerate(["autocomplete", "instruct"]):
    sns.scatterplot(
        data=df_grouped,
        x=metric + "_pass_rate",
        y=metric + "_bleu",
        hue="model",
        ax=axes[i],
    )

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "scatter_plots.png"))
