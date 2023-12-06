# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create the parser and add arguments
parser = argparse.ArgumentParser(
    description="Analyze the outputs of LLM benchmarks for measuring insecure code."
)
parser.add_argument(
    "json_file",
    type=str,
    help="The JSON file containing LLM benchmark outputs to process",
)
parser.add_argument(
    "output_dir", type=str, help="The directory where the output files will be saved"
)

# Parse the arguments
args = parser.parse_args()

# Check if output directory exists
if not os.path.exists(args.output_dir):
    logging.error(f"The specified output directory {args.output_dir} does not exist.")
    exit()

# Load the JSON data
logging.info(f"Loading data from {args.json_file}")
with open(args.json_file, "r") as f:
    icd_data = json.load(f)

# Flatten the nested dictionary into a list of dictionaries
logging.info("Processing data")
data = []
for model, languages in icd_data.items():
    for language, metrics in languages.items():
        metrics["model"] = model
        metrics["language"] = language
        metrics["autocomplete_vunerable_percentage"] = (
            1 - metrics["autocomplete_vunerable_percentage"]
        )
        metrics["instruct_vunerable_percentage"] = (
            1 - metrics["instruct_vunerable_percentage"]
        )
        data.append(metrics)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to have each metric as a separate row for each model and language
df_melted = df.melt(
    id_vars=["model", "language"], var_name="metric", value_name="value"
)

# Group the data by 'model' and 'metric' and calculate the mean
df_grouped = df_melted.groupby(["model", "metric"]).mean().reset_index()

# Create a DataFrame that contains the sorted 'model' values
df_sorted_models = df_grouped[
    df_grouped["metric"] == "autocomplete_vunerable_percentage"
].sort_values(by="value", ascending=False)["model"]

# Create a figure with 2x2 subplots for bar plots
logging.info("Creating bar plots")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

# Create bar plots
for i, metric in enumerate(
    [
        "autocomplete_vunerable_percentage",
        "instruct_vunerable_percentage",
        "autocomplete_bleu",
        "instruct_bleu",
    ]
):
    df_sorted = (
        df_grouped[df_grouped["metric"] == metric]
        .set_index("model")
        .loc[df_sorted_models]
        .reset_index()
    )
    sns.barplot(
        data=df_sorted, x="model", y="value", ax=axes[i // 2, i % 2], palette="Blues_d"
    )
    axes[i // 2, i % 2].set_title(
        f'Average {metric.replace("_vunerable_percentage", " insecure code test cases passed").replace("_bleu", " code quality BLEU score")} by model',
        fontsize=16,
    )
    axes[i // 2, i % 2].tick_params(
        axis="x", rotation=90, labelsize=14
    )  # Set labelsize to 14
    axes[i // 2, i % 2].xaxis.label.set_size(10)

# Adjust the layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(args.output_dir, "bar_plots.png"))
logging.info(f"Bar plots saved to {os.path.join(args.output_dir, 'bar_plots.png')}")

# Create a figure with 1 row and 2 columns for regression plots
logging.info("Creating regression plots")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# Create regression plots
for i, metric in enumerate(["autocomplete", "instruct"]):
    df_metric_bleu = df_grouped[(df_grouped["metric"] == f"{metric}_bleu")]
    df_metric_vunerable = df_grouped[
        (df_grouped["metric"] == f"{metric}_vunerable_percentage")
    ]
    df_metric_merged = pd.merge(
        df_metric_bleu,
        df_metric_vunerable,
        on=["model"],
        suffixes=("_bleu", "_vunerable"),
    )

    # Create a scatter plot with different colors for each model
    sns.scatterplot(
        data=df_metric_merged,
        x="value_bleu",
        y="value_vunerable",
        hue="model",
        ax=axes[i],
        palette="Dark2",
        s=100,
    )

    # Fit a linear regression model and create a line plot
    X = df_metric_merged["value_bleu"].values.reshape(-1, 1)
    y = df_metric_merged["value_vunerable"].values
    reg = LinearRegression().fit(X, y)
    x_line = np.linspace(
        df_metric_merged["value_bleu"].min(), df_metric_merged["value_bleu"].max(), 100
    )
    y_line = reg.predict(x_line.reshape(-1, 1))
    sns.lineplot(x=x_line, y=y_line, color="black", ax=axes[i])

    # Set the title and labels
    axes[i].set_title(
        f"Relationship between code quality and insecure code test cases passed for {metric}",
        fontsize=16,
    )
    axes[i].set_xlabel("Code quality BLEU score", fontsize=14)
    axes[i].set_ylabel("Insecure code test cases passed", fontsize=14)

# Adjust the layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(args.output_dir, "regression_plots.png"))
logging.info(
    f"Regression plots saved to {os.path.join(args.output_dir, 'regression_plots.png')}"
)

logging.info("Script completed successfully")
