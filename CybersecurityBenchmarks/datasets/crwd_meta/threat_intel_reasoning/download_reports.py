# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from pdf2image import convert_from_path

from PIL import Image
from pypdf import PdfReader


def download_pdf(
    url: str,
    report_id: str,
    base_directory_path: str,
    missing_reports: set,
    max_retries: int,
) -> Optional[str]:
    """
    Downloads a PDF file from a URL and saves it to a local file.

    :param url: The URL of the PDF file to download.
    :param report_id: The ID of the report to download.
    :param downloaded_pdf_path: The path to save the downloaded PDF file.
    :param missing_reports: A set to add the report ID to if the download fails.
    :param max_retries: The maximum number of retries to attemptting to download the url when it fails.

    :return: The path to the downloaded PDF file, or None if the download failed.
    """
    downloaded_pdf_path = os.path.join(base_directory_path, report_id + ".pdf")
    if not os.path.exists(downloaded_pdf_path):
        retries = 0
        while retries <= max_retries:
            try:
                print(f"Attempting to download PDF from '{url}'")
                response = requests.get(url, allow_redirects=True)
                if response.status_code == 200:
                    with open(downloaded_pdf_path, "wb") as f:
                        f.write(response.content)
                    print(f"Successfully downloaded PDF at '{downloaded_pdf_path}'.")
                    return downloaded_pdf_path
                else:
                    print(
                        f"Failed to download URL '{url}' (attempt {retries + 1}/{max_retries + 1}) with status code '{response.status_code}. Retrying...'"
                    )
                    if retries < max_retries:
                        time.sleep(1)  # Wait for 1 second before trying again
                    retries += 1
                    missing_reports.add(report_id)
            except requests.exceptions.RequestException as e:
                print(
                    f"Request failed (attempt {retries + 1}/{max_retries + 1}): '{e}'. Retrying..."
                )
                if retries < max_retries:
                    time.sleep(1)
                retries += 1
                missing_reports.add(report_id)
    return None


def convert_pdf_to_text(
    report_id: str,
    downloaded_pdf_path: Optional[str],
    base_directory_path: str,
    missing_text_files: set,
) -> None:
    """
    Converts a PDF file to a single text file.

    :param downloaded_pdf_path: The input path to the downloaded PDF file.
    :param converted_text_file_path: The output path of the converted text file.
    :param base_directory_path: The base directory path to save the converted text file.
    :param missing_text_files: A set to add the report ID to if the conversion fails.
    """
    converted_text_file_path = os.path.join(base_directory_path, report_id + ".txt")

    if not downloaded_pdf_path:
        missing_text_files.add(report_id)
        return

    if not os.path.exists(converted_text_file_path):
        try:
            print(f"Converting PDF to text at '{downloaded_pdf_path}'")
            pdf = PdfReader(downloaded_pdf_path)
            with open(converted_text_file_path, "w") as f:
                for page in pdf.pages:
                    f.write(page.extract_text())
            print(f"Converted PDF to text at '{converted_text_file_path}'")
        except Exception as e:
            print(f"Error converting PDF to PNG: {e}")
            missing_text_files.add(report_id)


def convert_pdf_to_multiple_images(
    report_id: str,
    downloaded_pdf_path: Optional[str],
    base_directory_path: str,
    missing_images: set,
) -> None:
    """
    Converts a PDF file to a single PNG image.

    :param downloaded_pdf_path: The input path to the downloaded PDF file.
    :param converted_image_path: The output path of the converted image file.
    :param base_directory_path: The base directory path to save the converted image file.
    :param missing_images: A set to add the report ID to if the conversion fails.
    """
    if not downloaded_pdf_path:
        missing_images.add(report_id)
        return

    converted_images_path = os.path.join(base_directory_path, report_id)

    try:
        images = convert_from_path(downloaded_pdf_path, dpi=100)
        total_height = sum(image.height for image in images)
        max_width = max(image.width for image in images)
        combined_image = Image.new("L", (max_width, total_height))
        y_offset = 0
        for image in images:
            combined_image.paste(image, (0, y_offset))
            y_offset += image.height
        height = 7900
        num_images = -(-total_height // height)  # Ceiling division
        output_images = []
        for i in range(num_images):
            top = i * height
            bottom = min((i + 1) * height, total_height)
            output_images.append(combined_image.crop((0, top, max_width, bottom)))
        for i, image in enumerate(output_images):
            image_path = f"{converted_images_path}__{i}__.png"
            image.save(image_path)

    except Exception as e:
        print(f"Error converting PDF to PNG: {e}")
        missing_images.add(report_id)


def main(base_directory_path: str, json_name: str, max_retries: int) -> None:
    """
    Attempts to download and convert all PDFs referenced in the JSON file to text and images.

    :param base_directory_path: The base directory path to save the converted files.
    :param json_name: The name of the JSON file to read that has the URLs to download.
    """
    # Load JSON data from the file
    json_data_path = os.path.join(base_directory_path, json_name)
    with open(json_data_path, "r") as file:
        json_data = json.load(file)

    # Holds report IDs for files that were not downloaded or converted
    missing_reports = set()
    missing_text_files = set()
    missing_images = set()

    # Iterate through each entry in the JSON data
    for entry in json_data:
        report_id = entry["report_id"]
        url = entry["url_source"]
        source = entry["source"]

        if source == "CrowdStrike":
            downloaded_pdf_path = str(
                Path(base_directory_path, "crowdstrike-reports", report_id + ".pdf")
            )
        else:
            # Download PDF
            downloaded_pdf_path = download_pdf(
                url, report_id, base_directory_path, missing_reports, max_retries
            )

        # Convert the PDF -> Text file
        convert_pdf_to_text(
            report_id, downloaded_pdf_path, base_directory_path, missing_text_files
        )

        # Convert the PDF -> Image
        convert_pdf_to_multiple_images(
            report_id, downloaded_pdf_path, base_directory_path, missing_images
        )

    print(f"Missing Reports: {missing_reports}")
    print(f"Missing Text files: {missing_text_files}")
    print(f"Missing Images: {missing_images}")


if __name__ == "__main__":
    # Get $DATASETS environment variable
    datasets_path = os.environ.get("DATASETS", "./CybersecurityBenchmarks/datasets")

    main(
        base_directory_path=str(
            Path(datasets_path) / "crwd_meta" / "threat_intel_reasoning"
        ),
        json_name="report_questions.json",
        max_retries=2,
    )
