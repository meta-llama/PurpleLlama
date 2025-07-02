# ClassifyIt: Google Workspace Bulk Content Classification

ClassifyIt is a variant of an internal processor we created to perform content
analysis and data categorization of files stored in our company Google
Workspace. It leverages the Google Drive Files API to authenticate to Drive and
retrieve files for classification, Llama Stack to send the extracted text into a
running Llama Stack server for content analysis and classification, Apache Tika
to extract content from complex file types, and the Google Drive API to apply
content tags on files detected to match the prompt.

After classification, the tool will output a CSV file with the file ID, file
name, MIME type, model response, and a list of actions taken by the processor.

Applying Google Workspace labels to files is a highly valuable feature for many
organizations, and this tool is a simple way to automate that process.

# Setup

## Google Workspace & GCP setup

You need to configure a Google Cloud Platform project with a Service Account.
The Service Account has to have Domain-Wide Delegation to your Workspace domain
with the scope "https://www.googleapis.com/auth/drive", permitting the SA to
read and write file contents and metadata on behalf of drive organizers in your
Workspace.

Setup guide:
https://developers.google.com/identity/protocols/oauth2/service-account

After setting up the Service Account with DWD, export a Service Account
authentication key in JSON format and store it in a secure path.

To assign labels to a file on LLM content match, set up a label in Google
Workspace and publish it to Google Drive.

Setup guide: https://support.google.com/a/answer/13127870?hl=en

To obtain the label's Fields and Selection Values, the simplest way is to assign
the label to a test file and then check the metadata in the Google Drive
Files.get API:

API reference: https://developers.google.com/drive/api/v3/reference/files/get

Include the label ID in the "includeLabels" parameter in the API request.

## Llama Stack setup

To use Llama to classify the content, you need to have Llama Stack configured
and running. Llama Stack supports LLM providers running either locally or on a
remote server.

Setup guide: https://llama-stack.readthedocs.io/. The Quick Start guides you
through setup of a local Llama 3.2:3b LLM provider, using Ollama as the
provider.

# Installation

You can install ClassifyIt as a Python package + CLI:

```bash
pip install -e .
```

This will install the ClassifyIt package and make the `classifyit` command
available in your PATH.

# Tests

Unit tests can be run using the `unittest` module.

```
python -munittest tests/*py
```

# CLI usage

Once setup is complete and you have the desired target label values, invoke the
tool:

    classifyit \
        --drive_type my_drive \
        --drive_owner "username@domain.com" \
        --service_account_file "/path/to/key.json" \
        --output_dir "/path/to/output" \
        --session_config "/path/to/session_config.yaml"

For a Shared Drive, also provide the Shared Drive ID with the --shared_drive_id
CLI flag:

```
--shared_drive_id foobar
```

## Database Support

ClassifyIt now includes SQLite database support to track Google Drive file
metadata and classification results. This allows the tool to:

1. Track which files have been classified and when
2. Only process files that are new or have been modified since the last scan
3. Maintain a history of classification results

By default, the database is stored at
`/tmp/{drive_type}_{owner}_{label_id}.sqlite`. The database name includes the
drive type (my_drive or shared_drive), the owner's email, and the label ID to
ensure separate tracking for different label configurations. You can specify a
custom path using the `--db_path` flag:

```
--db_path "/path/to/custom/database.sqlite"
```

The database schema includes the following fields:

- `fileId`: The file ID (primary key)
- `driveId`: The ID of the drive ("my_drive" or the shared drive ID)
- `fileExtension`: The file extension
- `mimeType`: The file mimetype, as reported by Google
- `md5Checksum`: The MD5 checksum of the file
- `size`: The size of the file
- `name`: The file name
- `createdTime`: When the file was created
- `modifiedTime`: When the file was last modified
- `lastSeen`: When the file was last seen by the ClassifyIt drive scanner

The Label table includes:

- `fileId`: Foreign key to File table
- `labelId`: The Google Workspace label ID
- `selectionId`: The selected value for the label
- `lastClassifiedTime`: When the file was last classified
- `lastClassificationExplanation`: The model explanation from the last
  classification

## Session Configuration

The session configuration is a YAML file that contains the settings for the
Llama Stack server, agent settings and prompt, Google Drive label information,
and valid classification categories. Here's an example:

````yaml
llama_server:
  host: localhost
  port: 8321
  model: llama3.2:3b

agent_settings:
  agent_instructions: "You are a security analyst who reviews documents to determine the type of data they contain. Answer the following question about the provided document as best you can and respond in the format requested."
  prompt: "Look for examples of PII in this file. Return a JSON object with the content category and brief explanation. For example: ```json {'category_matched': 'pii_present', 'brief_explanation': 'This document contains first name, last name, email, and phone number.'}``` or ```json {'category_matched': 'pii_not_present', 'brief_explanation': 'This document contains no information that could be used to uniquely identify an individual.'}```""
  response_categories:
    pii_present:
      policy_title: Personally Identifiable Information present
      policy_details:
        "This data includes employee, candidate, or customer contact information, addresses, national or taxpayer identity numbers, bank account numbers and more"
    pii_not_present:
      policy_title: No Personally Identifiable Information present
      policy_details:
        "This data contains no information that could be used to uniquely identify an individual"

google_drive_label_settings:
  label_id: 3J1dNlN7Q8tzLNDehYXliSpafrUXJLkJSzdSNNEbbFcb
  field_id: 0526F309F1
  selection_ids:
    pii_present: BA7821B229
    pii_not_present: 4F8E36952E
````

## Document parsing

Native Google Drive files are parsed using the Drive API's export functionality.
Most other documents are parsed using Apache Tika, which supports a wide range
of file types.

Parsed file output is capped at 1MB of text to prevent overwhelming the LLM.
Non-Google Workspace filetypes (e.g. .docx, .pdf) are also capped at 10MB total
size before content extraction.

## Output CSV

The CSV output contains the file ID, file name, MIME type, model response,
action taken, and any errors:

| file_id                                      | file_name         | mime_type                            | model_response | action_taken  | error |
| -------------------------------------------- | ----------------- | ------------------------------------ | -------------- | ------------- | ----- |
| 1nRcA7HD2-keSojJHJ9vR1vAFk06sVDSyzSWkaUqRwbQ | New Document Name | application/vnd.google-apps.document | Match Found.   | Applied label |       |

The `action_taken` column can have the following values:

- "Applied label": Label was successfully applied
- "Dry run": Would have applied label in non-dry-run mode
- "No match": No classification match found
- "Skipped": File was skipped due to errors or unsupported format

## Python API

You can also use ClassifyIt as a Python package:

```python
import yaml
from classifyit import ClassifyIt

# Load session configuration
with open('session_config.yaml', 'r') as file:
    session_config = yaml.safe_load(file)

# Initialize ClassifyIt for a specific drive owner
classifier = ClassifyIt(
    drive_owner="username@domain.com",
    service_account_file="/path/to/key.json",
    session_config=session_config,
    output_dir="/path/to/output",
    verbose=True,
    dry_run=True,  # Enable dry run mode to preview classification without applying labels
    db_path="/path/to/database.sqlite"  # Optional
)

# Scan a drive
classifier.scan_drive(
    drive_type="my_drive",
    shared_drive_id=None  # Optional, required for shared drives
)
```

### Dry Run Mode

The `dry_run` parameter allows you to preview classification results without
applying labels. In dry run mode:

- Files are scanned and classified as normal
- Classification results are exported to CSV
- No labels are applied to files in Google Drive
- Database `lastSeen` timestamps are not updated, allowing you to run the
  classifier again without dry run mode to reclassify and apply the labels

This is particularly useful for:

- Previewing what files would be labeled
- Testing classification accuracy
- Validating label settings before applying them

## Docker

To build the Docker image, run the following command:

```
docker build -t classifyit .
```

The Docker image will run the scanner.py script with the provided arguments. In
order to provide the Service Account keyfile, you need to mount the file to the
container. If you want to save the output, you need to mount an output volume to
the container as well.

Accessing a local Llama Stack server requires the `--net=host` network setting,
or hosting the server on a remote machine and providing a host and port in the
Session Config.

Example command:

```
docker run -it \
  -v /tmp/session_config.yaml:/tmp/session_config.yaml \
  -v /path/to/key.json:/tmp/gdrive_creds/key.json \
  -v /path/to/local/output:/tmp/tool_output \
  --net=host \
  classifyit \
    --drive_type shared_drive \
    --drive_owner "username@domain.com" \
    --shared_drive_id "foobar123" \
    --service_account_file /tmp/gdrive_creds/key.json \
    --output_dir /tmp/tool_output \
    --session_config /tmp/session_config.yaml
```

### OSX notes:

The Docker stack on OSX does not support the `--net=host` flag, so you can
either set up a virtual Docker network (with Llama Stack hosted as a container
on that network) or host the Llama Stack server on a remote machine as described
above.

# Thanks

A special thanks to Chris A. Mattmann, JPL, for his excellent Tika-Python
library.

## License

ClassifyIt is MIT licensed, as found in the LICENSE file.
