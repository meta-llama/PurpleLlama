import yaml
import os


config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../config', 'settings.yaml')
)

with open(config_path, "r", encoding="utf-8") as in_file:
    CONFIG: dict = yaml.safe_load(in_file)