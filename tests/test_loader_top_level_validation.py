import json

import pytest
import yaml

from promptlens.loaders.json_loader import JSONLoader
from promptlens.loaders.yaml_loader import YAMLLoader


def test_json_loader_rejects_non_object_top_level(tmp_path):
    file_path = tmp_path / "golden_set.json"
    file_path.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON object at top level"):
        JSONLoader().load(str(file_path))


def test_yaml_loader_rejects_non_mapping_top_level(tmp_path):
    file_path = tmp_path / "golden_set.yaml"
    file_path.write_text(yaml.safe_dump([{"id": "x"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a YAML mapping at top level"):
        YAMLLoader().load(str(file_path))
