import json

import pytest

from promptlens.loaders.json_loader import JSONLoader
from promptlens.loaders.yaml_loader import YAMLLoader


def test_json_loader_rejects_non_object_root(tmp_path):
    p = tmp_path / "golden.json"
    p.write_text(json.dumps([{"id": "tc-1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="root value must be a JSON object"):
        JSONLoader().load(str(p))


def test_yaml_loader_rejects_non_mapping_root(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text("- id: tc-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root value must be a YAML mapping"):
        YAMLLoader().load(str(p))
