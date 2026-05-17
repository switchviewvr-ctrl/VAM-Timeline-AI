from pathlib import Path

from vam_timeline_ai.io.path_utils import default_reference_paths


def test_reference_paths_have_expected_defaults():
    refs = default_reference_paths()
    project_root = Path(__file__).resolve().parents[1]

    assert refs.project_root == project_root
    assert str(refs.mocap_compiler).endswith(r"vam_mocap_dataset_compiler")
    assert str(refs.timeline_repo).endswith(r"vam-timeline")
    assert str(refs.raw_mocap_research).endswith(r"data\raw\vam_scenes") or str(refs.raw_mocap_research).endswith("data/raw/vam_scenes")
    assert str(refs.virtual_companion).endswith(r"virtual_companion")
