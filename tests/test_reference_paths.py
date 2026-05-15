from pathlib import Path

from vam_timeline_ai.io.path_utils import default_reference_paths


def test_reference_paths_have_expected_defaults():
    refs = default_reference_paths()
    project_root = Path(__file__).resolve().parents[1]

    assert refs.project_root == project_root
    assert str(refs.mocap_compiler).endswith(r"vam_mocap_dataset_compiler")
    assert str(refs.timeline_repo).endswith(r"vam-timeline-master")
    assert str(refs.raw_mocap_research).endswith(r"MocapResearch")
    assert str(refs.virtual_companion).endswith(r"Virtual Companion")
