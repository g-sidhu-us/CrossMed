from pathlib import Path

from PIL import Image

from crossmed.prepare import prepare_busi


def test_busi_ignores_masks_and_normal(tmp_path: Path):
    for folder in ("benign", "malignant", "normal"):
        (tmp_path / folder).mkdir()
    for relative in ("benign/benign (1).png", "benign/benign (1)_mask.png", "malignant/malignant (1).png", "normal/normal (1).png"):
        Image.new("L", (8, 8)).save(tmp_path / relative)
    frame = prepare_busi(tmp_path)
    assert len(frame) == 2
    assert set(frame.label) == {0, 1}
    assert not frame.sample_id.str.contains("mask").any()

