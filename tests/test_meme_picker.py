'''Tests for app.meme_picker. Real tiny image files in tmp_path, no camera.'''

import random

import cv2
import numpy as np

from app.meme_picker import MemePicker


def _write_image(path, channels=4, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((4, 6, channels), value, dtype=np.uint8))


def _library(tmp_path, **counts):
    '''Create memes/<label>/<n>.png with the given number of files per label.'''
    for label, count in counts.items():
        (tmp_path / label).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            _write_image(tmp_path / label / f"{i}.png", value=i * 10)
    return MemePicker(tmp_path, rng=random.Random(0))


def test_pick_returns_a_meme_from_the_labels_category(tmp_path):
    picker = _library(tmp_path, happy=2, sad=1)

    meme = picker.pick("happy")

    assert meme is not None
    assert meme.path.parent.name == "happy"
    assert meme.image.shape == (4, 6, 4)
    assert meme.image.dtype == np.uint8


def test_pick_never_repeats_the_previous_meme_for_that_label(tmp_path):
    picker = _library(tmp_path, happy=3)

    paths = [picker.pick("happy").path for _ in range(50)]

    assert all(a != b for a, b in zip(paths, paths[1:]))


def test_a_category_with_one_meme_repeats_it(tmp_path):
    picker = _library(tmp_path, happy=1)

    assert picker.pick("happy").path == picker.pick("happy").path


def test_the_no_repeat_memory_is_per_label(tmp_path):
    picker = _library(tmp_path, happy=2, sad=2)
    first_happy = picker.pick("happy").path

    picker.pick("sad")

    assert picker.pick("happy").path != first_happy


def test_empty_category_and_unknown_label_give_no_meme(tmp_path):
    picker = _library(tmp_path, happy=1, neutral=0)

    assert picker.pick("neutral") is None
    assert picker.pick("angry") is None


def test_images_without_alpha_get_an_opaque_alpha_channel(tmp_path):
    _write_image(tmp_path / "happy" / "photo.jpg", channels=3)
    picker = MemePicker(tmp_path)

    meme = picker.pick("happy")

    assert meme.image.shape[2] == 4
    assert (meme.image[:, :, 3] == 255).all()


def test_extensions_are_matched_case_insensitively(tmp_path):
    _write_image(tmp_path / "happy" / "PHOTO.JPG", channels=3)

    assert MemePicker(tmp_path).pick("happy") is not None


def test_unreadable_files_are_skipped_and_reported(tmp_path):
    _write_image(tmp_path / "happy" / "ok.png")
    (tmp_path / "happy" / "anim.gif").write_bytes(b"not an image")
    (tmp_path / "happy" / "notes.txt").write_text("hi")
    picker = MemePicker(tmp_path)

    assert picker.pick("happy").path.name == "ok.png"
    assert "anim.gif" in picker.startup_warning()
    assert "notes.txt" not in picker.startup_warning()


def test_startup_warning_names_every_empty_category(tmp_path):
    picker = _library(tmp_path, happy=1, neutral=0, sad=0)

    warning = picker.startup_warning()

    assert "neutral" in warning and "sad" in warning
    assert "happy" not in warning


def test_a_category_holding_only_unreadable_files_counts_as_empty(tmp_path):
    (tmp_path / "sad").mkdir()
    (tmp_path / "sad" / "anim.gif").write_bytes(b"not an image")
    picker = MemePicker(tmp_path)

    assert picker.pick("sad") is None
    assert "sad" in picker.startup_warning()


def test_no_warning_when_every_category_has_a_meme(tmp_path):
    assert _library(tmp_path, happy=1, sad=1).startup_warning() is None


def test_a_missing_library_folder_warns_instead_of_crashing(tmp_path):
    picker = MemePicker(tmp_path / "nope")

    assert picker.pick("happy") is None
    assert "nope" in picker.startup_warning()
