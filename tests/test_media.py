import base64

import pytest

from decision_circuits import Audio, Image
from decision_circuits.types import to_jsonable


def test_image_from_bytes_is_a_data_uri_state():
    st = to_jsonable(Image(data=b"\x89PNG...", media_type="image/png", text="the receipt"))
    assert st["text"] == "the receipt"
    assert st["image"].startswith("data:image/png;base64,")
    assert base64.b64decode(st["image"].split(",", 1)[1]) == b"\x89PNG..."


def test_image_from_path_guesses_type(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"jpegbytes")
    assert Image(p).uri.startswith("data:image/jpeg;base64,")


def test_url_passes_through():
    assert to_jsonable(Audio("https://example.com/call.wav")) == {"audio": "https://example.com/call.wav"}


def test_audio_default_type_and_no_text_key():
    st = to_jsonable(Audio(data=b"RIFF"))
    assert set(st) == {"audio"} and st["audio"].startswith("data:audio/wav;base64,")


def test_needs_exactly_one_source():
    with pytest.raises(ValueError):
        Image()
    with pytest.raises(ValueError):
        Image("a.png", data=b"x")
