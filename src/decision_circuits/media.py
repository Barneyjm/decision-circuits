"""Media states: an image or a clip as the thing the questions are about.

    c.run(backend, Image("receipt.png"), model="circuit-vl-4b")
    c.run(backend, Audio("call.wav", text="Support line, Tuesday"), model="circuit-audio-7b")

A media state is sent as {"image": <data URI or URL>, "text": <optional>}
(or "audio"), which is what the circuit vision and audio models accept.
Text-only requests are unchanged, so a text model never sees this form.
Nothing here needs a dependency: files are read as bytes and base64
encoded; URLs are passed through for the server to fetch.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Audio", "Image", "Media"]

_DEFAULT_TYPE = {"image": "image/png", "audio": "audio/wav"}


@dataclass(frozen=True)
class Media:
    """One of `source` (a path or a URL) or `data` (bytes), plus an optional caption."""

    kind: str  # "image" | "audio"
    source: str | Path | None = None
    data: bytes | None = None
    media_type: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if (self.source is None) == (self.data is None):
            raise ValueError(f"{type(self).__name__} takes a path or URL, or bytes, not both or neither")

    @property
    def uri(self) -> str:
        """A data URI for local files and bytes; a URL is passed through."""
        if self.data is not None:
            return self._data_uri(self.data, self.media_type or _DEFAULT_TYPE[self.kind])
        src = str(self.source)
        if src.startswith(("http://", "https://", "data:")):
            return src
        mt = self.media_type or mimetypes.guess_type(src)[0] or _DEFAULT_TYPE[self.kind]
        return self._data_uri(Path(src).read_bytes(), mt)

    @staticmethod
    def _data_uri(data: bytes, media_type: str) -> str:
        return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.kind: self.uri}
        if self.text:
            out["text"] = self.text
        return out


class Image(Media):
    """An image state: `Image("receipt.png")`, `Image(url)`, or `Image(data=png_bytes)`."""

    def __init__(self, source: str | Path | None = None, *, data: bytes | None = None, media_type: str | None = None, text: str | None = None) -> None:
        super().__init__("image", source, data, media_type, text)


class Audio(Media):
    """A clip state: `Audio("call.wav")`, `Audio(url)`, or `Audio(data=wav_bytes)`. 16 kHz mono is ideal; the server resamples."""

    def __init__(self, source: str | Path | None = None, *, data: bytes | None = None, media_type: str | None = None, text: str | None = None) -> None:
        super().__init__("audio", source, data, media_type, text)
