"""Download the pinned public papers used by the retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "benchmark" / "papers.json"
OUTPUT_DIR = ROOT / "data" / "benchmark" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(follow_redirects=True, timeout=90.0) as client:
        for paper in papers:
            destination = OUTPUT_DIR / paper["filename"]
            if not destination.exists():
                temporary = destination.with_suffix(".pdf.part")
                with client.stream("GET", paper["url"]) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as target:
                        for block in response.iter_bytes():
                            target.write(block)

                with temporary.open("rb") as downloaded:
                    is_pdf = downloaded.read(4) == b"%PDF"
                if not is_pdf:
                    temporary.unlink()
                    raise ValueError(f"Invalid PDF response: {paper['url']}")
                expected = paper.get("sha256")
                if expected and sha256(temporary) != expected:
                    temporary.unlink()
                    raise ValueError(f"SHA-256 mismatch: {destination.name}")
                temporary.replace(destination)

            digest = sha256(destination)
            expected = paper.get("sha256")
            if expected and digest != expected:
                raise ValueError(f"SHA-256 mismatch: {destination.name}")
            print(f"{destination.name}: {digest}")


if __name__ == "__main__":
    main()
