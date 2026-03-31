#!/usr/bin/env python3
"""Generate <shape>_black.png from <shape>_green.png under assets/shapes/*/."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_black_shape(green_png: Path, black_png: Path) -> None:
    img = Image.open(green_png).convert("RGBA")
    pixels = img.load()
    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a > 0:
                pixels[x, y] = (0, 0, 0, a)
            else:
                pixels[x, y] = (r, g, b, a)

    img.save(black_png)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    shapes_root = repo_root / "assets" / "shapes"
    if not shapes_root.exists():
        raise FileNotFoundError(f"Missing directory: {shapes_root}")

    generated = 0
    skipped = 0

    for shape_dir in sorted(p for p in shapes_root.iterdir() if p.is_dir()):
        shape = shape_dir.name
        green_png = shape_dir / f"{shape}_green.png"
        black_png = shape_dir / f"{shape}_black.png"

        if not green_png.exists():
            print(f"[SKIP] no green image: {green_png}")
            skipped += 1
            continue

        make_black_shape(green_png, black_png)
        print(f"[OK] {green_png.name} -> {black_png.name}")
        generated += 1

    print(f"Done. generated={generated}, skipped={skipped}")


if __name__ == "__main__":
    main()
