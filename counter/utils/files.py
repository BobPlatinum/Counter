from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    imgs = [p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)

