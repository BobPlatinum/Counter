from pathlib import Path


def safe_relpath(file_path: Path, source: Path) -> Path:
    """
    批量模式下保持子目录结构；单文件则只返回文件名。
    """
    if source.is_dir():
        return file_path.relative_to(source)
    return Path(file_path.name)

