"""Count lines of code matching a glob pattern."""
import os
from pathlib import Path

pattern = args.get("pattern", "**/*.py")
workspace = Path(args.get("workspace", "."))

total_lines = 0
total_files = 0
file_details = []

for p in workspace.glob(pattern):
    if p.is_file():
        try:
            lines = p.read_text("utf-8", errors="replace").count("\n")
            total_lines += lines
            total_files += 1
            file_details.append(f"  {p.relative_to(workspace)}: {lines} lines")
        except Exception:
            pass

if total_files == 0:
    result = f"No files found matching '{pattern}'"
else:
    result = f"📊 {total_files} files, {total_lines} total lines\n"
    result += f"  Average: {total_lines // max(total_files, 1)} lines/file\n\n"
    result += "\n".join(file_details[:50])
    if len(file_details) > 50:
        result += f"\n  ... and {len(file_details) - 50} more files"
