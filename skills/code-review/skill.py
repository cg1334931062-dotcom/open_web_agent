"""Simple code review skill - counts lines and checks for basic issues."""
import os

files = args.get("files", [])
# workspace is injected as a separate variable (not inside args)
ws = workspace if isinstance(workspace, str) else str(workspace)

results = []
for f in files:
    path = os.path.join(ws, f)
    if not os.path.exists(path):
        results.append(f"⚠️ {f}: File not found")
        continue
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    issues = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped and len(line) > 120:
            issues.append(f"  - Line {i}: Line too long ({len(line)} chars)")
        if "TODO" in stripped:
            issues.append(f"  - Line {i}: Contains TODO")
        if "FIXME" in stripped:
            issues.append(f"  - Line {i}: Contains FIXME")
    report = f"📄 {f}: {len(lines)} lines"
    if issues:
        report += "\n" + "\n".join(issues[:5])
        if len(issues) > 5:
            report += f"\n  ... and {len(issues) - 5} more issues"
    else:
        report += " ✅ No issues found"
    results.append(report)

result = "\n\n".join(results) or "No files specified."
