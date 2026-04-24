---
name: count-lines
description: Count lines of code in files matching a glob pattern
parameters:
  type: object
  properties:
    pattern:
      type: string
      description: Glob pattern to match files (e.g. "**/*.py")
      default: "**/*.py"
  required: []
run:
  type: python
  file: skill.py
---

# Count Lines

Count lines of code in files matching a glob pattern.
