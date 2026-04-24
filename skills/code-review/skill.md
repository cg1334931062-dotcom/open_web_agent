---
name: code-review
description: Review code changes in files for quality, bugs, and security issues
parameters:
  type: object
  properties:
    files:
      type: array
      items: { type: string }
      description: List of file paths to review
  required: []
run:
  type: python
  file: skill.py
---

# Code Review

Review code changes for quality, bugs, and security issues.
