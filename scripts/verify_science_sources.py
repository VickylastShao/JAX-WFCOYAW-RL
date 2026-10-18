#!/usr/bin/env python3
"""Verify indexed science files without executing scientific modules."""
import ast
import hashlib
import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / "science/SOURCE_INDEX.json").read_text())
    checked = 0
    for row in index["records"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe source-index member")
        path = root / rel
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Scientific source SHA differs: " + row["path"])
        if path.suffix == ".py":
            ast.parse(data.decode("utf-8"), filename=row["path"])
        checked += 1
    print(json.dumps({"status": "pass", "source_files": checked,
                      "python_files": index["python_files"], "scientific_modules_executed": False,
                      "new_GPU_hours": 0}))

if __name__ == "__main__":
    main()
