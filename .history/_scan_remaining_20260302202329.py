"""Scan for remaining placeholder strings."""
import os, re

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"
patterns = [
    r'"[^"]*\bsample\b[^"]*"',  # "...sample..." in strings
    r"'[^']*\bsample\b[^']*'",  # '...sample...' in strings
    r"General factor",
    r"General adjustment",
    r"\bobjectEN\b",
    r"\bclassEN\b",
    r"\bgraspEN\b",
    r"\bplanningEN\b",
    r"\bfailureEN\b",
    r"\bexperiencesEN\b",
    r"\binferenceEN\b",
    r"\bargsEN\b",
    r"\bpathEN\b",
    r"\bsceneEN\b",
    r"\bgripperEN\b",
    r"\btargetEN\b",
    r"\bimageEN\b",
    r"\bpathplanningEN\b",
    r"\bfailureclassEN\b",
    r"\bENgrasp\b",
    r"\bENReAct\b",
    r"\bgraspposition\b",
    r"\bsampleEN\b",
]
combined = "|".join(patterns)
rgx = re.compile(combined)

count = 0
for root_dir, dirs, files in os.walk(ROOT):
    for fn in files:
        if fn.endswith(".py"):
            fp = os.path.join(root_dir, fn)
            rel = os.path.relpath(fp, ROOT)
            with open(fp, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    # Skip function/class names like create_sample_dataset
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith("def create_sample") or stripped.startswith('"""create_sample'):
                        continue
                    if "create_sample" in stripped and ("def " in stripped or '"""' in stripped):
                        continue
                    if "sample_" in stripped and '"' not in stripped and "'" not in stripped:
                        continue
                    if rgx.search(line):
                        count += 1
                        print(f"{rel}:{i}: {stripped}")

print(f"\nTotal remaining: {count}")
