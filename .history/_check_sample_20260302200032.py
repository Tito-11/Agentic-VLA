"""Find remaining 'sample' placeholder text in string literals."""
import os, re, glob

ROOT = r"d:\trae_proj\Agentic-RAG-VLM\robot_grasp_rag"
cnt = 0
for f in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
    for i, l in enumerate(open(f, encoding="utf-8"), 1):
        s = l.strip()
        if s.startswith("import") or s.startswith("from") or s.startswith("#"):
            continue
        # Look for suspicious 'sample' usage in string context
        if '"sample"' in l or "'sample'" in l:
            rel = os.path.relpath(f, ROOT)
            print(f"{rel}:{i}: {l.rstrip()[:120]}")
            cnt += 1
        elif re.search(r'"[^"]*sample[^"]*sample[^"]*"', l):
            rel = os.path.relpath(f, ROOT)
            print(f"{rel}:{i}: {l.rstrip()[:120]}")
            cnt += 1
        elif re.search(r"'[^']*sample[^']*sample[^']*'", l):
            rel = os.path.relpath(f, ROOT)
            print(f"{rel}:{i}: {l.rstrip()[:120]}")
            cnt += 1

print(f"\n({cnt} matches)")
