#!/usr/bin/env python3
"""Repair truncated def-rule signature bodies in docs/spec/port/.

Wave-1 markup seeded each function's `def` rule with the signature from
the port manifest. Manifests written before the nplan signature fix cut
multi-line signatures at the first line ("Tensor& add_out("), so the
seeded def bodies are truncated. This replaces any def body line that is
a strict prefix of the (now full) manifest signature with the full one.

Idempotent; safe to re-run after `nplan port extract`.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "plan" / ".port-manifest.styx"
SPEC_ROOT = REPO / "docs" / "spec" / "port"


def norm(s: str) -> str:
    return " ".join(s.split())


def manifest_signatures() -> dict[str, str]:
    sigs = {}
    entry = re.compile(r'\{id (\S+), kind @\w+, name \w+, qualified \S+, signature "((?:[^"\\]|\\.)*)"')
    for m in entry.finditer(MANIFEST.read_text()):
        sigs[m.group(1)] = m.group(2).replace('\\"', '"')
    return sigs


def main() -> int:
    sigs = manifest_signatures()
    marker = re.compile(r"^> \[spec:et:def:([a-z0-9.-]+)(\+\d+)?\]\s*$")
    repaired = 0
    for path in sorted(SPEC_ROOT.rglob("*.md")):
        lines = path.read_text().splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines[:-1]):
            m = marker.match(line)
            if not m or m.group(1) not in sigs:
                continue
            body = lines[i + 1]
            if not body.startswith("> "):
                continue
            old = norm(body[2:])
            new = norm(sigs[m.group(1)])
            if old != new and new.startswith(old) and old.endswith("("):
                lines[i + 1] = f"> {new}\n"
                changed = True
                repaired += 1
        if changed:
            path.write_text("".join(lines))
    print(f"repaired {repaired} truncated def signature(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
