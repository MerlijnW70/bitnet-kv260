"""Fail if any script or code file in the tree carries a comment.

usage: python3 tools/no_comments.py [ROOT]
       (ROOT defaults to the repository this file sits in)

This repository states what its code does in the code, in docs/ and in results/, never in comments.
The rule is absolute for .c, .h, .py, .sh, .tcl, .v and .dtso files, and CI runs this on every push.

What is not a comment and is left alone: a shebang line; a C or Verilog preprocessor directive
(#define, #include, `ifdef); a Python docstring; a Verilog attribute (* ... *); integer division //
in Python; a device-tree property name such as #address-cells; a "#" or "//" inside a string or a
character literal.

exit 0 nothing found, 1 a comment was found, 2 a file could not be read.
"""
import io
import os
import sys
import tokenize

SUFFIXES = (".c", ".h", ".py", ".sh", ".tcl", ".v", ".dtso")
SKIP_DIRS = {".git", "__pycache__", "vectors", "reports", "results", "THIRD_PARTY_LICENSES"}


def python_comments(text):
    found = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                if tok.start[0] == 1 and tok.string.startswith("#!"):
                    continue
                found.append((tok.start[0], tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return found


def slash_comments(text):
    found = []
    line, col, n = 1, 0, len(text)
    state = "code"
    start_line = 0
    while col < n:
        ch = text[col]
        nxt = text[col + 1] if col + 1 < n else ""
        if ch == "\n":
            line += 1
            if state == "line":
                state = "code"
            col += 1
            continue
        if state == "code":
            if ch == '"':
                state = "dq"
            elif ch == "'":
                state = "sq"
            elif ch == "/" and nxt == "/":
                found.append((line, "//"))
                state = "line"
                col += 2
                continue
            elif ch == "(" and nxt == "*":
                state = "attr"
                col += 2
                continue
            elif ch == "/" and nxt == "*":
                start_line = line
                state = "block"
                col += 2
                continue
        elif state == "dq":
            if ch == "\\":
                col += 2
                continue
            if ch == '"':
                state = "code"
        elif state == "sq":
            if ch == "\\":
                col += 2
                continue
            if ch == "'":
                state = "code"
        elif state == "attr":
            if ch == "*" and nxt == ")":
                state = "code"
                col += 2
                continue
        elif state == "block":
            if ch == "*" and nxt == "/":
                found.append((start_line, "/* */"))
                state = "code"
                col += 2
                continue
        col += 1
    if state == "block":
        found.append((start_line, "/* unterminated"))
    return found


def shell_comments(text):
    found = []
    for number, raw in enumerate(text.splitlines(), 1):
        if number == 1 and raw.startswith("#!"):
            continue
        state = "code"
        prev = " "
        for pos, ch in enumerate(raw):
            if state == "code":
                if ch == "'":
                    state = "sq"
                elif ch == '"':
                    state = "dq"
                elif ch == "\\":
                    state = "esc"
                elif ch == "#" and prev in " \t" and not raw[:pos].rstrip().endswith("$"):
                    found.append((number, raw.strip()[:60]))
                    break
            elif state == "sq":
                if ch == "'":
                    state = "code"
            elif state == "dq":
                if ch == '"':
                    state = "code"
            elif state == "esc":
                state = "code"
            prev = ch
        else:
            if raw.strip().startswith("#") and not raw.strip().startswith("#!"):
                found.append((number, raw.strip()[:60]))
    return found


def tcl_comments(text):
    found = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            found.append((number, stripped[:60]))
    return found


def check(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    suffix = os.path.splitext(path)[1]
    if suffix == ".py":
        return python_comments(text)
    if suffix in (".c", ".h", ".v", ".dtso"):
        return slash_comments(text)
    if suffix == ".sh":
        return shell_comments(text)
    if suffix == ".tcl":
        return tcl_comments(text)
    return []


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits, files = 0, 0
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS]
        for name in sorted(names):
            if not name.endswith(SUFFIXES):
                continue
            path = os.path.join(base, name)
            files += 1
            try:
                found = check(path)
            except OSError as exc:
                print(f"cannot read {path}: {exc}", file=sys.stderr)
                return 2
            for number, snippet in found:
                rel = os.path.relpath(path, root).replace(os.sep, "/")
                print(f"{rel}:{number}: comment: {snippet}")
                hits += 1
    print(f"{files} files checked, {hits} comments found")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
