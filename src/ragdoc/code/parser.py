"""Symbol-level code parsing.

Python uses the standard-library ``ast`` (exact line ranges, docstrings, calls, imports).
Brace languages (JS/TS, Go, Java, Kotlin, C#, Rust, C/C++, Scala, Swift, PHP) use declaration
patterns plus brace matching on comment/string-stripped source, which is robust enough for
chunking and call graphs without native parser dependencies. Other files fall back to
fixed line windows. Classes become an *outline* chunk (signature, fields, member list) while
their methods become their own chunks, so no code is indexed twice.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pydantic import BaseModel, Field

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "csharp",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".scala": "scala",
    ".swift": "swift",
    ".php": "php",
    ".rb": "ruby",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".proto": "proto",
    ".graphql": "graphql",
    ".tf": "terraform",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".gradle": "groovy",
    ".xml": "xml",
}
SPECIAL_FILES = {"Dockerfile": "dockerfile", "Makefile": "make", "Jenkinsfile": "groovy"}
BRACE_LANGUAGES = {
    "javascript",
    "typescript",
    "go",
    "java",
    "kotlin",
    "csharp",
    "rust",
    "c",
    "cpp",
    "scala",
    "swift",
    "php",
}
_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "function",
    "new",
    "else",
    "do",
    "try",
    "sizeof",
    "typeof",
    "await",
    "throw",
    "case",
    "when",
    "match",
    "super",
    "this",
    "print",
    "len",
    "range",
    "make",
    "append",
    "func",
    "fn",
    "def",
    "class",
    "assert",
    "with",
}


def detect_language(path: Path) -> str | None:
    if path.name in SPECIAL_FILES:
        return SPECIAL_FILES[path.name]
    if path.name.startswith("Dockerfile"):
        return "dockerfile"
    return EXTENSIONS.get(path.suffix.lower())


class Symbol(BaseModel):
    name: str
    qualname: str
    kind: str  # function | method | class | interface | struct | module | window
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    calls: list[str] = Field(default_factory=list)
    parent: str | None = None
    outline: str | None = None  # for classes: signature + members instead of the full body


class CodeFile(BaseModel):
    path: str
    language: str
    imports: list[str] = Field(default_factory=list)
    symbols: list[Symbol] = Field(default_factory=list)
    n_lines: int = 0


# --------------------------------------------------------------------------- python


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else fn.attr
                if isinstance(fn, ast.Attribute)
                else None
            )
            if name and name not in names and name not in _KEYWORDS:
                names.append(name)
    return names


def parse_python(text: str, path: str) -> CodeFile:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return parse_windows(text, path, "python")
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    symbols: list[Symbol] = []

    def signature(node: ast.AST) -> str:
        return lines[node.lineno - 1].strip() if hasattr(node, "lineno") else ""  # type: ignore[attr-defined]

    def visit(body: list[ast.stmt], parent: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{parent}.{node.name}" if parent else node.name
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                symbols.append(
                    Symbol(
                        name=node.name,
                        qualname=qual,
                        kind="method" if parent else "function",
                        line_start=start,
                        line_end=node.end_lineno or node.lineno,
                        signature=signature(node),
                        docstring=(ast.get_docstring(node) or "")[:500],
                        calls=_call_names(node),
                        parent=parent,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                qual = f"{parent}.{node.name}" if parent else node.name
                members = [
                    f"    {lines[n.lineno - 1].strip()}"
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                fields = [
                    f"    {lines[n.lineno - 1].strip()}"
                    for n in node.body
                    if isinstance(n, (ast.Assign, ast.AnnAssign))
                ]
                doc = ast.get_docstring(node) or ""
                outline = "\n".join(
                    [signature(node)]
                    + ([f'    """{doc[:400]}"""'] if doc else [])
                    + fields[:30]
                    + members
                )
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                symbols.append(
                    Symbol(
                        name=node.name,
                        qualname=qual,
                        kind="class",
                        line_start=start,
                        line_end=node.end_lineno or node.lineno,
                        signature=signature(node),
                        docstring=doc[:500],
                        parent=parent,
                        outline=outline,
                    )
                )
                visit(node.body, qual)

    visit(tree.body, None)
    return CodeFile(
        path=path, language="python", imports=imports, symbols=symbols, n_lines=len(lines)
    )


# --------------------------------------------------------------------------- brace languages

_DECLARATIONS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "go": [
        ("function", re.compile(r"^func\s+(?:\((?P<recv>[^)]*)\)\s*)?(?P<name>\w+)\s*[\[(]")),
        ("struct", re.compile(r"^type\s+(?P<name>\w+)\s+(?:struct|interface)\b")),
    ],
    "javascript": [
        ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(?P<name>\w+)")),
        (
            "function",
            re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+(?P<name>\w+)"
            ),
        ),
        (
            "function",
            re.compile(
                r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>"
            ),
        ),
        (
            "method",
            re.compile(
                r"^\s+(?:(?:public|private|protected|static|readonly|override)\s+)*(?:async\s+)?(?:get\s+|set\s+)?(?P<name>\w+)\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^{=;]+)?\{"
            ),
        ),
    ],
    "java": [
        (
            "class",
            re.compile(
                r"^\s*(?:@\w+\s*)*(?:(?:public|private|protected|abstract|final|static|sealed)\s+)*(?:class|interface|enum|record)\s+(?P<name>\w+)"
            ),
        ),
        (
            "method",
            re.compile(
                r"^\s+(?:@\w+(?:\([^)]*\))?\s*)*(?:(?:public|private|protected|static|final|abstract|synchronized|default)\s+)+[\w<>\[\],.? ]+\s+(?P<name>\w+)\s*\("
            ),
        ),
    ],
    "kotlin": [
        (
            "class",
            re.compile(
                r"^\s*(?:(?:data|sealed|open|abstract|enum|internal|private|public)\s+)*(?:class|interface|object)\s+(?P<name>\w+)"
            ),
        ),
        (
            "function",
            re.compile(
                r"^\s*(?:(?:override|suspend|private|public|internal|protected|inline|open)\s+)*fun\s+(?:<[^>]+>\s*)?(?:\w+\.)?(?P<name>\w+)\s*\("
            ),
        ),
    ],
    "csharp": [
        (
            "class",
            re.compile(
                r"^\s*(?:\[[^\]]+\]\s*)*(?:(?:public|private|protected|internal|abstract|sealed|static|partial)\s+)*(?:class|interface|struct|record|enum)\s+(?P<name>\w+)"
            ),
        ),
        (
            "method",
            re.compile(
                r"^\s+(?:\[[^\]]+\]\s*)*(?:(?:public|private|protected|internal|static|virtual|override|async|abstract|sealed)\s+)+[\w<>\[\],.? ]+\s+(?P<name>\w+)\s*\("
            ),
        ),
    ],
    "rust": [
        (
            "function",
            re.compile(
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>\w+)"
            ),
        ),
        (
            "struct",
            re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(?P<name>\w+)"),
        ),
        ("class", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(?P<name>\w+)")),
    ],
    "c": [
        (
            "function",
            re.compile(
                r"^(?!\s)(?!(?:if|for|while|switch|return)\b)[\w\*\s&:<>,]+?\b(?P<name>[A-Za-z_][\w:~]*)\s*\([^;]*$"
            ),
        ),
        ("struct", re.compile(r"^(?:typedef\s+)?(?:struct|class|enum|union)\s+(?P<name>\w+)")),
    ],
    "scala": [
        (
            "class",
            re.compile(
                r"^\s*(?:(?:case|abstract|sealed|final)\s+)*(?:class|trait|object)\s+(?P<name>\w+)"
            ),
        ),
        ("function", re.compile(r"^\s*(?:(?:override|private|protected)\s+)*def\s+(?P<name>\w+)")),
    ],
    "swift": [
        (
            "class",
            re.compile(
                r"^\s*(?:(?:public|private|internal|final|open)\s+)*(?:class|struct|protocol|enum|extension)\s+(?P<name>\w+)"
            ),
        ),
        (
            "function",
            re.compile(
                r"^\s*(?:(?:public|private|internal|static|override|open)\s+)*func\s+(?P<name>\w+)"
            ),
        ),
    ],
    "php": [
        (
            "class",
            re.compile(r"^\s*(?:(?:abstract|final)\s+)?(?:class|interface|trait)\s+(?P<name>\w+)"),
        ),
        (
            "function",
            re.compile(r"^\s*(?:(?:public|private|protected|static)\s+)*function\s+(?P<name>\w+)"),
        ),
    ],
}
_DECLARATIONS["typescript"] = _DECLARATIONS["javascript"] + [
    ("interface", re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(?P<name>\w+)")),
]
_DECLARATIONS["cpp"] = _DECLARATIONS["c"]

_STRIP_RE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`",
    re.DOTALL,
)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(?:[\w*{}\s,]+\s+from\s+)?[\"']([^\"']+)[\"']"  # js/ts
    r"|import\s+\(?\s*\"([^\"]+)\""  # go single
    r"|import\s+(?:static\s+)?([\w.]+)\s*;?"  # java/kotlin/scala
    r"|using\s+([\w.]+)\s*;"  # c#
    r"|use\s+([\w:]+)"  # rust
    r"|#include\s+[<\"]([^>\"]+)[>\"])",  # c/c++
    re.MULTILINE,
)


def _blank_strings_and_comments(text: str) -> str:
    """Replace comments/strings with spaces, preserving newlines (so line numbers hold)."""
    return _STRIP_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _match_brace(clean_lines: list[str], start: int) -> int:
    """Line index where the block opened at/after ``start`` closes (or ``start``)."""
    depth = 0
    opened = False
    for i in range(start, len(clean_lines)):
        line = clean_lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return i
        if not opened and (line.rstrip().endswith(";") or i - start > 5):
            return start  # declaration without a body (interface method, prototype)
    return len(clean_lines) - 1


def parse_braced(text: str, path: str, language: str) -> CodeFile:
    lines = text.splitlines()
    clean = _blank_strings_and_comments(text).splitlines()
    patterns = _DECLARATIONS.get(language, [])
    imports = [next(g for g in m.groups() if g) for m in _IMPORT_RE.finditer(text)]

    raw: list[Symbol] = []
    for i, line in enumerate(clean):
        for kind, pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group("name")
            if name in _KEYWORDS:
                break
            end = _match_brace(clean, i)
            # Include leading doc comments / annotations in the symbol.
            start = i
            while start > 0 and lines[start - 1].strip().startswith(
                ("//", "*", "/*", "@", "#[", "///")
            ):
                start -= 1
            body = "\n".join(clean[i : end + 1])
            calls = [
                c for c in dict.fromkeys(_CALL_RE.findall(body)) if c not in _KEYWORDS and c != name
            ]
            if language == "go" and match.groupdict().get("recv"):
                kind = "method"
            raw.append(
                Symbol(
                    name=name,
                    qualname=name,
                    kind=kind,
                    line_start=start + 1,
                    line_end=end + 1,
                    signature=lines[i].strip(),
                    calls=calls[:60],
                )
            )
            break

    # Nesting: methods inside a class get a qualified name; classes get an outline.
    containers = [
        s for s in raw if s.kind in ("class", "struct", "interface") and s.line_end > s.line_start
    ]
    symbols: list[Symbol] = []
    for s in raw:
        parent = next(
            (
                c
                for c in containers
                if c is not s and c.line_start <= s.line_start and s.line_end <= c.line_end
            ),
            None,
        )
        if parent is not None:
            s = s.model_copy(
                update={
                    "parent": parent.name,
                    "qualname": f"{parent.name}.{s.name}",
                    "kind": "method",
                }
            )
        symbols.append(s)
    for idx, s in enumerate(symbols):
        if s in containers or (
            s.kind in ("class", "struct", "interface") and s.line_end > s.line_start
        ):
            members = ["    " + m.signature for m in symbols if m.parent == s.name and m is not s]
            if members:
                symbols[idx] = s.model_copy(
                    update={"outline": "\n".join([s.signature, *members, "}"])}
                )
    return CodeFile(
        path=path, language=language, imports=imports, symbols=symbols, n_lines=len(lines)
    )


# --------------------------------------------------------------------------- fallback + entry


def parse_windows(text: str, path: str, language: str, window: int = 120) -> CodeFile:
    n = len(text.splitlines())
    symbols = [
        Symbol(
            name=f"{Path(path).name}:{i + 1}",
            qualname=f"{Path(path).name}:{i + 1}",
            kind="window",
            line_start=i + 1,
            line_end=min(n, i + window),
        )
        for i in range(0, max(n, 1), window)
    ]
    return CodeFile(path=path, language=language, symbols=symbols, n_lines=n)


def parse_code(text: str, path: str, language: str) -> CodeFile:
    if language == "python":
        return parse_python(text, path)
    if language in BRACE_LANGUAGES:
        parsed = parse_braced(text, path, language)
        return parsed if parsed.symbols else parse_windows(text, path, language)
    return parse_windows(text, path, language)
