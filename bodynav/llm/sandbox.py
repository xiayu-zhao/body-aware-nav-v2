"""Static allowlist and execution sandbox for generated code (Sec. III-D).

The manuscript's claim is that the agent emits **schema-validated
executable code over an approved toolset**, not free Python. Two gates
enforce that, in this order:

1. a static pass over the AST that rejects anything outside a small
   allowlist -- no imports, no attribute access except on ``np``, no
   dunders, no comprehensible route to ``__builtins__``;
2. execution in a namespace containing the toolset and numpy and nothing
   else, with builtins removed.

The static pass runs first and on the *text*, so unsafe code is never
executed even once. Neither gate is a sandbox in the security sense --
a determined adversary with a model that emits arbitrary Python is a
different threat model -- but together they make the failure mode a
rejected generation rather than an arbitrary call, which is what the
validation study measures.
"""

from __future__ import annotations

import ast

#: AST nodes the generated program may contain.
ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Return,
    ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Call, ast.keyword, ast.Attribute,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice, ast.Subscript,
    ast.IfExp, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    # comprehensions: composing a per-hazard sum requires iterating the
    # hazards, and a comprehension is the bounded, side-effect-free way to
    # do it. Statement-level loops stay out.
    ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.comprehension,
    ast.Starred,
)

#: Attribute access is allowed on numpy itself ...
ATTRIBUTE_ROOTS = frozenset({"np"})

#: ... and, on any value, only these array methods. A generated program
#: naturally writes ``mask.astype(float)``; forbidding that outright made
#: the subset unusable and bought nothing, because the risk was never
#: method calls on arrays -- it is reaching a module, a file, or a dunder.
ATTRIBUTE_WHITELIST = frozenset({
    "astype", "reshape", "ravel", "flatten", "copy", "clip", "sum", "mean",
    "min", "max", "abs", "round", "any", "all", "shape", "size", "ndim",
    "dtype", "T", "tolist", "item", "argmin", "argmax", "std", "cumsum",
})


class UnsafeCode(Exception):
    """Generated code left the approved subset."""


def check_code(source: str, allowed_names: set[str]) -> ast.Module:
    """Parse and statically verify ``source``; raise :class:`UnsafeCode`.

    ``allowed_names`` is the toolset plus whatever the caller binds. A
    name that is neither allowed nor assigned by the program itself is a
    rejection: that is what stops generated code from reaching for
    ``open`` or ``__import__`` by simply mentioning them.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCode(f"syntax error: {exc}") from exc

    assigned: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise UnsafeCode(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCode(f"private attribute {node.attr!r}")
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            on_numpy = isinstance(root, ast.Name) and root.id in ATTRIBUTE_ROOTS
            if not on_numpy and node.attr not in ATTRIBUTE_WHITELIST:
                raise UnsafeCode(
                    f"attribute {node.attr!r} is not an allowed array method")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise UnsafeCode(f"dunder name {node.id!r}")
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names and node.id not in assigned:
                raise UnsafeCode(f"unknown name {node.id!r}")
    return tree


def _numpy_only_import(name, *args, **kwargs):
    """The one builtin the sandbox keeps, narrowed to numpy.

    numpy imports its own submodules lazily -- ``arr.clip(...)`` can
    trigger one -- so an empty ``__builtins__`` makes ordinary array
    operations fail with a confusing KeyError. Rather than restore
    builtins wholesale, imports are permitted for numpy and refused for
    everything else. Generated code still cannot invoke this: the static
    pass rejects both ``import`` statements and dunder names.
    """
    root = name.split(".")[0]
    if root != "numpy":
        raise UnsafeCode(f"import of {name!r} is not allowed")
    return __import__(name, *args, **kwargs)


def run(source: str, namespace: dict, *, outputs: tuple[str, ...]) -> dict:
    """Statically check, then execute, then collect ``outputs``.

    The execution namespace carries the toolset, numpy, and a guarded
    ``__import__`` -- nothing else. The check happens before the exec, so
    rejected code never runs.
    """
    tree = check_code(source, set(namespace))
    env = dict(namespace)
    env["__builtins__"] = {"__import__": _numpy_only_import}
    try:
        exec(compile(tree, filename="<agent>", mode="exec"), env)  # noqa: S102
    except Exception as exc:
        raise UnsafeCode(f"generated code raised {type(exc).__name__}: {exc}") from exc

    missing = [name for name in outputs if name not in env]
    if missing:
        raise UnsafeCode(f"generated code did not define {missing}")
    return {name: env[name] for name in outputs}
