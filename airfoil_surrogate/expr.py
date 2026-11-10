"""파생 변수용 안전한 수식 계산기.

설정 파일의 ``[derived]`` 항목(예: ``flow_x = "cos(radians(aoa))"``)을
``eval`` 없이 계산한다. 숫자, 변수, 사칙연산, math 함수만 허용한다.
"""

from __future__ import annotations

import ast
import math
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCS = {
    name: getattr(math, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sqrt", "exp", "log", "log10", "radians", "degrees", "hypot",
    )
}
_FUNCS.update({"abs": abs, "min": min, "max": max, "round": round})
_CONSTS = {"pi": math.pi, "e": math.e}


class ExpressionError(ValueError):
    pass


def evaluate(expression: str, variables: dict) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"수식 문법 오류: {expression!r}") from exc
    return _eval(tree.body, variables, expression)


def _eval(node, variables, src):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ExpressionError(f"정의되지 않은 변수 {node.id!r} (수식: {src!r})")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](
            _eval(node.left, variables, src), _eval(node.right, variables, src)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand, variables, src))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        args = [_eval(a, variables, src) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ExpressionError(f"허용되지 않는 수식 요소 (수식: {src!r})")
