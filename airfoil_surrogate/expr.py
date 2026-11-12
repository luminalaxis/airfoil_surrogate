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
    ast.Pow: lambda a, b: _pow(a, b),
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


MAX_INT_EXPONENT = 64  # 정수 거듭제곱은 이 지수까지만 정수로 계산 (2**10**10 같은 수식이 멈추지 않도록)


def evaluate(expression: str, variables: dict) -> float:
    if not isinstance(expression, str):
        raise ExpressionError(f"수식은 문자열이어야 합니다: {expression!r}")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"수식 문법 오류: {expression!r}") from exc
    try:
        return _eval(tree.body, variables, expression)
    except ExpressionError:
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        # 0으로 나누기, sqrt(음수), 넘침, 문자열 변수에 산술 등
        shown = {k: v for k, v in variables.items() if k in expression}
        raise ExpressionError(f"수식 계산 오류: {expression!r} ({type(exc).__name__}: {exc}; 변수 {shown})") from None


def _pow(a, b):
    if isinstance(a, int) and isinstance(b, int) and abs(b) > MAX_INT_EXPONENT:
        return float(a) ** b  # 너무 크면 OverflowError
    return operator.pow(a, b)


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
