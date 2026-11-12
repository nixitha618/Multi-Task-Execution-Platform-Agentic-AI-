import math
import re
from typing import Dict, Any, Optional
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from backend.app.models.schemas import ToolExecutionResult

def smart_sum(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return sum(args[0])
    return sum(args)

def smart_hypot(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return math.hypot(*args[0])
    return math.hypot(*args)

def smart_min(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return min(args[0])
    return min(*args)

def smart_max(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        return max(args[0])
    return max(*args)

class CalculatorTool:
    """Safe mathematical & symbolic calculation tool."""

    ALLOWED_NAMES = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    ALLOWED_NAMES.update({
        "abs": abs,
        "round": round,
        "min": smart_min,
        "max": smart_max,
        "sum": smart_sum,
        "hypot": smart_hypot,
        "pow": pow,
    })

    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

    @classmethod
    def normalize_expression(cls, expr: str) -> str:
        s = expr.strip()
        # If already formatted as a tool command, don't re-wrap or duplicate
        if any(s.startswith(p) for p in ["diff(", "integrate(", "limit(", "factor(", "expand("]):
            return s
        if s.startswith("solve(") and s.endswith(")"):
            return s

        s = re.sub(r"^(?:calculate|what is|compute|eval|evaluate|find|give me)\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*(?:and\s+show\s+.*|show\s+.*|step[- ]by[- ]step.*|please|\?)$", "", s, flags=re.IGNORECASE).strip()
        s = s.rstrip("=? ")

        # Natural language conversions
        s = re.sub(r"(?:hypot|hypotenuse)\s*(?:of)?\s*([\d\.]+)\s*(?:and|,)\s*([\d\.]+)", r"hypot(\1, \2)", s, flags=re.IGNORECASE)
        s = re.sub(r"(?:square\s*root\s*(?:of)?|sqrt\s*(?:of)?)\s*([\d\.\-]+)", r"sqrt(\1)", s, flags=re.IGNORECASE)
        s = re.sub(r"(?:cube\s*root\s*(?:of)?|cbrt\s*(?:of)?)\s*([\d\.\-]+)", r"pow(\1, 1/3)", s, flags=re.IGNORECASE)
        s = re.sub(r"([\d\.]+)\s*(?:to the power of|raised to|to the)\s*([\d\.]+)", r"\1 ** \2", s, flags=re.IGNORECASE)
        s = re.sub(r"([\d\.]+)\s*squared", r"\1 ** 2", s, flags=re.IGNORECASE)
        s = re.sub(r"([\d\.]+)\s*cubed", r"\1 ** 3", s, flags=re.IGNORECASE)
        s = re.sub(r"([\d\.]+)\s*(?:%|percent)\s*of\s*([\d\.]+)", r"(\2 * (\1 / 100))", s, flags=re.IGNORECASE)

        # 2x -> 2*x (implicit multiplication on single variable letters)
        s = re.sub(r"(\d+)\s*([a-zA-Z])\b", r"\1*\2", s)
        s = s.replace("^", "**")

        # Min/Max/Sum natural language conversions
        s = re.sub(r"\b(?:min|minimum)\s+of\s+([\[\(]?[\d\s\,\.\-]+[\]\)]?)", r"min(\1)", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:max|maximum)\s+of\s+([\[\(]?[\d\s\,\.\-]+[\]\)]?)", r"max(\1)", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:sum|summation)\s+of\s+([\[\(]?[\d\s\,\.\-]+[\]\)]?)", r"sum(\1)", s, flags=re.IGNORECASE)

        # Words to operators
        s = re.sub(r"\bplus\b", "+", s, flags=re.IGNORECASE)
        s = re.sub(r"\bminus\b", "-", s, flags=re.IGNORECASE)
        s = re.sub(r"\btimes\b|\bmultiplied\s*by\b", "*", s, flags=re.IGNORECASE)
        s = re.sub(r"\bdivided\s*by\b|\bover\b", "/", s, flags=re.IGNORECASE)
        s = re.sub(r"\bmod(?:ulo)?\b", "%", s, flags=re.IGNORECASE)

        # Calculus natural language
        m_diff = re.search(r"^(?:derivative|diff)\s*(?:of)?\s*(.+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?$", s, flags=re.IGNORECASE)
        if m_diff:
            var = m_diff.group(2) if m_diff.group(2) else "x"
            expr_body = m_diff.group(1).strip()
            return f"diff({expr_body}, {var})"

        m_int_def = re.search(r"^(?:integral|integrate)\s*(?:of)?\s*(.+?)\s+from\s+([-\d\.]+)\s+to\s+([-\d\.]+)$", s, flags=re.IGNORECASE)
        if m_int_def:
            expr_body = m_int_def.group(1).strip()
            return f"integrate({expr_body}, (x, {m_int_def.group(2)}, {m_int_def.group(3)}))"

        m_int = re.search(r"^(?:integral|integrate)\s*(?:of)?\s*(.+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?$", s, flags=re.IGNORECASE)
        if m_int:
            var = m_int.group(2) if m_int.group(2) else "x"
            expr_body = m_int.group(1).strip()
            return f"integrate({expr_body}, {var})"

        m_lim = re.search(r"^limit\s*(?:of)?\s*(.+?)\s+as\s+([a-zA-Z])\s*(?:approaches|->|to)\s*([-\d\.\w]+)$", s, flags=re.IGNORECASE)
        if m_lim:
            expr_body = m_lim.group(1).strip()
            return f"limit({expr_body}, {m_lim.group(2)}, {m_lim.group(3)})"

        m_solve = re.search(r"^solve\s*(?:for\s+[a-zA-Z]\s*:)?\s*(.+)", s, flags=re.IGNORECASE)
        if m_solve:
            body = m_solve.group(1).strip()
            return f"solve({body})"

        m_factor = re.search(r"^factor\s*(?:of)?\s*(.+)", s, flags=re.IGNORECASE)
        if m_factor:
            body = m_factor.group(1).strip()
            return f"factor({body})"

        m_expand = re.search(r"^expand\s*(?:of)?\s*(.+)", s, flags=re.IGNORECASE)
        if m_expand:
            body = m_expand.group(1).strip()
            return f"expand({body})"

        return s.strip()

    @classmethod
    def execute(cls, expression: Optional[str] = None, **kwargs) -> ToolExecutionResult:
        try:
            expr_raw = expression or kwargs.get("expr") or kwargs.get("query") or kwargs.get("math_expression") or ""
            expr = cls.normalize_expression(str(expr_raw))

            # Check if symbolic computation (SymPy) is requested or appropriate
            is_symbolic = (
                "=" in expr
                or any(expr.startswith(p) for p in ["diff(", "integrate(", "limit(", "solve(", "factor(", "expand("])
                or bool(re.search(r"\b[xyztiI]\b", expr))
                or "sqrt(-" in expr
            )

            if is_symbolic:
                try:
                    # Equation solving (e.g. solve 2x + 5 = 15 or 2x + 5 = 15)
                    if "=" in expr:
                        inner = expr
                        while inner.startswith("solve(") and inner.endswith(")"):
                            inner = inner[6:-1].strip()
                        while inner.startswith("(") and inner.endswith(")"):
                            inner = inner[1:-1].strip()
                        lhs_str, rhs_str = inner.split("=", 1)
                        lhs = parse_expr(lhs_str.strip(), transformations=cls.TRANSFORMATIONS)
                        rhs = parse_expr(rhs_str.strip(), transformations=cls.TRANSFORMATIONS)
                        sol = sp.solve(sp.Eq(lhs, rhs))
                        return ToolExecutionResult(
                            tool_name="calculator",
                            success=True,
                            result=str(sol),
                            error=None
                        )

                    if expr.startswith("factor(") and expr.endswith(")"):
                        inner = expr[7:-1].strip()
                        parsed = parse_expr(inner, transformations=cls.TRANSFORMATIONS)
                        res = sp.factor(parsed)
                    elif expr.startswith("expand(") and expr.endswith(")"):
                        inner = expr[7:-1].strip()
                        parsed = parse_expr(inner, transformations=cls.TRANSFORMATIONS)
                        res = sp.expand(parsed)
                    else:
                        parsed = parse_expr(expr, transformations=cls.TRANSFORMATIONS)
                        if hasattr(parsed, "doit"):
                            parsed = parsed.doit()
                        res = sp.simplify(parsed)

                    if hasattr(res, "is_number") and res.is_number:
                        if getattr(res, "is_integer", False):
                            res_val = int(res)
                        elif getattr(res, "is_real", False):
                            res_val = round(float(res), 6)
                            if res_val.is_integer():
                                res_val = int(res_val)
                        else:
                            res_val = str(res)
                    else:
                        res_val = str(res)

                    return ToolExecutionResult(
                        tool_name="calculator",
                        success=True,
                        result=res_val,
                        error=None
                    )
                except Exception:
                    # If symbolic evaluation fails, fall through to numerical sandbox
                    pass

            # Standard Numerical Evaluation
            # Safety check: allow numbers, operators, parens, brackets, commas, dots, approved letters, equals
            if not re.match(r"^[\d\s\+\-\*\/\%\(\)\[\]\.\,\^a-zA-Z_=\:]+$", expr):
                return ToolExecutionResult(
                    tool_name="calculator",
                    success=False,
                    result=None,
                    error="Expression contains invalid or unsafe characters."
                )

            compiled = compile(expr, "<string>", "eval")
            for name in compiled.co_names:
                if name not in cls.ALLOWED_NAMES:
                    return ToolExecutionResult(
                        tool_name="calculator",
                        success=False,
                        result=None,
                        error=f"Function or variable '{name}' is not allowed for security reasons."
                    )

            result = eval(compiled, {"__builtins__": {}}, cls.ALLOWED_NAMES)
            if isinstance(result, float):
                result = round(result, 6)
                if result.is_integer():
                    result = int(result)

            return ToolExecutionResult(
                tool_name="calculator",
                success=True,
                result=result,
                error=None
            )
        except Exception as e:
            # If standard eval failed with ValueError (e.g. math domain error on negative sqrt), try SymPy as fallback
            try:
                parsed = parse_expr(expr, transformations=cls.TRANSFORMATIONS)
                res = sp.simplify(parsed)
                return ToolExecutionResult(
                    tool_name="calculator",
                    success=True,
                    result=str(res),
                    error=None
                )
            except Exception:
                pass

            return ToolExecutionResult(
                tool_name="calculator",
                success=False,
                result=None,
                error=f"Evaluation error: {str(e)}"
            )
