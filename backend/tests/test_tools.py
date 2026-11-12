import os
import pytest
from backend.app.tools.calculator import CalculatorTool
from backend.app.tools.search import SearchTool
from backend.app.tools.data_analysis import DataAnalysisTool
from backend.app.tools.file_gen import FileGenTool
from backend.app.tools.email_tool import EmailTool
from backend.app.tools.registry import tool_registry

def test_calculator_tool():
    res = CalculatorTool.execute("12 * (4 + 5) / 3")
    assert res.success is True
    assert res.result == 36.0

    res_math = CalculatorTool.execute("sqrt(144) + 10")
    assert res_math.success is True
    assert res_math.result == 22.0

    # Tests for min, max, hypot, sum (both individual arguments and list arguments)
    res_min = CalculatorTool.execute("min(12, 45, 8, 99)")
    assert res_min.success is True
    assert res_min.result == 8

    res_max = CalculatorTool.execute("max of 12, 45, 8, 99")
    assert res_max.success is True
    assert res_max.result == 99

    res_hypot = CalculatorTool.execute("hypot of 3 and 4")
    assert res_hypot.success is True
    assert res_hypot.result == 5

    res_sum1 = CalculatorTool.execute("sum(10, 20, 30)")
    assert res_sum1.success is True
    assert res_sum1.result == 60

    res_sum2 = CalculatorTool.execute("sum([10, 20, 30])")
    assert res_sum2.success is True
    assert res_sum2.result == 60

    # Scope & Boundaries: Equation Solving
    res_eq = CalculatorTool.execute("solve 2x + 5 = 15")
    assert res_eq.success is True
    assert "[5]" in str(res_eq.result)

    # Scope & Boundaries: Calculus (Derivatives & Integrals)
    res_diff = CalculatorTool.execute("derivative of x^3 + 2x")
    assert res_diff.success is True
    assert "3*x**2 + 2" in str(res_diff.result)

    res_int = CalculatorTool.execute("integral of 2x")
    assert res_int.success is True
    assert "x**2" in str(res_int.result)

    # Scope & Boundaries: Complex Numbers
    res_complex = CalculatorTool.execute("Abs(3 + 4*I)")
    assert res_complex.success is True
    assert res_complex.result == 5

    res_invalid = CalculatorTool.execute("__import__('os').system('ls')")
    assert res_invalid.success is False

def test_search_tool():
    res = SearchTool.execute("Artificial Intelligence Agents", max_results=2)
    assert res.success is True
    assert res.result is not None

def test_data_analysis_tool():
    # Test summary calculation on default demo dataset
    res = DataAnalysisTool.execute(action="summary")
    assert res.success is True
    assert "shape" in res.result
    assert "statistics" in res.result

    # Test plot generation
    res_plot = DataAnalysisTool.execute(action="plot", chart_type="line", x_col="Month", y_col="Revenue")
    assert res_plot.success is True
    assert len(res_plot.artifacts) > 0
    assert os.path.exists(res_plot.artifacts[0])

def test_file_gen_tool():
    res = FileGenTool.execute(content="# Project Report\nThis is a generated report.", file_type="markdown")
    assert res.success is True
    assert len(res_artifacts := res.artifacts) > 0
    assert os.path.exists(res_artifacts[0])

def test_email_tool():
    res = EmailTool.execute(recipient="user@example.com", subject="Test", body="Hello from agentic AI")
    assert res.success is True

@pytest.mark.asyncio
async def test_tool_registry():
    res = await tool_registry.aexecute("calculator", expression="25 * 4")
    assert res.success is True
    assert res.result == 100
