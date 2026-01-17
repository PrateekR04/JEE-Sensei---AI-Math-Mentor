"""
Calculator Tool
Production-grade symbolic and numeric math engine using SymPy
"""

import sympy as sp
from sympy import symbols, solve, simplify, diff, integrate
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)
from typing import List, Any


# Enable safe math parsing
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor
)


class Calculator:
    """
    Mathematical calculator using SymPy.
    Supports equation solving, evaluation, differentiation, integration and verification.
    """

    # -------------------------
    # Internal Safe Parser
    # -------------------------
    @staticmethod
    def _safe_parse(expr: str):
        """
        Safely parse math expression into SymPy expression.
        """
        try:
            return parse_expr(expr, transformations=TRANSFORMATIONS, evaluate=True)
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: '{expr}' → {str(e)}")

    # -------------------------
    # Equation Solver
    # -------------------------
    @staticmethod
    def solve_equation(equation: str, variable: str = "x") -> List[Any]:
        """
        Solve an equation for a variable.
        Example: "2*x + 3 = 5"
        """

        equation = equation.strip()

        # Must contain exactly one "="
        if equation.count("=") != 1:
            raise ValueError("Equation must contain exactly one '=' sign")

        left, right = equation.split("=")
        left = left.strip()
        right = right.strip()

        left_expr = Calculator._safe_parse(left)
        right_expr = Calculator._safe_parse(right)

        expr = left_expr - right_expr

        var = symbols(variable)

        try:
            solutions = solve(expr, var)
        except Exception as e:
            raise ValueError(f"Could not solve equation: {str(e)}")

        # Convert to numeric if possible, else symbolic
        results = []
        for sol in solutions:
            try:
                results.append(float(sol))
            except:
                results.append(sol)

        return results

    # -------------------------
    # Expression Evaluation
    # -------------------------
    @staticmethod
    def evaluate(expression: str, **variables) -> float:
        """
        Evaluate a mathematical expression numerically.
        """

        expr = Calculator._safe_parse(expression)

        if variables:
            subs = {symbols(k): v for k, v in variables.items()}
            expr = expr.subs(subs)

        try:
            return float(expr.evalf())
        except Exception:
            raise ValueError("Expression could not be evaluated numerically")

    # -------------------------
    # Differentiation
    # -------------------------
    @staticmethod
    def differentiate(expression: str, variable: str = "x") -> str:
        """
        Compute derivative of an expression.
        """

        expr = Calculator._safe_parse(expression)
        var = symbols(variable)

        try:
            derivative = diff(expr, var)
            return str(derivative)
        except Exception as e:
            raise ValueError(f"Error computing derivative: {str(e)}")

    # -------------------------
    # Integration
    # -------------------------
    @staticmethod
    def integrate_expr(expression: str, variable: str = "x") -> str:
        """
        Compute indefinite integral of an expression.
        """

        expr = Calculator._safe_parse(expression)
        var = symbols(variable)

        try:
            integral = integrate(expr, var)
            return str(integral)
        except Exception as e:
            raise ValueError(f"Error computing integral: {str(e)}")

    # -------------------------
    # Simplification
    # -------------------------
    @staticmethod
    def simplify_expr(expression: str) -> str:
        """
        Simplify a mathematical expression.
        """

        expr = Calculator._safe_parse(expression)

        try:
            simplified = simplify(expr)
            return str(simplified)
        except Exception as e:
            raise ValueError(f"Error simplifying expression: {str(e)}")

    # -------------------------
    # Solution Verification
    # -------------------------
    @staticmethod
    def verify_solution(equation: str, variable: str, value: float) -> bool:
        """
        Verify if a value is a solution to an equation.
        """

        if equation.count("=") != 1:
            return False

        left, right = equation.split("=")
        left_expr = Calculator._safe_parse(left)
        right_expr = Calculator._safe_parse(right)

        var = symbols(variable)

        try:
            left_val = left_expr.subs(var, value).evalf()
            right_val = right_expr.subs(var, value).evalf()
            return abs(float(left_val) - float(right_val)) < 1e-10
        except Exception:
            return False

    # -------------------------
    # System of Equations Solver
    # -------------------------
    @staticmethod
    def solve_system(equations: List[str], variables: List[str]) -> List[dict]:
        """
        Solve a system of equations.
        
        Args:
            equations: List of equation strings like ["2*x + y = 5", "x - y = 1"]
            variables: List of variable names like ["x", "y"]
            
        Returns:
            List of solution dicts like [{"x": 2.0, "y": 1.0}]
        """
        from sympy import Eq
        
        # Create symbol objects
        sym_dict = {v: symbols(v) for v in variables}
        sym_list = [sym_dict[v] for v in variables]
        
        # Parse equations
        eq_list = []
        for eq_str in equations:
            eq_str = eq_str.strip()
            if '=' not in eq_str:
                raise ValueError(f"Not a valid equation: {eq_str}")
            
            left, right = eq_str.split('=')
            left_expr = Calculator._safe_parse(left.strip())
            right_expr = Calculator._safe_parse(right.strip())
            eq_list.append(Eq(left_expr, right_expr))
        
        try:
            solutions = solve(eq_list, sym_list, dict=True)
        except Exception as e:
            raise ValueError(f"Could not solve system: {str(e)}")
        
        # Format results
        results = []
        for sol in solutions:
            sol_dict = {}
            for var_sym, value in sol.items():
                var_name = str(var_sym)
                try:
                    sol_dict[var_name] = float(value)
                except:
                    sol_dict[var_name] = str(value)
            results.append(sol_dict)
        
        return results

    # -------------------------
    # Limit Computation
    # -------------------------
    @staticmethod
    def compute_limit(expression: str, variable: str = "x", point: str = "0") -> str:
        """
        Compute the limit of an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable approaching the point
            point: The limit point (can be "oo" for infinity)
            
        Returns:
            The limit value as string
        """
        from sympy import limit, oo, sympify
        
        expr = Calculator._safe_parse(expression)
        var = symbols(variable)
        
        # Parse limit point
        if point in ['oo', 'infinity', 'inf']:
            point_val = oo
        elif point in ['-oo', '-infinity', '-inf']:
            point_val = -oo
        else:
            point_val = sympify(point)
        
        try:
            result = limit(expr, var, point_val)
            return str(result)
        except Exception as e:
            raise ValueError(f"Error computing limit: {str(e)}")

    # -------------------------
    # Optimization (Max/Min)
    # -------------------------
    @staticmethod
    def optimize(expression: str, variable: str = "x") -> dict:
        """
        Find critical points and determine maximum/minimum.
        
        Args:
            expression: Mathematical expression to optimize
            variable: Variable to optimize over
            
        Returns:
            Dict with critical_points, maximum, minimum
        """
        expr = Calculator._safe_parse(expression)
        var = symbols(variable)
        
        # Find derivative
        derivative = diff(expr, var)
        
        # Find critical points (where derivative = 0)
        try:
            critical_points = solve(derivative, var)
        except Exception as e:
            raise ValueError(f"Could not find critical points: {str(e)}")
        
        if not critical_points:
            return {
                "critical_points": [],
                "maximum": None,
                "minimum": None,
                "error": "No critical points found"
            }
        
        # Evaluate at critical points
        evaluations = []
        for cp in critical_points:
            try:
                value = float(expr.subs(var, cp))
                cp_float = float(cp)
                evaluations.append({"point": cp_float, "value": value})
            except:
                evaluations.append({"point": str(cp), "value": str(expr.subs(var, cp))})
        
        # Find max and min among numeric evaluations
        numeric_evals = [e for e in evaluations if isinstance(e["value"], (int, float))]
        
        maximum = None
        minimum = None
        
        if numeric_evals:
            max_eval = max(numeric_evals, key=lambda x: x["value"])
            min_eval = min(numeric_evals, key=lambda x: x["value"])
            maximum = {"point": max_eval["point"], "value": max_eval["value"]}
            minimum = {"point": min_eval["point"], "value": min_eval["value"]}
        
        return {
            "critical_points": evaluations,
            "maximum": maximum,
            "minimum": minimum,
            "derivative": str(derivative)
        }


# -------------------------
# CLI Test
# -------------------------
def main():
    print("Calculator Tool Test")
    print("=" * 60)

    print("\nSolve: 2*x + 3 = 5")
    print(Calculator.solve_equation("2*x + 3 = 5", "x"))

    print("\nEvaluate: 2 + 3 * 4")
    print(Calculator.evaluate("2 + 3 * 4"))

    print("\nDifferentiate: x^2 + 3*x")
    print(Calculator.differentiate("x^2 + 3*x"))

    print("\nIntegrate: 2*x + 3")
    print(Calculator.integrate_expr("2*x + 3"))

    print("\nSimplify: (x + 1)^2 - (x^2 + 2*x + 1)")
    print(Calculator.simplify_expr("(x + 1)^2 - (x^2 + 2*x + 1)"))

    print("\nVerify: x=1 is solution of 2*x + 3 = 5")
    print(Calculator.verify_solution("2*x + 3 = 5", "x", 1))


if __name__ == "__main__":
    main()
