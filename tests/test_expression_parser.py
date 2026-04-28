"""
Test cases for expression parser (without pytest)

Uses plain assert statements and manual exception checks.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser.expression_parser import parse_and_evaluate, ExpressionError


def assert_raises(func, *args):
    """
    Simple helper to verify an exception is raised.
    """
    try:
        func(*args)
        assert False, "Expected ExpressionError but no exception was raised"
    except ExpressionError:
        pass


# ── Basic operations ─────────────────────────────────────────

def test_addition():
    assert parse_and_evaluate([6, '+', 7]) == 13


def test_subtraction():
    assert parse_and_evaluate([40, '-', 9]) == 31


def test_multiplication():
    assert parse_and_evaluate([8, '×', 3]) == 24


def test_division():
    result = parse_and_evaluate([40, '÷', 9])
    assert abs(result - 4.444) < 0.01


def test_multi_digit_operands():
    assert parse_and_evaluate([12, '+', 3]) == 15
    assert parse_and_evaluate([7, '-', 1]) == 6


# ── Precedence ───────────────────────────────────────────────

def test_multiplication_before_addition():
    assert parse_and_evaluate([2, '+', 3, '×', 4]) == 14


def test_division_before_subtraction():
    assert parse_and_evaluate([10, '-', 6, '÷', 2]) == 7


def test_left_to_right_same_precedence():
    assert parse_and_evaluate([10, '-', 3, '-', 2]) == 5


# ── Single operand ───────────────────────────────────────────

def test_single_number():
    assert parse_and_evaluate([42]) == 42


# ── Validation errors ────────────────────────────────────────

def test_empty_tokens():
    assert_raises(parse_and_evaluate, [])


def test_operator_only():
    assert_raises(parse_and_evaluate, ['+'])


def test_two_operators_in_a_row():
    assert_raises(parse_and_evaluate, [6, '+', '+', 7])


def test_ends_with_operator():
    assert_raises(parse_and_evaluate, [6, '+'])


def test_starts_with_operator():
    assert_raises(parse_and_evaluate, ['+', 6])


def test_unknown_operator():
    assert_raises(parse_and_evaluate, [6, '/', 7])


def test_division_by_zero():
    assert_raises(parse_and_evaluate, [6, '÷', 0])


def test_two_numbers_no_operator():
    assert_raises(parse_and_evaluate, [6, 7])


if __name__ == "__main__":
    # Run all tests manually
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"✓ {name} passed")

    print("\nAll tests passed.")