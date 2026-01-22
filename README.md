# Calcinator — Handwritten Calculator (System Contract, v1)

## 1. Project Goal
This project builds a handwritten calculator that converts a single atomic input—
a grayscale image containing a handwritten arithmetic expression—into a single
numeric result or an explicit error.

The system prioritizes correctness over coverage and refuses to compute when uncertain.

## 2. System Overview
The system is composed of two independent subsystems:
1. A perception system that recognizes handwritten symbols probabilistically.
2. A symbolic system that parses and evaluates expressions deterministically.

These systems are strictly separated and communicate only through a fixed symbol
interface. No subsystem is allowed to bypass this boundary.

## 3. Role of Machine Learning
Machine learning is used only for recognizing handwritten digits and operators
from image segments.

ML outputs probabilities, not decisions.

Machine learning is never used for arithmetic, operator precedence,
expression parsing, or error handling.

## 4. Rule-Based Logic
All rule-based components are deterministic: the same input symbols always
produce the same output or the same error, including:
- Grouping digits into numbers
- Operator precedence
- Expression validation
- Arithmetic computation

## 5. Boundary of Uncertainty
Uncertainty exists only during symbol recognition.

If any recognized symbol has confidence below a globally defined confidence threshold,
the system rejects the input and returns an explicit error without attempting
any parsing or arithmetic.

Once symbols are accepted, all further computation is exact.

## 6. Error Philosophy
The system does not guess.
Errors are explicit and categorized, including:
- Low-confidence recognition
- Invalid expressions
- Division by zero
The system never silently corrects, guesses, or auto-fixes invalid input.

## 7. Scope (Version 1)
Supported:
- Digits 0–9
- Operators + - × ÷
- Multi-digit numbers
- Expressions are evaluated using standard arithmetic operator precedence.

Not Supported:
- Parentheses
- Decimals
- Negative numbers
- Scientific notation

## 8. System Boundaries (Non-Negotiable)

- The perception system:
  - Accepts images only
  - Outputs symbols with confidence scores
  - Never performs parsing or arithmetic

- The symbolic system:
  - Accepts symbols only
  - Performs parsing and arithmetic deterministically
  - Never accesses images or probabilities

