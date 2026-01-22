# calcinator - A Handwritten Calculator — System Design

## 1. Project Goal
This project builds a handwritten calculator that converts a single grayscale image
containing a handwritten arithmetic expression into a numeric result.

The system prioritizes correctness over coverage and refuses to compute when uncertain.

## 2. System Overview
The system is composed of two independent subsystems:
1. A perception system that recognizes handwritten symbols probabilistically.
2. A symbolic system that parses and evaluates expressions deterministically.

These systems are strictly separated.

## 3. Role of Machine Learning
Machine learning is used only for recognizing handwritten digits and operators
from image segments.

ML outputs probabilities, not decisions.

ML is not used for parsing, arithmetic, or evaluation logic.

## 4. Rule-Based Logic
All symbolic reasoning is rule-based and deterministic, including:
- Grouping digits into numbers
- Operator precedence
- Expression validation
- Arithmetic computation

## 5. Boundary of Uncertainty
Uncertainty exists only during symbol recognition.

If any recognized symbol has confidence below a fixed threshold,
the system rejects the input and requests a redraw.

Once symbols are accepted, all further computation is exact.

## 6. Error Philosophy
The system does not guess.
Errors are explicit and categorized, including:
- Low-confidence recognition
- Invalid expressions
- Division by zero

## 7. Scope (Version 1)
Supported:
- Digits 0–9
- Operators + - × ÷
- Multi-digit numbers
- Standard operator precedence

Not Supported:
- Parentheses
- Decimals
- Negative numbers
- Scientific notation
