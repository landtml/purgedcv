"""Pytest rootdir marker.

Its mere presence puts the project root on ``sys.path`` (pytest prepend import
mode), so ``import cpcv`` resolves from ``tests/`` without an installed package.
"""
