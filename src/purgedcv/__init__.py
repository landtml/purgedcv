"""purgedcv — Combinatorial Purged Cross-Validation for financial time series.

A scikit-learn-compatible implementation of Combinatorial Purged
Cross-Validation with embargo, following Marcos Lopez de Prado's
*Advances in Financial Machine Learning* (AFML), Chapter 7.

See the project README and PROOF.md for the correctness guarantee, an
independent empirical verification against ~98 years of real market data,
and usage examples.
"""

from __future__ import annotations

from ._splitter import CombinatorialPurgedCV, CPCVPaths, make_t1

__all__ = ["CombinatorialPurgedCV", "CPCVPaths", "make_t1"]
__version__ = "0.1.0"
