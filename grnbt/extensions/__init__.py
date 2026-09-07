"""Optional extensions and improvements that are not part of the paper baseline.

These modules are deliberately isolated from the core ``grnbt/``
package — they are **not** re-exported by ``grnbt.__init__`` and are
not used in the official paper experiments. They exist to showcase
production-grade improvements (histogram binning, sparse features,
column subsampling) without polluting the faithful reproduction.

Anything in this sub-package must keep the paper's mathematical
contract intact: leaf weights, gain expressions, and the ``λ_k``
recipes must be byte-compatible with the corresponding core
implementation.
"""
