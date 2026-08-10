"""CutoutML: GPU-accelerated image and video segmentation platform.

The package is intentionally split so that the heavy inference code has no
dependency on the web/queue layer:

* ``cutoutml.core``       - configuration, logging, imaging primitives
* ``cutoutml.models``     - segmentation model adapters + registry
* ``cutoutml.pipelines``  - image / video end-to-end pipelines
* ``cutoutml.datasets``   - synthetic generator + real dataset adapters
* ``cutoutml.training``   - CutoutNet training loop
* ``cutoutml.benchmarks`` - latency/accuracy harness
* ``cutoutml.db``         - SQLAlchemy 2.0 models
* ``cutoutml.storage``    - local/S3 object storage abstraction
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
