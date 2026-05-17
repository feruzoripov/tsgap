# Installation

TSGap requires Python 3.9 or newer and NumPy 1.19 or newer.

## From PyPI

```bash
pip install tsgap
```

## From Source

```bash
git clone https://github.com/feruzoripov/tsgap.git
cd tsgap
pip install -e ".[dev]"
```

The development extra installs the test dependencies:

```bash
pytest tsgap/tests/ -v
```

## Import Check

```python
import numpy as np
from tsgap import simulate_missingness

X = np.random.default_rng(42).standard_normal((100, 4))
X_missing, mask = simulate_missingness(X, "mcar", 0.2, seed=42)

print(X_missing.shape)
print((~mask).mean())
```
