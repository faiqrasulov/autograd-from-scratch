# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: what reproduces exactly from the pinned versions and what depends on the linear-algebra build.

## 2026-11-15

- First version: reverse-mode autodiff and a small neural-network library in NumPy, checked op by op against PyTorch; MLP and CNN parity runs.
- README: hyperparameters are fixed in `cli.py`; the search that chose them is not recorded.
