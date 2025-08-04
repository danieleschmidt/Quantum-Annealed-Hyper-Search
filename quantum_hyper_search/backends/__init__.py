"""Quantum computing backends for different hardware and simulators."""

from .backend_factory import get_backend

__all__ = ["get_backend"]