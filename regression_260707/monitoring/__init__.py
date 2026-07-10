"""Standalone, read-only monitoring UI for the MFT optimization campaign."""

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
