__version__ = "0.1.0"


def main() -> None:
    from .server import main as _main

    _main()
