"""CLI."""

from __future__ import annotations

import argparse

import uvicorn

from citeeval.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="citeeval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    settings = Settings()
    if args.cmd == "serve":
        uvicorn.run(
            "citeeval.api:app",
            host=args.host or settings.host,
            port=args.port or settings.port,
        )


if __name__ == "__main__":
    main()
