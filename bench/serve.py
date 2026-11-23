import argparse
import asyncio
from pathlib import Path
import signal

import yaml

from .matrix import Backend


async def serve(args):
    config = yaml.safe_load(Path(args.config).read_text())
    backend = Backend(config, args.logs)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    try:
        await backend.start(args.group, args.tag)
        print("BACKEND_READY", flush=True)
        await stopped.wait()
    finally:
        await backend.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/engine.yaml")
    parser.add_argument("--group", default="fcfs", choices=["fcfs", "native"])
    parser.add_argument("--logs", default="artifacts/development")
    parser.add_argument("--tag", default="initial")
    asyncio.run(serve(parser.parse_args()))


if __name__ == "__main__":
    main()
