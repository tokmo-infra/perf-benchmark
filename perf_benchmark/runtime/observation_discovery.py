"""Per-observer Alloy file discovery, published only after committed-result readiness."""

from pathlib import Path
from uuid import UUID, uuid4

import orjson


class DiscoveryRegistration:
    def __init__(self, directory: str | None, identity: str | None = None):
        self.directory = Path(directory) if directory is not None else None
        self.identity = str(UUID(identity)) if identity else str(uuid4())

    def publish(self, groups):
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            for kind, targets in groups.items():
                path = self.directory / f"{self.identity}.{kind}.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(orjson.dumps(targets))
                temporary.replace(path)
        except BaseException:
            self.remove()
            raise

    def remove(self):
        if self.directory is not None:
            for path in self.directory.glob(f"{self.identity}.*"):
                if path.suffix in (".json", ".tmp"):
                    path.unlink(missing_ok=True)
