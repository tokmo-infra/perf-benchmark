"""Static workbench and browser navigation; formal data uses existing APIs."""
from pathlib import Path
import re
from urllib.parse import urlsplit

from aiohttp import web

WORKBENCH = web.AppKey("workbench", dict)
ASSETS = Path(__file__).parent.parent / "workbench"


def public_url(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or any(c.isspace() for c in value)):
        raise ValueError("grafana-url must be an HTTP(S) browser URL without credentials, query or fragment")
    return value.rstrip("/")


async def settings(request):
    return web.json_response(request.app[WORKBENCH])


async def index(request):
    path = ASSETS / "index.html"
    if ASSETS.is_symlink() or path.is_symlink() or not path.is_file():
        raise web.HTTPServiceUnavailable(text="Workbench assets are not built; run npm ci and npm run build in frontend")
    return web.FileResponse(path)


async def asset(request):
    name = request.match_info["name"]
    # Vite emits flat content-hashed files. Never follow symlinks or expose a directory.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{8,}\.(?:js|css|woff2|svg)", name):
        raise web.HTTPNotFound()
    path = ASSETS / "assets" / name
    if ASSETS.is_symlink() or path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


def add_routes(app, *, control_enabled, grafana_url):
    app[WORKBENCH] = {"control_enabled": control_enabled, "grafana_url": public_url(grafana_url)}
    app.router.add_get("/", index)
    app.router.add_get("/assets/{name}", asset, name="workbench_asset")
    app.router.add_get("/api/workbench", settings)
