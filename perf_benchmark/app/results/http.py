"""HTTP routes for read-only benchmark results."""

from aiohttp import web
import orjson
from .service import SERVICE, ResultsService

@web.middleware
async def errors(request, handler):
    try:
        response = await handler(request)
    except ValueError as exc:
        reason = str(exc)
        if reason in {"detail_expired", "detail_unavailable", "detail_source_changed"}:
            response = web.json_response({"availability": "unavailable", "reason": reason},
                                     status={"detail_expired": 410, "detail_unavailable": 404, "detail_source_changed": 409}[reason])
        else:
            response = web.Response(status=400, text=reason)
    except web.HTTPException as exc:
        response = web.Response(status=exc.status, text=exc.text, headers=exc.headers)
    if request.match_info.route.name != "workbench_asset" or response.status not in {200, 304}:
        response.headers["Cache-Control"] = "no-store"
    return response


async def endpoint(request):
    service = request.app[SERVICE]
    query = dict(request.query)
    if "run_id" not in request.match_info:
        data = await service.read(lambda: service.catalog.runs(query))
    else:
        operation = request.match_info.get("operation", "run")
        if operation not in {"run", "trials", "stages", "metrics", "series", "requests", "turn", "attempts", "groups", "targets", "timeline", "statistics", "windows", "evidence"}:
            raise web.HTTPNotFound()
        data = await service.read(lambda: service.query(request.match_info["run_id"], operation, query))
        if operation == 'timeline':
            from .observation import enrich_timeline
            await enrich_timeline(request, data)
            data["summary"] = [{key: value for key, value in data.items()
                                if key not in {"items", "outputs", "coverage", "summary"}}]
    return web.Response(body=orjson.dumps(data), content_type="application/json")


async def comparison_endpoint(request):
    from perf_benchmark.app.control import same_origin
    from perf_benchmark.plugins.metrics.standard import StandardMetrics, parse_settings
    same_origin(request)
    payload = await request.json()
    if not isinstance(payload, dict) or set(payload) != {"baseline", "candidate", "specification"}:
        raise ValueError("comparison requires baseline, candidate and specification")
    service = request.app[SERVICE]
    def compare():
        return StandardMetrics(parse_settings({})).compare(service.catalog.reader(payload["baseline"]),
            service.catalog.reader(payload["candidate"]), payload["specification"])
    result = await service.read(compare)
    return web.Response(body=orjson.dumps(result), content_type="application/json")


async def health(request):
    return web.json_response({"status": "ready"})


def create_app(root, *, config_root=None, prometheus_url="http://127.0.0.1:9090", loki_url="http://127.0.0.1:3100", tempo_url="http://127.0.0.1:3200", grafana_url=None):
    from .observation import add_routes
    from .web import add_routes as add_web

    app = web.Application(middlewares=[errors], client_max_size=1024)
    app[SERVICE] = ResultsService(root)
    add_web(app, control_enabled=config_root is not None, grafana_url=grafana_url)
    if config_root is not None:
        from perf_benchmark.app.control import add_routes as add_control
        add_control(app, config_root, root)
    app.router.add_post("/api/compare", comparison_endpoint)
    app.router.add_get("/health", health)
    app.router.add_get("/api/runs", endpoint)
    app.router.add_get("/api/runs/{run_id}", endpoint)
    add_routes(app, prometheus_url, loki_url, tempo_url)
    app.router.add_get("/api/runs/{run_id}/{operation}", endpoint)
    return app


def serve(root, host="127.0.0.1", port=8088, **backends):
    web.run_app(create_app(root, **backends), host=host, port=port, access_log=None)
