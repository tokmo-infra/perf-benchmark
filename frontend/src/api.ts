import { QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef } from "react";

export type Row = Record<string, unknown>;
export interface Document {
  items?: Row[];
  total?: number;
  page?: number;
  state?: string;
  errors?: Row[];
  duplicate_run_ids?: string[];
  windows?: Row[];
  summary?: Row[];
  metrics?: Row[];
  timings?: Row[];
  evidence?: Row;
  trials?: Row[];
  capacity?: Row[];
}
export interface Settings {
  control_enabled: boolean;
  grafana_url: string | null;
}
export const text = (value: unknown): string =>
  value == null
    ? "不可用"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
export const numeric = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
// Plotly date-axis relayout strings are UTC, without an explicit timezone.
export const plotTime = (value: unknown): number => {
  if (typeof value === "number") return value;
  const timestamp = String(value).replace(" ", "T");
  return Date.parse(
    /[zZ]$|[+-]\d{2}:\d{2}$/.test(timestamp)
      ? timestamp
      : timestamp + (timestamp.length === 10 ? "T00:00:00Z" : "Z"),
  );
};
export const terminal = (status: unknown) =>
  ["succeeded", "failed", "cancelled"].includes(String(status));

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public reason?: string,
  ) {
    super(message);
  }
}

// One browser tab shares four slots. Aborted queued work never reaches the server.
let active = 0;
const waiting: (() => void)[] = [];
async function slot<T>(
  signal: AbortSignal | undefined,
  operation: () => Promise<T>,
): Promise<T> {
  await new Promise<void>((resolve, reject) => {
    const enter = () => {
      signal?.removeEventListener("abort", abort);
      active++;
      resolve();
    };
    const abort = () => {
      const i = waiting.indexOf(enter);
      if (i >= 0) waiting.splice(i, 1);
      reject(new DOMException("Aborted", "AbortError"));
    };
    if (signal?.aborted) return abort();
    if (active < 4) enter();
    else {
      waiting.push(enter);
      signal?.addEventListener("abort", abort, { once: true });
    }
  });
  try {
    return await operation();
  } finally {
    active--;
    waiting.shift()?.();
  }
}

function pause(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const done = () => {
      signal?.removeEventListener("abort", abort);
      resolve();
    };
    const timer = setTimeout(done, ms);
    const abort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
  });
}

export async function api<T>(
  path: string,
  signal?: AbortSignal,
  body?: unknown,
): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await slot(signal, async () => {
        const url = new URL(
          path.replace(/^\//, ""),
          new URL("./", window.location.href),
        );
        const response = await fetch(url, {
          signal,
          cache: "no-store",
          ...(body === undefined
            ? {}
            : {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              }),
        });
        if (!response.ok) {
          const raw = await response.text();
          let reason: string | undefined;
          try {
            reason = JSON.parse(raw).reason;
          } catch {
            /* HTTP text errors retain their original message. */
          }
          const error = new ApiError(
            response.status,
            reason || raw || `HTTP ${response.status}`,
            reason,
          );
          if (response.status === 503 && body === undefined && attempt === 0) {
            const after = response.headers.get("Retry-After");
            const seconds =
              after && /^\d+$/.test(after)
                ? Number(after)
                : after
                  ? Math.max(0, (Date.parse(after) - Date.now()) / 1000)
                  : 1;
            await pause(
              (Number.isFinite(seconds) ? seconds : 1) * 1000,
              signal,
            );
          }
          throw error;
        }
        return (await response.json()) as T;
      });
    } catch (error) {
      if (
        body !== undefined ||
        attempt > 0 ||
        signal?.aborted ||
        (error instanceof ApiError && error.status !== 503)
      )
        throw error;
      if (!(error instanceof ApiError)) await pause(500, signal);
    }
  }
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
      gcTime: 30_000,
      retry: false,
      refetchOnWindowFocus: false,
    },
    mutations: { retry: false },
  },
});

function detailScope(path: unknown) {
  return typeof path === "string"
    ? path.match(
        /^(\/api\/runs\/[^/]+)\/(requests|turn|attempts|server-evidence)(\?|$)/,
      )?.[1]
    : undefined;
}

export function useRead(path: string | null, enabled = true) {
  const client = useQueryClient();
  const scope = detailScope(path);
  const markerKey = ["detail-unavailable", scope];
  const marker = useQuery<ApiError | null>({
    queryKey: markerKey,
    queryFn: () => null,
    initialData: null,
    enabled: false,
  });
  const blocked = scope ? marker.data : null;
  const result = useQuery({
    queryKey: ["api", path],
    queryFn: async ({ signal }) => {
      try {
        return await api<Document>(path!, signal);
      } catch (error) {
        if (
          scope &&
          error instanceof ApiError &&
          [404, 409, 410].includes(error.status)
        ) {
          // Identity/expiry failure revokes every detail of this Run, including
          // in-flight reads and cached Turn/Attempt pages. Summary stays readable.
          client.setQueryData(markerKey, error);
          client.removeQueries({
            predicate: (q) =>
              q.queryKey[0] === "api" && detailScope(q.queryKey[1]) === scope,
          });
        }
        throw error;
      }
    },
    enabled: !!path && enabled && !blocked,
  });
  return {
    ...result,
    error: blocked || result.error,
    data: blocked ? undefined : result.data,
    refetch: () => {
      if (scope) client.setQueryData(markerKey, null);
      return result.refetch(); // Explicit read is the only way to resume details.
    },
  };
}

export function useRefresh(enabled: boolean, scope: string, visible = true) {
  const client = useQueryClient();
  const batch = useRef<Promise<void> | null>(null);
  const refresh = useCallback(() => {
    if (!batch.current) {
      batch.current = client
        .refetchQueries(
          {
            type: "active",
            predicate: (q) =>
              q.queryKey[0] === "api" &&
              !(
                q.state.error instanceof ApiError &&
                [400, 404, 409, 410].includes(q.state.error.status)
              ),
          },
          { cancelRefetch: false },
        )
        .finally(() => {
          batch.current = null;
        });
    }
    return batch.current;
  }, [client]);
  useEffect(() => {
    if (!visible) return;
    let closed = false;
    let running = false;
    let terminalPending = !enabled;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      if (closed || running) return;
      running = true;
      if (!document.hidden) await refresh();
      running = false;
      if (!closed && enabled) timer = setTimeout(tick, 5000);
    };
    const finalRefresh = async () => {
      if (closed || running) return;
      running = true;
      // Finish an older batch first, then read every visible query after the
      // terminal Run was observed. An earlier partial batch cannot substitute.
      await batch.current;
      if (!closed && !document.hidden) {
        await refresh();
        terminalPending = false;
      }
      running = false;
    };
    if (enabled) timer = setTimeout(tick, 5000);
    else void finalRefresh();
    const visibilityChanged = () => {
      if (!document.hidden && enabled) {
        clearTimeout(timer);
        void tick();
      } else if (!document.hidden && terminalPending) void finalRefresh();
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      closed = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [enabled, scope, visible, refresh]);
  return refresh;
}

export function usePageCache(
  scope: string,
  kind: "requests" | "attempts",
  currentPath: string,
) {
  const client = useQueryClient();
  useEffect(() => {
    const entries = client
      .getQueryCache()
      .findAll({
        predicate: (q) =>
          q.queryKey[0] === "api" &&
          String(q.queryKey[1]).startsWith(scope + "/" + kind + "?"),
      })
      .sort((a, b) => b.state.dataUpdatedAt - a.state.dataUpdatedAt);
    const keep = kind === "requests" ? 4 : 2;
    const old = entries.filter((q) => q.queryKey[1] !== currentPath);
    for (const q of old.slice(keep - 1))
      if (!q.isActive())
        client.removeQueries({ queryKey: q.queryKey, exact: true });
  }, [client, scope, kind, currentPath]);
}

export function path(
  runId: string,
  operation: string,
  values: Record<string, string | number | undefined> = {},
) {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(values))
    if (v !== undefined && v !== "") query.set(k, String(v));
  return `/api/runs/${encodeURIComponent(runId)}/${operation}?${query}`;
}

export function grafanaLink(
  base: string | null | undefined,
  relative: unknown,
  timezone?: string,
  params?: URLSearchParams,
) {
  if (!base || typeof relative !== "string") return undefined;
  let route: URL;
  try {
    route = new URL(relative, "http://workbench.invalid");
  } catch {
    return undefined;
  }
  if (
    route.origin !== "http://workbench.invalid" ||
    !["/d/perf-run", "/d/perf-runs", "/explore"].includes(route.pathname)
  )
    return undefined;
  for (const [key, value] of params ?? [])
    if (["from", "to"].includes(key)) route.searchParams.set(key, value);
    else if (key === "trial") route.searchParams.set("var-trial", value);
  if (route.pathname === "/explore") {
    try {
      const panes = JSON.parse(route.searchParams.get("panes") ?? "null");
      if (!panes?.workbench) return undefined;
      for (const key of ["from", "to"])
        if (params?.has(key))
          panes.workbench.range = {
            ...panes.workbench.range,
            [key]: params.get(key),
          };
      route.searchParams.set("panes", JSON.stringify(panes));
    } catch {
      return undefined;
    }
  }
  if (timezone) route.searchParams.set("timezone", timezone);
  return base.replace(/\/$/, "") + route.pathname + route.search;
}
