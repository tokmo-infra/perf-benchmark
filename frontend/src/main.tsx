import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import {
  HashRouter,
  Link,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import {
  QueryClientProvider,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api, queryClient, type Settings } from "./api";
import { ErrorMessage } from "./components";
import Runs from "./pages/Runs";
import Run from "./pages/Run";
import Requests from "./pages/Requests";
import Compare from "./pages/Compare";
import Control from "./pages/Control";
import "./styles.css";

function App() {
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: ({ signal }) => api<Settings>("/api/workbench", signal),
  });
  const location = useLocation();
  const client = useQueryClient();
  useEffect(() => {
    const run = location.pathname.match(/^\/runs\/([^/]+)/)?.[1];
    void client.cancelQueries({
      predicate: (q) => q.queryKey[0] === "api" && !q.isActive(),
    });
    client.removeQueries({
      predicate: (q) =>
        q.queryKey[0] === "api" &&
        !q.isActive() &&
        /\/(requests|turn|attempts)\?/.test(String(q.queryKey[1])) &&
        (!run || !String(q.queryKey[1]).startsWith("/api/runs/" + run + "/")),
    });
  }, [client, location.pathname]);
  return (
    <div className="shell">
      <aside className="sidebar">
        <Link className="brand" to="/runs">
          <span className="brand-mark">p/</span>
          <span>
            perf-benchmark<small>压测工作台</small>
          </span>
        </Link>
        <div className="nav-label">测试工作台</div>
        <nav>
          <NavLink to="/runs">
            测试记录 <span>01</span>
          </NavLink>
          <NavLink to="/compare">
            结果对比 <span>02</span>
          </NavLink>
          <NavLink to="/control">
            启动测试 <span>03</span>
          </NavLink>
        </nav>
        <div className="sidebar-foot">
          <span className="dot" /> 结果自动保存
          <p>测试结果与运行监控</p>
          <small>
            汇总结果长期保留
            <br />
            请求明细保留七天
          </small>
        </div>
      </aside>
      <main>
        <div className="topbar">
          <span>压测工作台</span>
          <span>查看测试结果，比较性能，定位问题</span>
        </div>
        <ErrorMessage error={settings.error} />
        <Routes>
          <Route path="/" element={<Navigate to="/runs" replace />} />
          <Route path="/runs" element={<Runs />} />
          <Route
            path="/runs/:runId"
            element={<Run key={location.pathname} settings={settings.data} />}
          />
          <Route
            path="/runs/:runId/requests"
            element={
              <Requests key={location.pathname} settings={settings.data} />
            }
          />
          <Route path="/compare" element={<Compare />} />
          <Route
            path="/control"
            element={<Control settings={settings.data} />}
          />
          <Route
            path="*"
            element={
              <div className="empty">
                页面不存在 · <Link to="/runs">返回测试列表</Link>
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <App />
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
