import { useEffect, useState } from "react";

function Frame({ url, title }: { url: string; title: string }) {
  const [state, setState] = useState<"loading" | "slow" | "loaded">("loading");
  useEffect(() => {
    if (state !== "loading") return;
    const timer = setTimeout(() => setState("slow"), 15_000);
    return () => clearTimeout(timer);
  }, [state]);
  return (
    <>
      {state !== "loaded" && (
        <p role="status" className="muted">
          {state === "slow"
            ? "监控页面仍在加载，可稍候或重新加载。"
            : "正在加载运行监控…"}
        </p>
      )}
      <iframe
        className="diagnostic-frame"
        src={url}
        title={title}
        onLoad={() => setState("loaded")}
      />
    </>
  );
}

export default function DiagnosticFrame({
  url,
  title,
  unavailable,
}: {
  url?: string;
  title: string;
  unavailable: string;
}) {
  const [revision, setRevision] = useState(0);
  const [visible, setVisible] = useState(!document.hidden);
  useEffect(() => {
    const changed = () => setVisible(!document.hidden);
    document.addEventListener("visibilitychange", changed);
    return () => document.removeEventListener("visibilitychange", changed);
  }, []);
  if (!url)
    return (
      <p className="empty" role="status">
        {unavailable}
      </p>
    );
  const embedded = new URL(url, window.location.href);
  embedded.searchParams.set("kiosk", "1");
  return (
    <div className="diagnostic-view">
      <div className="toolbar">
        <span className="muted">
          查看
          CPU、内存、GPU、请求链路、日志和性能分析。首次打开请使用管理员提供的
          Grafana 账号在下方登录。
        </span>
        <button onClick={() => setRevision((value) => value + 1)}>
          重新加载监控页面
        </button>
      </div>
      {visible ? (
        <Frame
          key={`${embedded.href}-${revision}`}
          url={embedded.href}
          title={title}
        />
      ) : (
        <p role="status" className="muted">
          页面暂未显示，已暂停加载监控。
        </p>
      )}
    </div>
  );
}
