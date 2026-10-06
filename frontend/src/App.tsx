import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  Copy,
  GitBranch,
  KeyRound,
  LayoutDashboard,
  ListChecks,
  Activity,
  Play,
  RefreshCw,
  Settings,
  Boxes,
  ShieldCheck,
} from "lucide-react";
import { api, ModelDetail, ModelSummary, MonitoringModel, MonitoringResult } from "./api";

type Page = "dashboard" | "models" | "monitoring" | "detail";

/* ---------- helpers ---------- */

function statusTone(status: string | null | undefined) {
  switch ((status ?? "").toUpperCase()) {
    case "READY":
    case "PASSED":
    case "HEALTHY":
    case "CLEAR":
    case "GOOD":
    case "NORMAL":
      return "green";
    case "CONNECTED":
      return "blue";
    case "ERROR":
    case "FAILED":
    case "CRITICAL":
    case "DRIFTED":
      return "red";
    case "PENDING":
    case "WARNING":
    case "DEGRADED":
      return "amber";
    default:
      return "gray";
  }
}

function Badge({ status, label }: { status: string | null | undefined; label?: string }) {
  const text = label ?? status ?? "NONE";
  return <span className={`badge ${statusTone(status)}`}>{String(text).toUpperCase()}</span>;
}

function formatTime(value: unknown) {
  if (typeof value !== "string") return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

function formatElapsed(value: string | null | undefined) {
  if (!value) return "00:00";
  const started = new Date(value).getTime();
  if (!Number.isFinite(started)) return "00:00";
  const totalSeconds = Math.max(0, Math.floor((Date.now() - started) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function errMsg(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

type CheckResult = {
  status?: string;
  prediction_test_status?: string;
  sample_prediction?: unknown;
  expected_features?: number | null;
  model_loaded?: boolean;
  model_file_exists?: boolean;
  model_path?: string;
  errors?: string[];
  received_at?: string;
};

/* ---------- small components ---------- */

function StatCard({ label, value, hint }: { label: string; value: number; hint: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <b className="stat-value">{value}</b>
      <span className="stat-hint">{hint}</span>
    </div>
  );
}

function metric(value: unknown) {
  return typeof value === "number" ? value.toFixed(4) : "—";
}

function MonitoringPanel({ monitoring }: { monitoring: MonitoringResult | null }) {
  if (!monitoring) return <section className="card"><div className="card-head"><h2>Monitoring</h2></div><div className="empty">No monitoring run yet. The local Agent keeps the datasets on disk and uploads only calculated results.</div></section>;
  const performance = monitoring.performance ?? {};
  const features = Object.entries(monitoring.drift?.features ?? {});
  return <section className="card">
    <div className="card-head"><h2>Monitoring</h2><span className="muted">Last run {formatTime(monitoring.last_run)}</span></div>
    <div className="monitor-summary">
      <div><span>Model health</span><Badge status={monitoring.health} /></div>
      <div><span>Overall drift</span><Badge status={monitoring.overall_drift_status} /></div>
      <div><span>Data quality</span><Badge status={monitoring.data_quality?.status} /></div>
      <div><span>Performance</span><Badge status={performance.status} /></div>
    </div>
    {performance.status === "AVAILABLE" ? <div className="metric-grid">
      {(["accuracy", "precision", "recall", "f1"] as const).map((name) => <div key={name}><span>{name.toUpperCase()}</span><b>{metric(performance[name])}</b></div>)}
    </div> : <p className="muted">Performance = UNKNOWN{performance.reason ? ` — ${performance.reason}` : ""}</p>}
    <div className="card-head section-head"><h2>Feature-level drift</h2><span className="muted">Drift is an input signal, not automatic model failure.</span></div>
    {features.length ? <div className="table-wrap"><table><thead><tr><th>Feature</th><th>PSI</th><th>KS statistic</th><th>KS p-value</th><th>KL divergence</th><th>Status</th></tr></thead><tbody>
      {features.map(([name, value]) => <tr key={name}><td className="strong">{name}</td><td>{metric(value.psi)}</td><td>{metric(value.ks?.statistic)}</td><td>{metric(value.ks?.p_value)}</td><td>{metric(value.kl_divergence)}</td><td><Badge status={value.status} /></td></tr>)}
    </tbody></table></div> : <div className="empty">No comparable numeric features were available.</div>}
  </section>;
}

function qualityCheck(status: string | undefined, label: string) {
  return <div className="quality-item" key={label}><span>{label}</span><Badge status={status ?? "UNKNOWN"} label={status === "CLEAR" ? "Good" : status} /></div>;
}

function MonitoringWorkspace({ rows, selected, history, onSelect }: { rows: MonitoringModel[]; selected?: MonitoringModel; history: MonitoringResult[]; onSelect: (row: MonitoringModel) => void }) {
  const monitored = rows.filter((row) => row.monitoring);
  const count = (health: string) => monitored.filter((row) => row.monitoring?.health === health).length;
  const current = selected?.monitoring;
  const quality = current?.data_quality;
  const nullWarning = Object.values(quality?.null_rates ?? {}).some((value) => value > 0);
  const outlierWarning = Object.values((quality as { iqr_outliers?: Record<string, { count: number }> } | undefined)?.iqr_outliers ?? {}).some((value) => value.count > 0);
  const schema = (quality as { schema?: { missing_columns?: unknown[]; added_columns?: unknown[]; type_mismatches?: unknown[] } } | undefined)?.schema;
  const schemaBad = Boolean(schema?.missing_columns?.length || schema?.added_columns?.length || schema?.type_mismatches?.length);
  const performanceStatus = current?.performance?.status === "AVAILABLE" ? ((current.performance.f1 ?? 1) < .7 ? "DEGRADED" : "HEALTHY") : "UNKNOWN";
  return <>
    <div className="page-head"><div><h1>Monitoring</h1><p>Centralized health and drift results from local Agent monitoring runs.</p></div></div>
    <section className="stats"><StatCard label="Models Monitored" value={monitored.length} hint={`${rows.length} registered`} /><StatCard label="Healthy" value={count("HEALTHY")} hint="Latest run" /><StatCard label="Warning" value={count("WARNING")} hint="Latest run" /><StatCard label="Critical" value={count("CRITICAL")} hint="Latest run" /></section>
    <section className="card"><div className="card-head"><h2>Model monitoring</h2><span className="muted">Select a model for detailed results</span></div>
      <div className="table-wrap"><table><thead><tr><th>Model name</th><th>Model ID</th><th>Health</th><th>Drift status</th><th>Performance</th><th>Last check</th><th aria-label="Action" /></tr></thead><tbody>
        {rows.map((row) => { const run = row.monitoring; const perf = run?.performance?.status === "AVAILABLE" ? ((run.performance.f1 ?? 1) < .7 ? "DEGRADED" : "HEALTHY") : "UNKNOWN"; return <tr key={row.model_id} className={selected?.model_id === row.model_id ? "selected" : ""}><td className="strong">{row.name}</td><td className="mono">{row.model_id.slice(0, 8)}…</td><td>{run ? <Badge status={run.health} /> : <span className="muted">Not monitored</span>}</td><td>{run ? <Badge status={run.overall_drift_status === "CLEAR" ? "NORMAL" : run.overall_drift_status} /> : "—"}</td><td>{run ? <Badge status={perf} /> : "—"}</td><td>{run ? formatTime(run.last_run) : "Never"}</td><td className="right"><button className="btn sm" onClick={() => onSelect(row)}>View</button></td></tr>; })}
      </tbody></table></div>
    </section>
    {!selected ? <section className="card"><div className="empty">Select a model to inspect its stored monitoring results. No values are generated by the dashboard.</div></section> : !current ? <section className="card"><div className="card-head"><h2>{selected.name}</h2></div><div className="empty">No monitoring data available. Run local Agent monitoring to generate results.</div></section> : <>
      <section className="card"><div className="card-head"><h2>{selected.name}</h2><span className="muted">Last monitoring run {formatTime(current.last_run)}</span></div><dl className="dl"><dt>Model ID</dt><dd className="mono">{selected.model_id}</dd><dt>Framework</dt><dd>{selected.framework}</dd><dt>Current version</dt><dd>{selected.current_version}</dd><dt>Health</dt><dd><Badge status={current.health} /></dd><dt>Drift</dt><dd><Badge status={current.overall_drift_status === "CLEAR" ? "NORMAL" : current.overall_drift_status} /></dd><dt>Performance</dt><dd><Badge status={performanceStatus} /></dd><dt>Data quality</dt><dd><Badge status={quality?.status === "CLEAR" ? "GOOD" : quality?.status} /></dd></dl></section>
      <section className="two-col"><div className="card"><div className="card-head"><h2>Data quality</h2></div><div className="quality-list">{qualityCheck(nullWarning ? "WARNING" : "CLEAR", "Missing values")}{qualityCheck((quality?.duplicate_rows ?? 0) > 0 ? "WARNING" : "CLEAR", "Duplicate rows")}{qualityCheck(outlierWarning ? "WARNING" : "CLEAR", "Outliers")}{qualityCheck(schemaBad ? "CRITICAL" : "CLEAR", "Schema consistency")}{qualityCheck(undefined, "Invalid/unexpected values")}</div></div><div className="card"><div className="card-head"><h2>Performance</h2></div>{current.performance?.status === "AVAILABLE" ? <div className="metric-grid">{(["accuracy", "precision", "recall", "f1"] as const).map((name) => <div key={name}><span>{name.toUpperCase()}</span><b>{metric(current.performance[name])}</b></div>)}</div> : <div className="empty">Performance: UNKNOWN. Ground-truth labels were unavailable for this run.</div>}</div></section>
      <MonitoringPanel monitoring={current} />
      <section className="card"><div className="card-head"><h2>Monitoring history</h2><span className="muted">Stored local-Agent results</span></div>{history.length ? <div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Health</th><th>Drift</th><th>Performance</th><th>Data quality</th></tr></thead><tbody>{history.map((run) => <tr key={`${run.last_run}-${run.health}`}><td>{formatTime(run.last_run)}</td><td><Badge status={run.health} /></td><td><Badge status={run.overall_drift_status === "CLEAR" ? "NORMAL" : run.overall_drift_status} /></td><td><Badge status={run.performance?.status === "AVAILABLE" ? "HEALTHY" : "UNKNOWN"} /></td><td><Badge status={run.data_quality?.status === "CLEAR" ? "GOOD" : run.data_quality?.status} /></td></tr>)}</tbody></table></div> : <div className="empty">No monitoring history is available.</div>}</section>
    </>}
  </>;
}

function ModelsTable({
  models,
  selectedId,
  onView,
}: {
  models: ModelSummary[];
  selectedId?: string;
  onView: (id: string) => void;
}) {
  if (models.length === 0) {
    return <div className="empty">No models registered yet. Register a model from the Models page.</div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Model ID</th>
            <th>Framework</th>
            <th>Status</th>
            <th>Agent</th>
            <th>Last check</th>
            <th aria-label="Action" />
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model_id} className={m.model_id === selectedId ? "selected" : ""}>
              <td className="strong">{m.name}</td>
              <td className="mono" title={m.model_id}>{m.model_id.slice(0, 8)}…</td>
              <td>{m.framework}</td>
              <td><Badge status={m.status} /></td>
              <td>
                <Badge
                  status={m.agent_connected ? "CONNECTED" : "REGISTERED"}
                  label={m.agent_connected ? "Connected" : "Not connected"}
                />
              </td>
              <td>
                {m.last_check_status ? <Badge status={m.last_check_status} /> : <span className="muted">Never</span>}
              </td>
              <td className="right">
                <button className="btn sm" onClick={() => onView(m.model_id)}>View</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- app ---------- */

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [loaded, setLoaded] = useState(false);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [monitoringRows, setMonitoringRows] = useState<MonitoringModel[]>([]);
  const [monitoringSelected, setMonitoringSelected] = useState<MonitoringModel>();
  const [monitoringHistory, setMonitoringHistory] = useState<MonitoringResult[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [detail, setDetail] = useState<ModelDetail>();
  const [name, setName] = useState("");
  const [modelPath, setModelPath] = useState("../demo_project/model.joblib");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const [pendingTick, setPendingTick] = useState(0);
  const selectedIdRef = useRef<string | undefined>(selectedId);
  selectedIdRef.current = selectedId;

  const loadModels = useCallback(async (preferId?: string) => {
    try {
      const list = await api.models();
      setModels(list);
      const activeId = preferId ?? selectedIdRef.current ?? list[0]?.model_id;
      if (activeId) {
        setSelectedId(activeId);
        setDetail(await api.model(activeId));
      }
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const loadMonitoring = useCallback(async () => {
    try {
      const rows = await api.monitoring();
      setMonitoringRows(rows);
      const active = rows.find((row) => row.model_id === monitoringSelected?.model_id) ?? rows[0];
      if (active) {
        setMonitoringSelected(active);
        setMonitoringHistory((await api.monitoringHistory(active.model_id)).runs);
      }
    } catch (e) {
      setError(errMsg(e));
    }
  }, [monitoringSelected?.model_id]);

  useEffect(() => {
    if (page === "monitoring") loadMonitoring();
  }, [page, loadMonitoring]);

  // Poll so terminal Agent runs update the UI automatically.
  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const list = await api.models();
        setModels((prev) => (JSON.stringify(prev) !== JSON.stringify(list) ? list : prev));
        const activeId = selectedIdRef.current;
        if (!activeId) return;
        const d = await api.model(activeId);
        setDetail((prev) => (JSON.stringify(prev) !== JSON.stringify(d) ? d : prev));
      } catch {
        // ignore polling network errors
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!detail?.check_pending) return;
    const timer = window.setInterval(() => setPendingTick((tick) => tick + 1), 1000);
    return () => window.clearInterval(timer);
  }, [detail?.check_pending]);

  const openModel = async (id: string) => {
    setSelectedId(id);
    setToken("");
    setError("");
    setPage("detail");
    try {
      setDetail(await api.model(id));
    } catch (e) {
      setError(errMsg(e));
    }
  };

  const openMonitoringModel = async (row: MonitoringModel) => {
    setMonitoringSelected(row);
    try {
      setMonitoringHistory((await api.monitoringHistory(row.model_id)).runs);
    } catch (e) {
      setError(errMsg(e));
    }
  };

  const handleRegister = async () => {
    if (busy || name.trim().length < 2 || modelPath.trim().length < 3) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.registerModel({
        name: name.trim(),
        framework: "scikit-learn",
        model_type: "classification",
        local_model_path: modelPath.trim(),
      });
      setName("");
      setToken("");
      setSelectedId(r.model_id);
      setDetail(r);
      setModels(await api.models());
      setPage("detail");
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCreateToken = async () => {
    const activeId = selectedIdRef.current;
    if (!activeId || busy) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.modelCredential(activeId);
      if (r?.token) setToken(r.token);
      setDetail(await api.model(activeId));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRotateToken = async () => {
    const activeId = selectedIdRef.current;
    if (!activeId || busy || !window.confirm("Rotate this Agent credential? The existing token will stop working immediately.")) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.rotateModelCredential(activeId);
      setToken(r.token);
      setDetail(await api.model(activeId));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnectAgent = async () => {
    const activeId = selectedIdRef.current;
    if (!activeId || busy || !window.confirm("Disconnect this Agent? Its active credential will be revoked. The registered model and monitoring history remain.")) return;
    setBusy(true);
    setError("");
    try {
      await api.disconnectModelAgent(activeId);
      setToken("");
      await loadModels(activeId);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleTriggerMonitoring = async () => {
    const activeId = selectedIdRef.current;
    if (!activeId || busy || detail?.check_pending) return;
    setBusy(true);
    setError("");
    try {
      setDetail(await api.triggerMonitoring(activeId));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleTriggerCheck = async () => {
    const activeId = selectedIdRef.current;
    if (!activeId || busy) return;
    setBusy(true);
    setError("");
    try {
      setDetail(await api.triggerCheck(activeId));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleRefresh = async () => {
    setBusy(true);
    setError("");
    try {
      await loadModels(selectedIdRef.current);
    } finally {
      setBusy(false);
    }
  };

  const copyToken = () => {
    if (!token) return;
    navigator.clipboard.writeText(token);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!loaded) {
    return (
      <div className="loading">
        <ShieldCheck size={26} />
        <b>SentinelOps</b>
        <p>{error || "Loading…"}</p>
      </div>
    );
  }

  const connectedAgents = models.filter((m) => m.agent_connected).length;
  const healthy = models.filter((m) => m.status.toUpperCase() === "READY").length;
  const issues = models.filter((m) => m.status.toUpperCase() === "ERROR").length;

  const navItems: { id: Page | null; label: string; icon: JSX.Element }[] = [
    { id: "dashboard", label: "Dashboard", icon: <LayoutDashboard size={16} /> },
    { id: "models", label: "Models", icon: <Boxes size={16} /> },
    { id: "monitoring", label: "Monitoring", icon: <Activity size={16} /> },
    { id: null, label: "Versions", icon: <GitBranch size={16} /> },
    { id: null, label: "Jobs", icon: <ListChecks size={16} /> },
    { id: null, label: "Settings", icon: <Settings size={16} /> },
  ];
  const activeNav: Page = page === "detail" ? "models" : page;

  const check = (detail?.last_check ?? null) as CheckResult | null;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <ShieldCheck size={20} />
          <b>SentinelOps</b>
        </div>
        <div className="topbar-right">
          <button className="btn sm" onClick={handleRefresh} disabled={busy}>
            <RefreshCw size={14} /> Refresh
          </button>
          <div className="user" title="Local workspace">
            <span className="avatar">LW</span>
            <span className="user-name">Local workspace</span>
          </div>
        </div>
      </header>

      <div className="body">
        <nav className="sidebar" aria-label="Main">
          {navItems.map((item) =>
            item.id ? (
              <button
                key={item.label}
                className={`nav-item ${activeNav === item.id ? "active" : ""}`}
                onClick={() => setPage(item.id as Page)}
              >
                {item.icon} {item.label}
              </button>
            ) : (
              <span key={item.label} className="nav-item disabled" aria-disabled="true">
                {item.icon} {item.label}
                <em>Soon</em>
              </span>
            )
          )}
        </nav>

        <main className="content">
          {error && (
            <div className="alert error" role="alert">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          {/* ---------------- Dashboard ---------------- */}
          {page === "dashboard" && (
            <>
              <div className="page-head">
                <div>
                  <h1>Dashboard</h1>
                  <p>Overview of registered models and their local Agents.</p>
                </div>
              </div>
              <section className="stats">
                <StatCard label="Models" value={models.length} hint="Registered" />
                <StatCard label="Connected Agents" value={connectedAgents} hint="Reporting to backend" />
                <StatCard label="Healthy Models" value={healthy} hint="Last check passed" />
                <StatCard label="Issues Detected" value={issues} hint="Last check failed" />
              </section>
              <section className="card">
                <div className="card-head">
                  <h2>Registered models</h2>
                </div>
                <ModelsTable models={models} selectedId={selectedId} onView={openModel} />
              </section>
            </>
          )}

          {/* ---------------- Models ---------------- */}
          {page === "models" && (
            <>
              <div className="page-head">
                <div>
                  <h1>Models</h1>
                  <p>Register scikit-learn classification models. Artifacts stay on your machine.</p>
                </div>
              </div>
              <section className="card">
                <div className="card-head">
                  <h2>Register model</h2>
                  <span className="muted">Stored with a unique Model ID</span>
                </div>
                <div className="form-row">
                  <label className="field">
                    <span>Model name</span>
                    <input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="e.g. Iris Classifier"
                    />
                  </label>
                  <label className="field grow">
                    <span>Local model path</span>
                    <input
                      value={modelPath}
                      onChange={(e) => setModelPath(e.target.value)}
                      placeholder="Path to model.joblib"
                    />
                  </label>
                  <button
                    className="btn primary"
                    disabled={busy || name.trim().length < 2 || modelPath.trim().length < 3}
                    onClick={handleRegister}
                  >
                    Register
                  </button>
                </div>
              </section>
              <section className="card">
                <div className="card-head">
                  <h2>All models</h2>
                </div>
                <ModelsTable models={models} selectedId={selectedId} onView={openModel} />
              </section>
            </>
          )}

          {/* ---------------- Monitoring ---------------- */}
          {page === "monitoring" && <MonitoringWorkspace rows={monitoringRows} selected={monitoringSelected} history={monitoringHistory} onSelect={openMonitoringModel} />}

          {/* ---------------- Model details ---------------- */}
          {page === "detail" && !detail && (
            <section className="card">
              <div className="empty">No model selected.</div>
            </section>
          )}

          {page === "detail" && detail && (
            <>
              <button className="back" onClick={() => setPage("models")}>
                <ChevronLeft size={14} /> Models
              </button>
              <div className="page-head">
                <div>
                  <h1>
                    {detail.name} <Badge status={detail.status} />
                  </h1>
                  <p className="mono">{detail.model_id}</p>
                </div>
                <div className="button-row">
                <button
                  className="btn primary"
                  disabled={busy || detail.check_pending}
                  onClick={handleTriggerCheck}
                >
                  <Play size={14} /> {detail.pending_action === "model_check" ? `Checking… ${formatElapsed(detail.check_requested_at)}` : "Run Model Check"}
                </button>
                <button
                  className="btn"
                  disabled={busy || detail.check_pending}
                  onClick={handleTriggerMonitoring}
                >
                  <Activity size={14} /> {detail.pending_action === "monitor_model" ? `Monitoring… ${formatElapsed(detail.check_requested_at)}` : "Run Monitoring"}
                </button>
              </div>
              </div>

              {detail.check_pending && (
                <div className="alert info">
                  <RefreshCw size={16} />
                  <span>
                    {detail.pending_action === "monitor_model" ? "Monitoring" : "Model check"} in progress · {formatElapsed(detail.check_requested_at)} elapsed. The connected local Agent will execute it automatically.
                  </span>
                </div>
              )}

              <div className="two-col">
                <section className="card">
                  <div className="card-head">
                    <h2>Model information</h2>
                  </div>
                  <dl className="dl">
                    <dt>Model name</dt>
                    <dd>{detail.name}</dd>
                    <dt>Model ID</dt>
                    <dd className="mono">{detail.model_id}</dd>
                    <dt>Framework</dt>
                    <dd>{detail.framework}</dd>
                    <dt>Model type</dt>
                    <dd>{detail.model_type}</dd>
                    <dt>Local path</dt>
                    <dd className="mono">{detail.local_model_path ?? "—"}</dd>
                    <dt>Agent status</dt>
                    <dd>
                      <Badge
                        status={detail.agent_connected ? "CONNECTED" : "REGISTERED"}
                        label={detail.agent_connected ? "Connected" : "Not connected"}
                      />
                    </dd>
                    <dt>Model status</dt>
                    <dd>
                      <Badge status={detail.check_pending ? "PENDING" : detail.status} />
                    </dd>
                    <dt>Last check</dt>
                    <dd>{check ? formatTime(check.received_at) : "Never"}</dd>
                  </dl>
                </section>

                <section className="card">
                  <div className="card-head">
                    <h2>Latest check result</h2>
                    {check && <Badge status={check.status ?? detail.last_check_status} />}
                  </div>
                  {check ? (
                    <>
                      <dl className="dl">
                        <dt>Prediction test</dt>
                        <dd><Badge status={check.prediction_test_status} /></dd>
                        <dt>Sample prediction</dt>
                        <dd>
                          {check.sample_prediction !== null && check.sample_prediction !== undefined
                            ? String(check.sample_prediction)
                            : "—"}
                        </dd>
                        <dt>Expected features</dt>
                        <dd>{check.expected_features ?? "—"}</dd>
                        <dt>Model loaded</dt>
                        <dd>{check.model_loaded ? "Yes (scikit-learn)" : "No"}</dd>
                        <dt>Artifact file</dt>
                        <dd>{check.model_file_exists ? "Found on disk" : "Missing"}</dd>
                        <dt>Resolved path</dt>
                        <dd className="mono">{String(check.model_path || detail.local_model_path || "—")}</dd>
                      </dl>
                      {Array.isArray(check.errors) && check.errors.length > 0 && (
                        <div className="alert error">
                          <AlertTriangle size={16} />
                          <div>
                            <b>Validation errors</b>
                            <ul>
                              {check.errors.map((err, i) => (
                                <li key={i}>{String(err)}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      )}
                      <details className="raw">
                        <summary>View JSON payload</summary>
                        <pre>{JSON.stringify(check, null, 2)}</pre>
                      </details>
                    </>
                  ) : (
                    <div className="empty">
                      No check has run yet. Create an Agent token, start the Agent, then click{" "}
                      <b>Run Model Check</b>.
                    </div>
                  )}
                </section>
              </div>

              <MonitoringPanel monitoring={detail.latest_monitoring} />

              <section className="card">
                <div className="card-head">
                  <h2>Agent connection</h2>
                  <span className="muted">Scoped token for the local Agent</span>
                </div>
                <div className="form-row">
                  {detail.credential_configured ? <><span className="muted"><Badge status="CONNECTED" label="Configured" /> Existing Agent credential retained. Its secret is not shown again.</span><button className="btn" disabled={busy} onClick={handleRotateToken}><RefreshCw size={14} /> Rotate credential</button><button className="btn danger" disabled={busy} onClick={handleDisconnectAgent}>Disconnect Agent</button></> : <button className="btn" disabled={busy} onClick={handleCreateToken}><KeyRound size={14} /> Create token</button>}
                </div>
                {token && (
                  <div className="token-row">
                    <code className="token">{token}</code>
                    <button className="btn sm" type="button" onClick={copyToken}>
                      {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy"}
                    </button>
                  </div>
                )}
                {!detail.credential_configured || token ? <><p className="muted label">One-shot check</p><pre className="cmd">sentinelops-agent check --api-url {API} --model-id {detail.model_id} --token {token || "YOUR_TOKEN"}</pre><p className="muted label">Run as daemon</p><pre className="cmd">sentinelops-agent serve --api-url {API} --model-id {detail.model_id} --token {token || "YOUR_TOKEN"}</pre><p className="muted label">Monitoring with local data</p><pre className="cmd">sentinelops-agent monitor-model --api-url {API} --model-id {detail.model_id} --token {token || "YOUR_TOKEN"} --reference PATH_TO_REFERENCE.csv --current PATH_TO_CURRENT.csv --label-column target</pre></> : <p className="muted label">Use the token saved when the Agent was first configured. Rotation and revocation remain explicit future actions.</p>}
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
