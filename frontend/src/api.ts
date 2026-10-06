export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type ModelSummary = {
  model_id: string;
  name: string;
  framework: string;
  model_type: string;
  local_model_path: string | null;
  status: string;
  registered_at: string;
  agent_connected: boolean;
  last_check_status: string | null;
};

export type ModelDetail = ModelSummary & {
  last_check: Record<string, unknown> | null;
  check_pending: boolean;
  check_requested_at: string | null;
  metadata: Record<string, unknown>;
  credential_configured: boolean;
  latest_monitoring: MonitoringResult | null;
};

export type FeatureDrift = {
  psi: number | null;
  ks: { statistic: number; p_value: number } | null;
  kl_divergence: number | null;
  status: string;
};

export type MonitoringResult = {
  last_run: string;
  health: string;
  overall_drift_status: string;
  data_quality: { status?: string; duplicate_rows?: number; null_rates?: Record<string, number> };
  drift: { features?: Record<string, FeatureDrift> };
  performance: { status?: string; accuracy?: number; precision?: number; recall?: number; f1?: number; reason?: string };
};

export type MonitoringModel = {
  model_id: string;
  name: string;
  framework: string;
  current_version: string;
  agent_connected: boolean;
  monitoring: MonitoringResult | null;
};

export type Project = {
  id: number;
  model_id: string;
  name: string;
  status: string;
  framework: string;
  drift_status: string;
};

export type Dashboard = {
  project: {
    id: number;
    model_id: string;
    name: string;
    framework: string;
    task_type: string;
    status: string;
    local_model_path: string | null;
    drift_status: string;
    metadata: Record<string, unknown>;
    last_check: Record<string, unknown> | null;
    check_pending: boolean;
    agent_connected: boolean;
  };
  agent_scan: unknown;
  latest_window: unknown;
  events: { kind: string; message: string; created_at: string }[];
};

async function request(path: string, options: RequestInit = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  models: (): Promise<ModelSummary[]> => request("/models"),
  model: (modelId: string): Promise<ModelDetail> => request(`/models/${modelId}`),
  monitoring: (): Promise<MonitoringModel[]> => request("/monitoring"),
  monitoringHistory: (modelId: string): Promise<{ model_id: string; runs: MonitoringResult[] }> => request(`/models/${modelId}/monitoring/history`),
  registerModel: (body: {
    name: string;
    framework: string;
    model_type: string;
    local_model_path: string;
  }): Promise<ModelDetail> =>
    request("/models", { method: "POST", body: JSON.stringify(body) }),
  triggerCheck: (modelId: string): Promise<ModelDetail> =>
    request(`/models/${modelId}/checks/trigger`, { method: "POST" }),
  modelCredential: (modelId: string): Promise<{ token: string | null; created: boolean; credential_configured: boolean; scopes: string[]; model_id: string }> =>
    request(`/models/${modelId}/credentials`, {
      method: "POST",
      body: JSON.stringify({
        label: "sentinelops-agent",
        scopes: ["model:check", "project:scan", "monitor:write"],
      }),
    }),
  rotateModelCredential: (modelId: string): Promise<{ token: string; credential_configured: boolean; scopes: string[]; model_id: string }> =>
    request(`/models/${modelId}/credentials/rotate`, {
      method: "POST",
      body: JSON.stringify({
        label: "sentinelops-agent",
        scopes: ["model:check", "project:scan", "monitor:write"],
      }),
    }),
  disconnectModelAgent: (modelId: string): Promise<{ model_id: string; disconnected: boolean; revoked_count: number; cancelled_pending_check: boolean }> =>
    request(`/models/${modelId}/credentials/disconnect`, { method: "POST" }),
  projects: (): Promise<Project[]> => request("/projects"),
  dashboard: (id: number): Promise<Dashboard> => request(`/projects/${id}/dashboard`),
  credential: (id: number) =>
    request(`/projects/${id}/credentials`, {
      method: "POST",
      body: JSON.stringify({
        label: "sentinelops-agent",
        scopes: ["model:check", "project:scan", "monitor:write"],
      }),
    }),
  demoMonitor: (id: number) => request(`/projects/${id}/monitoring/demo`, { method: "POST" }),
};
