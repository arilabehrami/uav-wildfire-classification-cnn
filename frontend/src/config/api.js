export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
export const CLASSIFY_URL = `${API_BASE}/api/classify`;
export const HEALTH_URL = `${API_BASE}/health`;
export const MAX_BYTES = 10 * 1024 * 1024;
