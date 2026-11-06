export interface SourceCitation {
  record_id: string;
  title: string;
  url: string;
  pdf_page: number;
  table_id?: string | null;
}

export interface ComparisonData {
  metric: string;
  abbreviation?: string;
  table_id?: string;
  geography: string;
  periods: string[];
  period_keys?: string[];
  values: Record<string, number>;
  unit: string;
  change_pp?: number | null;
  change_percentage_points?: number | null;
  direction?: string | null;
}

export interface AskResponse {
  status: "answered" | "insufficient_evidence";
  answer: string;
  sources: SourceCitation[];
  comparison?: ComparisonData | null;
}

export interface HealthResponse {
  status: string;
  chapter_chunks: number;
  table_rows: number;
}

export type ApiErrorKind =
  | "network"
  | "service_unavailable"
  | "timeout"
  | "client_error"
  | "configuration_error";

export class ApiRequestError extends Error {
  readonly kind: ApiErrorKind;
  readonly statusCode?: number;
  readonly retryable: boolean;

  constructor(
    message: string,
    kind: ApiErrorKind,
    statusCode?: number,
    retryable = true,
  ) {
    super(message);
    this.name = "ApiRequestError";
    this.kind = kind;
    this.statusCode = statusCode;
    this.retryable = retryable;
  }
}

const REQUEST_TIMEOUT_MS = 25000;

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");
const askUrl = `${apiBaseUrl}/api/ask`;
const healthUrl = `${apiBaseUrl}/api/health`;

function isPlaceholderOrigin(url: string): boolean {
  const lower = url.toLowerCase();
  return (
    lower.includes("your-backend") ||
    lower.includes("example.com") ||
    lower.includes("<") ||
    lower.includes(">")
  );
}

function isCapacitorOrProductionRuntime(): boolean {
  if (import.meta.env.PROD) {
    return true;
  }
  if (typeof window !== "undefined") {
    const { protocol, origin } = window.location;
    if (protocol === "capacitor:" || origin === "https://localhost") {
      return true;
    }
  }
  return false;
}

function assertProductionApiConfigured(): void {
  if (!isCapacitorOrProductionRuntime()) {
    return;
  }

  if (!apiBaseUrl || isPlaceholderOrigin(apiBaseUrl)) {
    throw new ApiRequestError(
      "VITE_API_BASE_URL is not configured for this build. Set VITE_API_BASE_URL to your deployed HTTPS backend URL in frontend/.env.production.local, then run npm run build and npx cap sync android.",
      "configuration_error",
      undefined,
      false,
    );
  }

  try {
    const parsed = new URL(apiBaseUrl);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
      throw new Error("Invalid protocol");
    }
  } catch {
    throw new ApiRequestError(
      `Invalid VITE_API_BASE_URL "${apiBaseUrl}". Provide a valid HTTPS origin (for example, https://your-service.onrender.com).`,
      "configuration_error",
      undefined,
      false,
    );
  }
}

export function getApiConfigNotice(): string | null {
  if (!isCapacitorOrProductionRuntime()) {
    return null;
  }
  if (!apiBaseUrl || isPlaceholderOrigin(apiBaseUrl)) {
    return "Development build notice: VITE_API_BASE_URL is not set to a deployed HTTPS backend yet. Set VITE_API_BASE_URL in frontend/.env.production.local and rebuild before submitting the APK.";
  }
  try {
    const parsed = new URL(apiBaseUrl);
    const host = parsed.hostname.toLowerCase();
    const isLocalHost =
      host === "localhost" ||
      host === "127.0.0.1" ||
      host === "10.0.2.2" ||
      host.startsWith("192.168.") ||
      host.startsWith("10.") ||
      host.endsWith(".local");
    if (parsed.protocol !== "https:" || isLocalHost) {
      return `Development build notice: Currently pointing to ${apiBaseUrl}. Set VITE_API_BASE_URL to a public HTTPS backend before final APK submission.`;
    }
  } catch {
    return `Configuration warning: VITE_API_BASE_URL (${apiBaseUrl}) is not a valid URL.`;
  }
  return null;
}

export async function fetchHealth(
  externalSignal?: AbortSignal,
): Promise<HealthResponse> {
  assertProductionApiConfigured();

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 6000);
  const onAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", onAbort, { once: true });

  try {
    const response = await fetch(healthUrl, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = "The Economic Survey backend is currently unavailable.";
      try {
        const body = (await response.json()) as { detail?: string };
        if (body?.detail) {
          detail = body.detail;
        }
      } catch {
        // Ignore JSON parse failure on error body
      }
      throw new ApiRequestError(
        detail,
        response.status === 503 ? "service_unavailable" : "network",
        response.status,
        true,
      );
    }

    return (await response.json()) as HealthResponse;
  } catch (err) {
    if (err instanceof ApiRequestError) {
      throw err;
    }
    throw new ApiRequestError(
      "Could not reach the backend server. Verify your network connection or backend service and retry.",
      "network",
      undefined,
      true,
    );
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", onAbort);
  }
}

export async function askSurvey(
  question: string,
  externalSignal?: AbortSignal,
): Promise<AskResponse> {
  assertProductionApiConfigured();

  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  const onExternalAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", onExternalAbort, { once: true });

  const signal = controller.signal;

  try {
    const response = await fetch(askUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim() }),
      signal,
    });

    if (!response.ok) {
      let detail = "";
      try {
        const errBody = (await response.json()) as { detail?: string };
        if (typeof errBody?.detail === "string") {
          detail = errBody.detail;
        }
      } catch {
        // Ignore non-JSON error body
      }

      if (response.status === 503) {
        throw new ApiRequestError(
          detail ||
            "The backend service or language model is temporarily unavailable (HTTP 503). Please check that the server and corpus are ready, then retry.",
          "service_unavailable",
          503,
          true,
        );
      }

      if (response.status === 400 || response.status === 422) {
        throw new ApiRequestError(
          detail || "Please enter a valid question (up to 1,000 characters).",
          "client_error",
          response.status,
          false,
        );
      }

      throw new ApiRequestError(
        detail ||
          `The server returned an unexpected error (HTTP ${response.status}). You can retry your question.`,
        "network",
        response.status,
        true,
      );
    }

    const data = (await response.json()) as AskResponse;
    return data;
  } catch (err) {
    if (err instanceof ApiRequestError) {
      throw err;
    }
    if (timedOut) {
      throw new ApiRequestError(
        "The request timed out while waiting for the backend. Please retry.",
        "timeout",
        undefined,
        true,
      );
    }
    throw new ApiRequestError(
      "Unable to reach the backend server. Check your internet connection or verify the backend is running, then retry.",
      "network",
      undefined,
      true,
    );
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", onExternalAbort);
  }
}
