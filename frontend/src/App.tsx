import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import {
  ApiRequestError,
  askSurvey,
  type AskResponse,
  type ComparisonData,
  type SourceCitation,
} from "./api";
import "./App.css";

const SAMPLE_QUESTION =
  "How did Tamil Nadu’s unemployment rate change between April–June and July–September 2025?";

const INDEXED_SOURCES: readonly {
  id: string;
  label: string;
  url: string;
}[] = [
  {
    id: "echap12",
    label: "Economic Survey 2025–26, Chapter 12: Employment and Skill Development",
    url: "https://www.indiabudget.gov.in/economicsurvey/doc/eschapter/echap12.pdf",
  },
  {
    id: "tab8.10",
    label: "Statistical Appendix Table 8.10: Labour Force Participation Rate (LFPR)",
    url: "https://www.indiabudget.gov.in/economicsurvey/doc/stat/tab8.10.pdf",
  },
  {
    id: "tab8.12",
    label: "Statistical Appendix Table 8.12: Unemployment Rate (UR)",
    url: "https://www.indiabudget.gov.in/economicsurvey/doc/stat/tab8.12.pdf",
  },
];

interface ConversationTurn {
  id: string;
  question: string;
  response: AskResponse;
}

interface ErrorBannerState {
  message: string;
  failedQuestion: string;
  retryable: boolean;
}

function buildSafePdfHref(url: string, pdfPage: number): string | null {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    const base = url.split("#")[0];
    if (Number.isInteger(pdfPage) && pdfPage > 0) {
      return `${base}#page=${pdfPage}`;
    }
    return base;
  } catch {
    return null;
  }
}

function resolvePeriodValue(
  comparison: ComparisonData,
  periodLabel: string,
  index: number,
): number | null {
  const direct = comparison.values[periodLabel];
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return direct;
  }
  const fallbackKey = comparison.period_keys?.[index];
  if (fallbackKey) {
    const byKey = comparison.values[fallbackKey];
    if (typeof byKey === "number" && Number.isFinite(byKey)) {
      return byKey;
    }
  }
  return null;
}

function formatRateValue(value: number | null, unit: string): string {
  if (value === null) {
    return "—";
  }
  const formatted = value.toFixed(1);
  return unit === "percent" ? `${formatted}%` : `${formatted} ${unit}`;
}

function formatChangeSummary(changePp: number): string {
  const absText = Math.abs(changePp).toFixed(1);
  if (changePp < 0) {
    return `${changePp.toFixed(1)} percentage points (decrease of ${absText} percentage points)`;
  }
  if (changePp > 0) {
    return `+${changePp.toFixed(1)} percentage points (increase of ${absText} percentage points)`;
  }
  return "0.0 percentage points (unchanged)";
}

function formatSourceLabel(source: SourceCitation): string {
  if (source.table_id) {
    return `Table ${source.table_id}: ${source.title} (PDF page ${source.pdf_page})`;
  }
  return `${source.title} — PDF page ${source.pdf_page}`;
}

function ComparisonTable({ comparison }: { comparison: ComparisonData }) {
  const periods = Array.isArray(comparison.periods) ? comparison.periods : [];
  if (periods.length === 0) {
    return null;
  }

  const rawChange =
    typeof comparison.change_pp === "number"
      ? comparison.change_pp
      : typeof comparison.change_percentage_points === "number"
        ? comparison.change_percentage_points
        : null;

  const rateHeading = comparison.abbreviation
    ? `${comparison.metric} (${comparison.abbreviation})`
    : comparison.metric;

  const tableCaption = comparison.table_id
    ? `${comparison.geography} — ${rateHeading} (Table ${comparison.table_id})`
    : `${comparison.geography} — ${rateHeading}`;

  return (
    <div className="comparison-block">
      <span className="section-label">Structured Figures</span>
      <p className="comparison-caption">{tableCaption}</p>
      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th scope="col">Period</th>
              <th scope="col" className="numeric-cell">
                Rate ({comparison.unit === "percent" ? "%" : comparison.unit})
              </th>
            </tr>
          </thead>
          <tbody>
            {periods.map((period, idx) => {
              const val = resolvePeriodValue(comparison, period, idx);
              return (
                <tr key={period}>
                  <td>{period}</td>
                  <td className="numeric-cell">
                    {formatRateValue(val, comparison.unit)}
                  </td>
                </tr>
              );
            })}
            {rawChange !== null && (
              <tr className="comparison-change-row">
                <td>Change</td>
                <td className="numeric-cell">{formatChangeSummary(rawChange)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function App() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const saved = window.localStorage.getItem("surveyer-theme");
      return saved === "dark" ? "dark" : "light";
    } catch {
      return "light";
    }
  });
  const [draft, setDraft] = useState<string>("");
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [errorBanner, setErrorBanner] = useState<ErrorBannerState | null>(null);

  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const activeAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    document.title = "The Economic Surveyer";
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("surveyer-theme", theme);
    } catch {
      // Ignore storage restrictions
    }
  }, [theme]);

  function handleToggleTheme() {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  }

  async function submitQuestion(rawQuestion: string) {
    const trimmed = rawQuestion.trim();
    if (!trimmed || isLoading) {
      return;
    }

    activeAbortRef.current?.abort();
    const controller = new AbortController();
    activeAbortRef.current = controller;

    setIsLoading(true);
    setPendingQuestion(trimmed);
    setErrorBanner(null);

    try {
      const response = await askSurvey(trimmed, controller.signal);
      const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      setTurns((prev) => [
        ...prev,
        {
          id: turnId,
          question: trimmed,
          response,
        },
      ]);
      setDraft("");
    } catch (err) {
      setDraft((current) => (current.trim() ? current : trimmed));
      if (err instanceof ApiRequestError) {
        setErrorBanner({
          message: err.message,
          failedQuestion: trimmed,
          retryable: err.retryable,
        });
      } else {
        setErrorBanner({
          message:
            "Request failed before receiving a response from FastAPI. Please retry.",
          failedQuestion: trimmed,
          retryable: true,
        });
      }
    } finally {
      setIsLoading(false);
      setPendingQuestion(null);
      activeAbortRef.current = null;
    }
  }

  function handleFormSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(draft);
  }

  function handleTextareaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion(draft);
    }
  }

  function handleSampleClick() {
    setDraft(SAMPLE_QUESTION);
    setErrorBanner(null);
    composerRef.current?.focus();
  }

  function handleNewConversation() {
    activeAbortRef.current?.abort();
    activeAbortRef.current = null;
    setIsLoading(false);
    setPendingQuestion(null);
    setTurns([]);
    setErrorBanner(null);
    setDraft("");
    composerRef.current?.focus();
  }

  function handleRetry() {
    if (!errorBanner || isLoading) {
      return;
    }
    const questionToRetry = draft.trim() || errorBanner.failedQuestion;
    void submitQuestion(questionToRetry);
  }

  const isConversationEmpty = turns.length === 0 && !isLoading;

  return (
    <div className="surveyer-shell">
      <header className="surveyer-header">
        <h1 className="surveyer-title">The Economic Surveyer</h1>
        <p className="surveyer-subtitle">
          Ask questions about employment and labour force figures in India.
        </p>
      </header>

      <div className="surveyer-grid">
        <main className="chat-column" aria-label="Survey question and answer thread">
          <div className="chat-controls-bar">
            <span className="section-label">Question &amp; Evidence Thread</span>
            <button
              type="button"
              className="new-conversation-btn"
              onClick={handleNewConversation}
              disabled={turns.length === 0 && !errorBanner && !draft && !isLoading}
            >
              New conversation
            </button>
          </div>

          <section className="composer-section" aria-label="Ask a question">
            <form className="composer-form" onSubmit={handleFormSubmit}>
              <label htmlFor="survey-question-input" className="composer-label">
                Question
              </label>
              <div className="composer-input-row">
                <textarea
                  id="survey-question-input"
                  ref={composerRef}
                  className="composer-textarea"
                  rows={2}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={handleTextareaKeyDown}
                  placeholder="Ask about employment, a state, or a quarter…"
                  disabled={isLoading}
                  maxLength={1000}
                />
                <button
                  type="submit"
                  className="composer-submit-btn"
                  disabled={isLoading || !draft.trim()}
                >
                  {isLoading ? "Asking…" : "Ask"}
                </button>
              </div>
              <p className="composer-hint">
                Press Enter to submit, Shift + Enter for a new line.
              </p>
            </form>
          </section>

          <div className="live-region" aria-live="polite" aria-atomic="true">
            {isLoading && pendingQuestion && (
              <div className="loading-banner" role="status">
                <strong>Retrieving evidence and preparing answer</strong> for “{pendingQuestion}”…
              </div>
            )}
          </div>

          {errorBanner && (
            <div className="live-region" role="alert">
              <div className="error-banner">
                <p className="error-banner-title">Unable to complete request</p>
                <p className="error-banner-message">{errorBanner.message}</p>
                {errorBanner.retryable && (
                  <div className="error-banner-actions">
                    <button
                      type="button"
                      className="retry-btn"
                      onClick={handleRetry}
                      disabled={isLoading}
                    >
                      Retry question
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          <section className="conversation-stream" aria-label="Conversation">
            {isConversationEmpty && (
              <div className="empty-state-block">
                <span className="section-label">Sample Question</span>
                <button
                  type="button"
                  className="sample-question-btn"
                  onClick={handleSampleClick}
                >
                  {SAMPLE_QUESTION}
                </button>
              </div>
            )}

            {turns.map((turn) => {
              const { response } = turn;
              const isInsufficient = response.status === "insufficient_evidence";
              const paragraphs = response.answer
                .split(/\n+/)
                .map((line) => line.trim())
                .filter(Boolean);
              const hasSources =
                !isInsufficient &&
                Array.isArray(response.sources) &&
                response.sources.length > 0;

              return (
                <article key={turn.id} className="turn-entry">
                  <div className="turn-question-block">
                    <span className="section-label">Question</span>
                    <h2 className="turn-question-text">{turn.question}</h2>
                  </div>

                  <div
                    className={
                      isInsufficient
                        ? "turn-answer-block insufficient-notice"
                        : "turn-answer-block"
                    }
                  >
                    <span className="section-label">
                      {isInsufficient ? "Insufficient Evidence" : "Answer"}
                    </span>
                    {paragraphs.map((para, idx) => (
                      <p key={`${turn.id}-p-${idx}`} className="answer-paragraph">
                        {para}
                      </p>
                    ))}
                  </div>

                  {!isInsufficient && response.comparison && (
                    <ComparisonTable comparison={response.comparison} />
                  )}

                  {hasSources && (
                    <div className="sources-block">
                      <span className="section-label">Sources</span>
                      <ul className="sources-list">
                        {response.sources.map((src) => {
                          const safeHref = buildSafePdfHref(src.url, src.pdf_page);
                          const label = formatSourceLabel(src);
                          return (
                            <li key={`${turn.id}-${src.record_id}`} className="source-item">
                              {safeHref ? (
                                <a
                                  href={safeHref}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="source-link"
                                >
                                  {label}
                                </a>
                              ) : (
                                <span>{label}</span>
                              )}
                              <span className="source-record-id">[{src.record_id}]</span>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </article>
              );
            })}
          </section>
        </main>

        <aside className="about-column" aria-label="About the Economic Survey">
          <div className="about-panel">
            <h2 className="about-panel-title">About the Economic Survey</h2>

            <div className="about-section">
              <h3 className="about-section-heading">What it is</h3>
              <p className="about-section-body">
                India’s Economic Survey reviews the economy and discusses topics
                including employment and skill development.
              </p>
            </div>

            <div className="about-section">
              <h3 className="about-section-heading">What you can ask here</h3>
              <p className="about-section-body">
                Look up unemployment and labour force participation figures,
                compare covered quarters, and read explanations supported by the
                survey.
              </p>
            </div>

            <div className="about-section">
              <h3 className="about-section-heading">Sources in this app</h3>
              <ul className="about-sources-list">
                {INDEXED_SOURCES.map((item) => (
                  <li key={item.id}>
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="source-link"
                    >
                      {item.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </aside>
      </div>

      <button
        type="button"
        className="theme-toggle-btn"
        onClick={handleToggleTheme}
        aria-label={
          theme === "light" ? "Switch to dark mode" : "Switch to light mode"
        }
        aria-pressed={theme === "dark"}
        title={
          theme === "light" ? "Switch to dark mode" : "Switch to light mode"
        }
      >
        <svg
          className="theme-toggle-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M9 18h6" />
          <path d="M10 22h4" />
          <path d="M12 2a7 7 0 0 0-4 12.74V17a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-2.26A7 7 0 0 0 12 2z" />
        </svg>
        <span>{theme === "light" ? "Dark mode" : "Light mode"}</span>
      </button>
    </div>
  );
}

export default App;
