import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function formatCurrency(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Math.round(value).toLocaleString("en-US");
}

function formatYears(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${value} years`;
}

function formatHistory(value) {
  return value ? "Available" : "No history";
}

function riskBand(score) {
  if (score < 0.33) {
    return { key: "lower", label: "Lower model score" };
  }

  if (score < 0.66) {
    return { key: "moderate", label: "Moderate model score" };
  }

  return { key: "higher", label: "Higher model score" };
}

const FEATURE_LABELS = {
  EXT_SOURCE_1: "External Risk Source 1",
  EXT_SOURCE_2: "External Risk Source 2",
  EXT_SOURCE_3: "External Risk Source 3",
  AMT_INCOME_TOTAL: "Annual Income",
  AMT_CREDIT: "Credit Amount",
  AMT_ANNUITY: "Annuity Amount",
  AMT_GOODS_PRICE: "Goods Price",
  DAYS_BIRTH: "Age",
  DAYS_EMPLOYED: "Employment Duration",
  NAME_EDUCATION_TYPE: "Education",
  NAME_INCOME_TYPE: "Income Type",
  NAME_FAMILY_STATUS: "Family Status",
  BUREAU_LOAN_COUNT: "Bureau Loan Count",
  BUREAU_DEBT_MEAN: "Average Bureau Debt",
  PREV_APPLICATION_COUNT: "Previous Application Count",
  PREV_CREDIT_TO_APPLICATION_RATIO:
    "Previous Credit / Application Ratio",
  INST_PAYMENT_RECORD_COUNT: "Installment Payment Count",
  INST_LATE_PAYMENT_RATE: "Late Payment Rate",
};

function formatFeatureLabel(featureName) {
  return FEATURE_LABELS[featureName] ?? featureName;
}

function formatFeatureValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (typeof value === "number") {
    if (Number.isInteger(value) || Math.abs(value) >= 100) {
      return Math.round(value).toLocaleString("en-US");
    }

    return value.toFixed(2);
  }

  return String(value);
}

function directionSymbol(direction) {
  if (direction === "increases_score") {
    return "↑";
  }

  if (direction === "decreases_score") {
    return "↓";
  }

  return "→";
}

function directionLabel(direction) {
  if (direction === "increases_score") {
    return "Increases model score";
  }

  if (direction === "decreases_score") {
    return "Decreases model score";
  }

  return "Neutral effect";
}

function formatMetric(value) {
  if (typeof value !== "number") {
    return "—";
  }

  return value.toFixed(4);
}

function formatPercent(value) {
  if (typeof value !== "number") {
    return "—";
  }

  return `${(value * 100).toFixed(2)}%`;
}

const VIEW_META = {
  dashboard: {
    eyebrow: "CREDIT RISK PLATFORM",
    title: "Dashboard",
    subtitle:
      "Explainable machine learning for credit risk assessment.",
  },
  "risk-analysis": {
    eyebrow: "CREDIT RISK PLATFORM",
    title: "Risk Analysis",
    subtitle:
      "Select a customer from the demo dataset and generate a live CreditLens model risk score.",
  },
  "model-information": {
    eyebrow: "CREDIT RISK PLATFORM",
    title: "Model Information",
    subtitle:
      "Model contract, validation, and stability metrics.",
  },
  fairness: {
    eyebrow: "CREDIT RISK PLATFORM",
    title: "Fairness Audit",
    subtitle:
      "CODE_GENDER is excluded from predictive inputs and is used only for post-hoc auditing.",
  },
};

function App() {
  const [activeView, setActiveView] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("loading");
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState("");

  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [customerProfile, setCustomerProfile] =
    useState(null);

  const [profileError, setProfileError] =
    useState("");

  const [explanation, setExplanation] =
    useState(null);

  const [explanationError, setExplanationError] =
    useState("");

  const [loadingCustomers, setLoadingCustomers] =
    useState(false);

  const [loadingProfile, setLoadingProfile] =
    useState(false);

  const [loadingExplanation, setLoadingExplanation] =
    useState(false);

  const [predicting, setPredicting] =
    useState(false);

  const [fairnessSummary, setFairnessSummary] =
    useState(null);

  const [fairnessError, setFairnessError] =
    useState("");

  const [loadingFairness, setLoadingFairness] =
    useState(false);

  const explanationRequestRef = useRef(null);

  function abortPendingExplanation() {
    if (explanationRequestRef.current) {
      explanationRequestRef.current.abort();
      explanationRequestRef.current = null;
    }
  }

  useEffect(() => {
    async function loadBackendData() {
      try {
        const healthResponse = await fetch(
          `${API_BASE_URL}/health`
        );

        if (!healthResponse.ok) {
          throw new Error(
            "Health endpoint returned an error."
          );
        }

        const healthData =
          await healthResponse.json();

        if (healthData.status !== "healthy") {
          throw new Error(
            "CreditLens API is not healthy."
          );
        }

        setApiStatus("ready");

        const modelResponse = await fetch(
          `${API_BASE_URL}/model-info`
        );

        if (!modelResponse.ok) {
          throw new Error(
            "Model information could not be loaded."
          );
        }

        const modelData =
          await modelResponse.json();

        setModelInfo(modelData);
      } catch (err) {
        console.error(err);

        setApiStatus("offline");

        setError(
          err instanceof Error
            ? err.message
            : "Backend connection failed."
        );
      }
    }

    loadBackendData();
  }, []);

  useEffect(() => {
    if (!analysisOpen || !selectedCustomer) {
      return;
    }

    let cancelled = false;

    async function loadCustomerProfile() {
      setLoadingProfile(true);
      setProfileError("");
      setCustomerProfile(null);

      try {
        const response = await fetch(
          `${API_BASE_URL}/demo/customers/${selectedCustomer}`
        );

        if (!response.ok) {
          const errorData =
            await response.json();

          throw new Error(
            errorData.detail ??
              "Customer profile could not be loaded."
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setCustomerProfile(data);
        }
      } catch (err) {
        if (!cancelled) {
          setProfileError(
            err instanceof Error
              ? err.message
              : "Customer profile failed."
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingProfile(false);
        }
      }
    }

    loadCustomerProfile();

    return () => {
      cancelled = true;
    };
  }, [analysisOpen, selectedCustomer]);

  useEffect(() => {
    if (activeView !== "fairness") {
      return;
    }

    let cancelled = false;

    async function loadFairnessSummary() {
      setLoadingFairness(true);
      setFairnessError("");
      setFairnessSummary(null);

      try {
        const response = await fetch(
          `${API_BASE_URL}/fairness-summary`
        );

        if (!response.ok) {
          const errorData =
            await response.json();

          throw new Error(
            errorData.detail ??
              "Fairness summary could not be loaded."
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setFairnessSummary(data);
        }
      } catch (err) {
        if (!cancelled) {
          setFairnessError(
            err instanceof Error
              ? err.message
              : "Fairness summary failed."
          );
        }
      } finally {
        if (!cancelled) {
          setLoadingFairness(false);
        }
      }
    }

    loadFairnessSummary();

    return () => {
      cancelled = true;
    };
  }, [activeView]);

  async function startAnalysis() {
    abortPendingExplanation();

    setAnalysisOpen(true);
    setPrediction(null);
    setExplanation(null);
    setExplanationError("");

    if (customers.length > 0) {
      return;
    }

    setLoadingCustomers(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/demo/customers?limit=50`
      );

      if (!response.ok) {
        throw new Error(
          "Demo customers could not be loaded."
        );
      }

      const data = await response.json();

      setCustomers(data.customers);

      if (data.customers.length > 0) {
        setSelectedCustomer(
          String(data.customers[0])
        );
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Customer loading failed."
      );
    } finally {
      setLoadingCustomers(false);
    }
  }

  async function loadExplanation(customerId) {
    abortPendingExplanation();

    const controller = new AbortController();
    explanationRequestRef.current = controller;

    setLoadingExplanation(true);
    setExplanationError("");
    setExplanation(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/demo/explain/${customerId}`,
        {
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        const errorData =
          await response.json();

        throw new Error(
          errorData.detail ??
            "Explanation could not be loaded."
        );
      }

      const data = await response.json();

      // Only apply this response if a newer
      // explanation request hasn't since replaced it.
      if (
        explanationRequestRef.current ===
        controller
      ) {
        setExplanation(data);
      }
    } catch (err) {
      // Aborted requests (e.g. the customer changed
      // before this one finished) are not user-facing
      // errors.
      if (err?.name === "AbortError") {
        return;
      }

      if (
        explanationRequestRef.current ===
        controller
      ) {
        setExplanationError(
          err instanceof Error
            ? err.message
            : "Explanation failed."
        );
      }
    } finally {
      if (
        explanationRequestRef.current ===
        controller
      ) {
        setLoadingExplanation(false);
      }
    }
  }

  async function runPrediction() {
    if (!selectedCustomer) {
      return;
    }

    abortPendingExplanation();

    setPredicting(true);
    setPrediction(null);
    setExplanation(null);
    setExplanationError("");
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/demo/predict/${selectedCustomer}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        const errorData =
          await response.json();

        throw new Error(
          errorData.detail ??
            "Prediction request failed."
        );
      }

      const data = await response.json();

      setPrediction(data);

      // Explanation is only requested once a risk
      // score has been generated successfully.
      loadExplanation(selectedCustomer);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Prediction failed."
      );
    } finally {
      setPredicting(false);
    }
  }

  const oof =
    modelInfo?.stability_cv_oof ?? {};

  const riskAnalysisPanel = (
    <div className="panel prediction-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">
            NEW ANALYSIS
          </p>

          <h3>
            Customer Risk Analysis
          </h3>
        </div>

        <span className="research-badge">
          Decision Support
        </span>
      </div>

      <p className="panel-description">
        Select a customer from the demo
        dataset and generate a live
        CreditLens model risk score.
      </p>

      {!analysisOpen && (
        <div className="empty-analysis">
          <div className="empty-icon">
            ↗
          </div>

          <h4>
            No customer selected
          </h4>

          <p>
            Start a risk analysis to load
            customers and generate a live
            FastAPI prediction.
          </p>

          <button
            onClick={startAnalysis}
            disabled={
              apiStatus !== "ready"
            }
          >
            Start Risk Analysis
          </button>
        </div>
      )}

      {analysisOpen && (
        <div className="analysis-box">
          <div className="analysis-field">
            <label>
              Demo Customer
            </label>

            {loadingCustomers ? (
              <div className="loading-text">
                Loading customers...
              </div>
            ) : (
              <select
                value={selectedCustomer}
                onChange={(event) => {
                  abortPendingExplanation();

                  setSelectedCustomer(
                    event.target.value
                  );

                  setPrediction(null);
                  setExplanation(null);
                  setExplanationError("");
                }}
              >
                {customers.map(
                  (customerId) => (
                    <option
                      key={customerId}
                      value={customerId}
                    >
                      Customer #{customerId}
                    </option>
                  )
                )}
              </select>
            )}
          </div>

          {loadingProfile && (
            <div className="loading-text">
              Loading customer profile...
            </div>
          )}

          {profileError && (
            <div className="error-banner">
              {profileError}
            </div>
          )}

          {customerProfile && !loadingProfile && (
            <div className="profile-card">
              <p className="eyebrow">
                CUSTOMER PROFILE
              </p>

              <div className="profile-grid">
                <div className="profile-item">
                  <span>Annual Income</span>
                  <strong>
                    {formatCurrency(
                      customerProfile.income_total
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Credit Amount</span>
                  <strong>
                    {formatCurrency(
                      customerProfile.credit_amount
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Annuity</span>
                  <strong>
                    {formatCurrency(
                      customerProfile.annuity_amount
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Goods Price</span>
                  <strong>
                    {formatCurrency(
                      customerProfile.goods_price
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Age</span>
                  <strong>
                    {formatYears(
                      customerProfile.age_years
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>
                    Employment Duration
                  </span>
                  <strong>
                    {formatYears(
                      customerProfile.employment_years
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Education</span>
                  <strong>
                    {customerProfile.education_type ??
                      "—"}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Income Type</span>
                  <strong>
                    {customerProfile.income_type ??
                      "—"}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Family Status</span>
                  <strong>
                    {customerProfile.family_status ??
                      "—"}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>Bureau History</span>
                  <strong>
                    {formatHistory(
                      customerProfile.has_bureau_history
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>
                    Previous Applications
                  </span>
                  <strong>
                    {formatHistory(
                      customerProfile.has_previous_application_history
                    )}
                  </strong>
                </div>

                <div className="profile-item">
                  <span>
                    Installment History
                  </span>
                  <strong>
                    {formatHistory(
                      customerProfile.has_installment_history
                    )}
                  </strong>
                </div>
              </div>
            </div>
          )}

          <button
            className="predict-button"
            onClick={runPrediction}
            disabled={
              predicting ||
              loadingCustomers ||
              !selectedCustomer
            }
          >
            {predicting
              ? "Analyzing customer..."
              : "Generate Risk Score"}
          </button>

          {prediction && (
            <div className="prediction-result">
              <p className="eyebrow">
                MODEL RESULT
              </p>

              <div className="result-header">
                <div>
                  <span>
                    Customer
                  </span>

                  <strong>
                    #
                    {
                      prediction.SK_ID_CURR
                    }
                  </strong>
                </div>
              </div>

              <div className="score-gauge">
                <div className="gauge-label-row">
                  <span>
                    Model Risk Score
                  </span>

                  <strong className="gauge-value">
                    {prediction.risk_score.toFixed(
                      3
                    )}
                  </strong>
                </div>

                <div className="score-track">
                  <div
                    className="score-fill"
                    style={{
                      width: `${Math.min(
                        Math.max(
                          prediction.risk_score *
                            100,
                          0
                        ),
                        100
                      )}%`,
                    }}
                  />
                </div>

                <div className="gauge-scale">
                  <span>0.00</span>
                  <span>1.00</span>
                </div>

                <span
                  className={`score-band score-band-${
                    riskBand(
                      prediction.risk_score
                    ).key
                  }`}
                >
                  {
                    riskBand(
                      prediction.risk_score
                    ).label
                  }
                </span>
              </div>

              <p className="result-note">
                This is an uncalibrated model
                score, not a probability of
                default. The band above is a
                descriptive visualization grouping
                only — it is not a model
                validation threshold or a credit
                approval/rejection decision.
              </p>

              <p className="result-note">
                {
                  prediction.interpretation
                }
              </p>
            </div>
          )}

          {loadingExplanation && (
            <div className="loading-text">
              Loading explanation...
            </div>
          )}

          {explanationError &&
            !loadingExplanation && (
              <div className="error-banner">
                {explanationError}
              </div>
            )}

          {explanation &&
            !loadingExplanation && (
              <div className="explanation-card">
                <p className="eyebrow">
                  MODEL EXPLANATION
                </p>

                <h4>
                  Top Factors Influencing
                  This Score
                </h4>

                <div className="factor-list">
                  {explanation.top_features.map(
                    (item) => (
                      <div
                        className="factor-row"
                        key={item.feature}
                      >
                        <div className="factor-info">
                          <span className="factor-name">
                            {formatFeatureLabel(
                              item.feature
                            )}
                          </span>

                          <span className="factor-value">
                            value:{" "}
                            {formatFeatureValue(
                              item.value
                            )}
                          </span>
                        </div>

                        <div
                          className={`factor-direction factor-direction-${item.direction}`}
                        >
                          <span className="factor-arrow">
                            {directionSymbol(
                              item.direction
                            )}
                          </span>

                          <span>
                            {directionLabel(
                              item.direction
                            )}
                          </span>
                        </div>
                      </div>
                    )
                  )}
                </div>

                <p className="result-note">
                  {explanation.disclaimer}
                </p>
              </div>
            )}
        </div>
      )}
    </div>
  );

  const modelPanel = (
    <div className="panel model-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">
            MODEL
          </p>

          <h3>Research Model</h3>
        </div>
      </div>

      <div className="model-row">
        <span>Algorithm</span>

        <strong>
          {modelInfo?.model_type ??
            "Loading..."}
        </strong>
      </div>

      <div className="model-row">
        <span>Trees</span>

        <strong>
          {modelInfo?.iterations
            ? modelInfo.iterations.toLocaleString(
                "en-US"
              )
            : "—"}
        </strong>
      </div>

      <div className="model-row">
        <span>ROC-AUC</span>

        <strong>
          {typeof oof.roc_auc ===
          "number"
            ? oof.roc_auc.toFixed(4)
            : "—"}
        </strong>
      </div>

      <div className="model-row">
        <span>PR-AUC</span>

        <strong>
          {typeof oof.pr_auc ===
          "number"
            ? oof.pr_auc.toFixed(4)
            : "—"}
        </strong>
      </div>

      <div className="model-row">
        <span>
          Sensitive Feature
        </span>

        <strong className="excluded">
          {modelInfo
            ?.excluded_sensitive_feature
            ? `${modelInfo.excluded_sensitive_feature} excluded`
            : "—"}
        </strong>
      </div>

      <div className="notice">
        {modelInfo?.probability_note ??
          "Model output is an uncalibrated risk score and should not be interpreted as a probability of default."}
      </div>
    </div>
  );

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            C
          </div>

          <div>
            <h1>CreditLens AI</h1>
            <span>Risk Intelligence</span>
          </div>
        </div>

        <nav className="nav">
          <button
            className={`nav-item ${
              activeView === "dashboard"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setActiveView("dashboard")
            }
          >
            Dashboard
          </button>

          <button
            className={`nav-item ${
              activeView === "risk-analysis"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setActiveView("risk-analysis")
            }
          >
            Risk Analysis
          </button>

          <button
            className={`nav-item ${
              activeView === "model-information"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setActiveView("model-information")
            }
          >
            Model Information
          </button>

          <button
            className={`nav-item ${
              activeView === "fairness"
                ? "active"
                : ""
            }`}
            onClick={() =>
              setActiveView("fairness")
            }
          >
            Fairness
          </button>
        </nav>

        <div className="sidebar-footer">
          Research & Decision Support
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">
              {VIEW_META[activeView].eyebrow}
            </p>

            <h2>
              {VIEW_META[activeView].title}
            </h2>

            <p className="subtitle">
              {VIEW_META[activeView].subtitle}
            </p>
          </div>

          <div
            className={`status ${
              apiStatus === "offline"
                ? "status-offline"
                : ""
            }`}
          >
            <span
              className={`status-dot ${
                apiStatus === "offline"
                  ? "status-dot-offline"
                  : ""
              }`}
            />

            {apiStatus === "loading" &&
              "Connecting..."}

            {apiStatus === "ready" &&
              "API Ready"}

            {apiStatus === "offline" &&
              "API Offline"}
          </div>
        </header>

        {error &&
          (activeView === "dashboard" ||
            activeView === "risk-analysis") && (
            <div className="error-banner">
              {error}
            </div>
          )}

        {activeView === "dashboard" && (
          <>
        <section className="stats-grid">
          <div className="stat-card">
            <span>Primary Model</span>

            <strong>
              {modelInfo
                ? "CatBoost"
                : "—"}
            </strong>

            <small>
              Gender-free classifier
            </small>
          </div>

          <div className="stat-card">
            <span>Model Features</span>

            <strong>
              {modelInfo?.feature_count ??
                "—"}
            </strong>

            <small>
              {modelInfo
                ? `${modelInfo.numeric_feature_count} numeric · ${modelInfo.categorical_feature_count} categorical`
                : "Loading feature contract"}
            </small>
          </div>

          <div className="stat-card">
            <span>OOF ROC-AUC</span>

            <strong>
              {typeof oof.roc_auc ===
              "number"
                ? oof.roc_auc.toFixed(4)
                : "—"}
            </strong>

            <small>
              3-fold stratified validation
            </small>
          </div>

          <div className="stat-card">
            <span>
              Training Customers
            </span>

            <strong>
              {modelInfo?.training_rows
                ? modelInfo.training_rows.toLocaleString(
                    "en-US"
                  )
                : "—"}
            </strong>

            <small>
              Home Credit dataset
            </small>
          </div>
        </section>

        <section className="content-grid">
          {riskAnalysisPanel}
          {modelPanel}
        </section>
          </>
        )}

        {activeView === "risk-analysis" && (
          <section className="content-grid content-grid-single">
            {riskAnalysisPanel}
          </section>
        )}

        {activeView === "model-information" && (
          <section className="content-grid content-grid-single">
            <div className="panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">
                    MODEL CONTRACT
                  </p>

                  <h3>Model Information</h3>
                </div>
              </div>

              {apiStatus === "loading" && (
                <div className="loading-text">
                  Loading model information...
                </div>
              )}

              {apiStatus === "offline" && (
                <div className="error-banner">
                  {error ||
                    "Model information could not be loaded."}
                </div>
              )}

              {apiStatus === "ready" &&
                modelInfo && (
                  <>
                    <div className="profile-grid">
                      <div className="profile-item">
                        <span>Model Name</span>
                        <strong>
                          {modelInfo.model_name ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Model Type</span>
                        <strong>
                          {modelInfo.model_type ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Training Rows</span>
                        <strong>
                          {typeof modelInfo.training_rows ===
                          "number"
                            ? modelInfo.training_rows.toLocaleString(
                                "en-US"
                              )
                            : "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Feature Count</span>
                        <strong>
                          {modelInfo.feature_count ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>
                          Numeric Feature Count
                        </span>
                        <strong>
                          {modelInfo.numeric_feature_count ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>
                          Categorical Feature Count
                        </span>
                        <strong>
                          {modelInfo.categorical_feature_count ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>
                          Excluded Sensitive Feature
                        </span>
                        <strong className="excluded">
                          {modelInfo.excluded_sensitive_feature ??
                            "—"}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Iterations</span>
                        <strong>
                          {typeof modelInfo.iterations ===
                          "number"
                            ? modelInfo.iterations.toLocaleString(
                                "en-US"
                              )
                            : "—"}
                        </strong>
                      </div>
                    </div>

                    <p className="section-subheading">
                      Development Validation
                    </p>

                    <div className="profile-grid">
                      <div className="profile-item">
                        <span>ROC-AUC</span>
                        <strong>
                          {formatMetric(
                            modelInfo
                              .development_validation
                              ?.roc_auc
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>PR-AUC</span>
                        <strong>
                          {formatMetric(
                            modelInfo
                              .development_validation
                              ?.pr_auc
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>F1</span>
                        <strong>
                          {formatMetric(
                            modelInfo
                              .development_validation
                              ?.f1_at_0_50
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Recall</span>
                        <strong>
                          {formatMetric(
                            modelInfo
                              .development_validation
                              ?.recall_at_0_50
                          )}
                        </strong>
                      </div>
                    </div>

                    <p className="section-subheading">
                      OOF / Stability Validation
                    </p>

                    <div className="profile-grid">
                      <div className="profile-item">
                        <span>ROC-AUC</span>
                        <strong>
                          {formatMetric(
                            oof.roc_auc
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>PR-AUC</span>
                        <strong>
                          {formatMetric(
                            oof.pr_auc
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>F1</span>
                        <strong>
                          {formatMetric(
                            oof.f1_at_0_50
                          )}
                        </strong>
                      </div>

                      <div className="profile-item">
                        <span>Recall</span>
                        <strong>
                          {formatMetric(
                            oof.recall_at_0_50
                          )}
                        </strong>
                      </div>
                    </div>

                    <div className="notice">
                      {modelInfo.probability_note ??
                        "Model output is an uncalibrated risk score and should not be interpreted as a probability of default."}
                      {" "}The predictive feature
                      set excludes{" "}
                      {modelInfo.excluded_sensitive_feature ??
                        "the sensitive feature"}{" "}
                      (gender-free predictive
                      inputs). The model uses a
                      class-balanced training
                      strategy and its output is a
                      risk_score, not a decision.
                    </div>
                  </>
                )}
            </div>
          </section>
        )}

        {activeView === "fairness" && (
          <section className="content-grid content-grid-single">
            <div className="panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">
                    POST-HOC AUDIT
                  </p>

                  <h3>Fairness Audit</h3>
                </div>
              </div>

              <p className="panel-description">
                CODE_GENDER is excluded from
                predictive inputs and is used
                only for post-hoc auditing.
              </p>

              {loadingFairness && (
                <div className="loading-text">
                  Loading fairness report...
                </div>
              )}

              {fairnessError &&
                !loadingFairness && (
                  <div className="error-banner">
                    {fairnessError}
                  </div>
                )}

              {fairnessSummary &&
                !loadingFairness && (
                  <>
                    <div className="fairness-groups">
                      {fairnessSummary.groups.map(
                        (group) => (
                          <div
                            className="fairness-group-card"
                            key={group.group}
                          >
                            <div className="fairness-group-header">
                              <span className="fairness-group-name">
                                Group{" "}
                                {group.group}
                              </span>

                              <span className="fairness-group-count">
                                {group.sample_count.toLocaleString(
                                  "en-US"
                                )}{" "}
                                samples
                              </span>
                            </div>

                            <div className="fairness-metric-row">
                              <span>
                                Actual Positive Rate
                              </span>
                              <strong>
                                {formatPercent(
                                  group.actual_positive_rate
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>
                                Selection Rate
                              </span>
                              <strong>
                                {formatPercent(
                                  group.selection_rate
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>Precision</span>
                              <strong>
                                {formatPercent(
                                  group.precision
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>Recall</span>
                              <strong>
                                {formatPercent(
                                  group.recall
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>
                                False Positive Rate
                              </span>
                              <strong>
                                {formatPercent(
                                  group.false_positive_rate
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>
                                False Negative Rate
                              </span>
                              <strong>
                                {formatPercent(
                                  group.false_negative_rate
                                )}
                              </strong>
                            </div>

                            <div className="fairness-metric-row">
                              <span>ROC-AUC</span>
                              <strong>
                                {formatMetric(
                                  group.roc_auc
                                )}
                              </strong>
                            </div>
                          </div>
                        )
                      )}
                    </div>

                    <div className="fairness-gap-card">
                      <p className="eyebrow">
                        MAJOR-GROUP GAPS
                      </p>

                      <h4>
                        Diagnostic group
                        differences
                      </h4>

                      <div className="profile-grid">
                        <div className="profile-item">
                          <span>Recall Gap</span>
                          <strong>
                            {formatPercent(
                              fairnessSummary.gaps
                                .recall
                            )}
                          </strong>
                        </div>

                        <div className="profile-item">
                          <span>
                            False Positive Rate Gap
                          </span>
                          <strong>
                            {formatPercent(
                              fairnessSummary.gaps
                                .false_positive_rate
                            )}
                          </strong>
                        </div>

                        <div className="profile-item">
                          <span>
                            False Negative Rate Gap
                          </span>
                          <strong>
                            {formatPercent(
                              fairnessSummary.gaps
                                .false_negative_rate
                            )}
                          </strong>
                        </div>

                        <div className="profile-item">
                          <span>ROC-AUC Gap</span>
                          <strong>
                            {formatMetric(
                              fairnessSummary.gaps
                                .roc_auc
                            )}
                          </strong>
                        </div>
                      </div>
                    </div>

                    {fairnessSummary
                      .excluded_groups &&
                      fairnessSummary
                        .excluded_groups.length >
                        0 && (
                        <p className="result-note">
                          {fairnessSummary.excluded_groups
                            .map(
                              (group) =>
                                `Group ${group.group} (n=${group.sample_count})`
                            )
                            .join(", ")}{" "}
                          excluded from the
                          headline comparison
                          above due to small
                          sample size.
                        </p>
                      )}

                    <p className="result-note">
                      {fairnessSummary.note}
                    </p>
                  </>
                )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
