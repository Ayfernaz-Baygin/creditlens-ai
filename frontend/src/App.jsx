import { useEffect, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [apiStatus, setApiStatus] = useState("loading");
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState("");

  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [loadingCustomers, setLoadingCustomers] =
    useState(false);

  const [predicting, setPredicting] =
    useState(false);

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

  async function startAnalysis() {
    setAnalysisOpen(true);
    setPrediction(null);

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

  async function runPrediction() {
    if (!selectedCustomer) {
      return;
    }

    setPredicting(true);
    setPrediction(null);
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
          <button className="nav-item active">
            Dashboard
          </button>

          <button className="nav-item">
            Risk Analysis
          </button>

          <button className="nav-item">
            Model Information
          </button>

          <button className="nav-item">
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
              CREDIT RISK PLATFORM
            </p>

            <h2>Dashboard</h2>

            <p className="subtitle">
              Explainable machine learning for
              credit risk assessment.
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

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

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
                        setSelectedCustomer(
                          event.target.value
                        );

                        setPrediction(null);
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
                    ? "Analyzing..."
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

                      <div className="score-block">
                        <span>
                          Risk Score
                        </span>

                        <strong>
                          {prediction.risk_score.toFixed(
                            6
                          )}
                        </strong>
                      </div>
                    </div>

                    <div className="score-track">
                      <div
                        className="score-fill"
                        style={{
                          width: `${
                            prediction.risk_score *
                            100
                          }%`,
                        }}
                      />
                    </div>

                    <p className="result-note">
                      {
                        prediction.interpretation
                      }
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

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
        </section>
      </main>
    </div>
  );
}

export default App;