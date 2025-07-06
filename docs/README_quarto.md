# Hedge-Forge Quarto Documentation

This repository contains the **Quarto documentation** for the Hedge-Forge portfolio optimization project.

The goal of this documentation is to produce **client-facing, professional reports** suitable for:

- Technical stakeholders
- Hiring managers
- Clients interested in understanding portfolio optimization methods and results

---

## 📂 Project File Structure

The documentation files are organized as follows:

```
hedge_forge/
├── docs/                          # Project documentation via Quarto
│   ├── index.qmd                  # Executive summary / overview
│   ├── methodology.qmd            # Stochastic modeling, math background
│   ├── data_pipeline.qmd          # Data collection, cleaning, and processing steps
│   ├── eda.qmd                    # Exploratory Data Analysis
│   ├── optimization_results.qmd   # Output, charts, interpretations
│   ├── risk_metrics.qmd           # Definitions and calculations for risk metrics
│   ├── appendix.qmd               # Additional formulas, derivations, references
│   └── _quarto.yml                # Quarto configuration for site structure and appearance
```

Below is a brief description of what each file contains.

---

## 📄 Documentation Pages

### `index.qmd`

- Provides an overview of the Hedge-Forge project
- Summarizes the business problem being solved
- High-level architecture diagrams
- Quick summary of results for non-technical audiences

---

### `methodology.qmd`

- Mathematical background of portfolio optimization:
  - Mean-variance optimization
  - Sharpe ratio maximization
  - Portfolio constraints (e.g. sector limits, no shorting)
- Definitions of risk metrics:
  - Volatility
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
- Explanations of why these methods are chosen
- LaTeX equations for mathematical clarity

---

### `data_pipeline.qmd`

- Details about the project’s data sources:
  - Market data APIs
  - Synthetic data generation
  - CSV data sources
- Data validation and cleaning steps:
  - Handling missing values
  - Detecting and managing outliers
  - Ensuring data consistency and correct types
- Flow diagrams illustrating the pipeline
- Descriptions of utility functions:
  - `load_data()`
  - `validate_data()`
  - `compute_log_returns()`
  - `compute_covariance_matrix()`
  - `compute_rolling_metrics()`

---

### `eda.qmd`

- Results of the Exploratory Data Analysis:
  - Return distributions
  - Correlation heatmaps
  - Rolling volatility trends
- Observations on data characteristics:
  - Skewness
  - Kurtosis
  - Market behavior anomalies
- Visualizations to illustrate insights before modeling

---

### `optimization_results.qmd`

- Description of optimization objectives:
  - Minimum variance portfolio
  - Maximum Sharpe ratio
- Explanation of constraints applied during optimization
- Presentation of optimized portfolios:
  - Weight allocations
  - Expected returns
  - Portfolio volatility
- Charts and plots:
  - Efficient frontier
  - Comparison between strategies
- Interpretation of results for stakeholders

---

### `risk_metrics.qmd`

- Definitions and explanations of risk metrics:
  - Volatility
  - Value at Risk (VaR)
  - Conditional VaR (CVaR)
  - Tracking error
- Risk calculations applied to optimized portfolios
- Visualizations and tables summarizing risk statistics

---

### `appendix.qmd`

- Detailed mathematical derivations
- Extended formulas and calculations not shown in main sections
- Troubleshooting notes and assumptions
- References to research papers, textbooks, and external resources

---

### `_quarto.yml`

- Quarto configuration file for:
  - Site navigation structure
  - Page titles and links
  - Theme and styling (e.g. colors, fonts)
  - Output formats (HTML, PDF, etc.)
- Ensures the documentation builds into a cohesive multi-page website or PDF report

---

## 💡 Why Use Quarto?

The Hedge-Forge documentation uses Quarto because it:

✅ Combines narrative text, math equations, code, and visuals seamlessly  
✅ Produces clean, professional HTML and PDF reports  
✅ Handles LaTeX equations for financial modeling  
✅ Supports multi-page documentation with logical navigation  
✅ Works well for both internal and client-facing deliverables

---

## 🚀 How to Build and View the Documentation

Follow these steps to build and view the Hedge-Forge documentation locally.

### 1. Install Quarto

Download and install Quarto from:

[https://quarto.org/docs/get-started/](https://quarto.org/docs/get-started/)

---

### 2. Navigate to the Project Directory

From your terminal:

```bash
cd hedge_forge/docs
```

---

### 3. Render the Documentation

To build the entire documentation site, run:

```bash
quarto render
```

This generates the HTML website in the `_site/` directory.

---

### 4. Preview the Documentation Locally

For live preview with auto-reload:

```bash
quarto preview
```

By default, this runs a local web server at:

```
http://localhost:4200
```

Open that URL in your web browser to browse the documentation.

---

### 5. Open the Site Manually

Alternatively, open the main HTML file directly:

- Navigate to:

    ```
    hedge_forge/docs/_site/index.html
    ```

- Double-click `index.html` or open it in your browser of choice.

---

### Optional: Build PDF Version

To generate a PDF report instead of HTML:

1. Add the following to `_quarto.yml`:

    ```yaml
    format:
      pdf:
        documentclass: article
    ```

2. Run:

    ```bash
    quarto render --to pdf
    ```

This produces a single PDF file with all pages combined.

---

## 🤝 Contributing

If you wish to help improve the documentation:

- Edit the relevant `.qmd` files
- Re-render the documentation using `quarto render`
- Submit changes via a pull request

---

**Hedge-Forge** aims to be a flagship-level fintech project. This Quarto documentation helps present the project in a professional, clear, and client-friendly format.

---
