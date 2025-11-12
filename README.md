# Anti-Money Laundering (AML) Transaction Monitoring
## XXXX Bank - SAR Prediction Analysis

---

## 📋 Project Overview

This project develops machine learning models to improve **XXXX Bank's** Anti-Money Laundering (AML) transaction monitoring system. The goal is to predict Suspicious Activity Reports (SARs) more effectively than the current rule-based approach.

### Business Context
- **Bank**: XXXX (Nordic bank, 1,000 customers, 1 year old)
- **Current Rule**: Investigate customers with turnover > €9,000/month
- **Problem**: Missing suspicious customers, inefficient investigations
- **Regulatory Requirement**: FSA mandates better transaction monitoring

### Objectives
1. ✅ **Find more suspicious customers (SARs)** - Improve detection rate
2. ✅ **Reduce False Positives** - Minimize unnecessary investigations  
3. ✅ **Create sustainable controls** - Justifiable and robust monitoring

---

## 📊 Data Description

### Files in `data/` folder:

| File | Description | Records |
|------|-------------|---------|
| `df_kyc.csv` | Customer background information | 1,000 customers |
| `df_transactions.csv` | All customer transactions | ~165,000 transactions |
| `df_label.csv` | SAR filings per customer-month | 12,000 rows (1000×12 months) |

### Key Features:
- **KYC Data**: Age, Sex, High Risk Country, Vulnerable Area, Payment Intentions
- **Transaction Data**: Transaction value, counterparty, type, month
- **Labels**: Binary SAR flag (0=No SAR, 1=SAR Filed)

---

## � Statistical Testing Framework for Feature Selection

### Theoretical Foundation

A critical component of this AML analysis is the **rigorous statistical feature selection framework** that determines which features have a genuine relationship with SAR outcomes versus those that might appear significant due to chance.

### Why Statistical Testing Matters in AML

In Anti-Money Laundering contexts:
1. **Regulatory Scrutiny**: Models must be explainable and defensible to regulators
2. **False Positives**: Weak features increase investigation burden without value
3. **Model Robustness**: Statistical significance ensures features generalize beyond training data
4. **Resource Allocation**: Compliance teams need confidence in which signals matter most

### Theoretical Approach: Multi-Test Framework

Our framework applies different statistical tests based on feature types, following established statistical methodology:

#### 1. **Continuous/Interval Features** (Transaction amounts, counts, ratios)

**Statistical Tests Applied:**
- **Independent Samples T-Test** (parametric)
  - **When Used**: Data follows normal distribution
  - **Null Hypothesis (H₀)**: μ₁ = μ₂ (means are equal between SAR and non-SAR groups)
  - **Theory**: Assumes normally distributed data with homogeneous variance
  - **Test Statistic**: t = (X̄₁ - X̄₂) / SE_difference
  
- **Mann-Whitney U Test** (non-parametric)
  - **When Used**: Data is non-normal or has outliers
  - **Null Hypothesis (H₀)**: Distributions are identical between groups
  - **Theory**: Rank-based test, doesn't assume normality
  - **Advantage**: Robust to outliers and skewed distributions

**Effect Size Metric: Cohen's d**
- **Formula**: d = (μ₁ - μ₂) / σ_pooled
- **Interpretation**:
  - d < 0.2: Negligible practical difference
  - 0.2 ≤ d < 0.5: Small effect
  - 0.5 ≤ d < 0.8: Medium effect
  - d ≥ 0.8: Large effect (strong practical significance)
- **Why Important**: P-value tells us IF there's a difference; Cohen's d tells us HOW MUCH

#### 2. **Binary/Categorical Features** (Risk flags, customer attributes)

**Statistical Test Applied:**
- **Chi-Square Test of Independence (χ²)**
  - **Null Hypothesis (H₀)**: No association between feature and SAR outcome
  - **Theory**: Compares observed frequencies to expected frequencies
  - **Test Statistic**: χ² = Σ[(O - E)² / E]
  - **Assumptions**: 
    - Expected frequency ≥ 5 for each cell
    - Independent observations
    - Mutually exclusive categories

**Effect Size Metric: Cramér's V**
- **Formula**: V = √(χ² / (n × min(r-1, c-1)))
- **Interpretation**:
  - V < 0.1: Negligible association
  - 0.1 ≤ V < 0.3: Small effect
  - 0.3 ≤ V < 0.5: Medium effect
  - V ≥ 0.5: Large effect (strong association)
- **Why Important**: Chi-square significance depends on sample size; Cramér's V standardizes effect size

#### 3. **Multiple Testing Correction: Bonferroni Adjustment**

**The Problem:**
- Testing 50 features at α = 0.05 means ~2.5 false positives expected by chance
- Type I Error Rate inflates with multiple tests
- Risk of including spurious features

**The Solution: Bonferroni Correction**
- **Adjusted Significance Level**: α_corrected = α / n_tests
- **Example**: Testing 50 features → α_corrected = 0.05/50 = 0.001
- **Trade-off**: More conservative (reduces false positives but may miss weak signals)
- **Implementation**: Create two feature tiers (Bonferroni-corrected vs. uncorrected)

**Mathematical Foundation:**
- **Family-Wise Error Rate (FWER)**: Probability of at least one Type I error
- Bonferroni ensures: FWER ≤ α across all tests
- Formula: P(at least one false positive) ≤ 1 - (1-α)^n ≈ nα for small α

### Guidelines for Applying the Framework

#### Step 1: Feature Type Classification
```
FOR each feature in dataset:
    IF unique_values == 2:
        → Classify as BINARY
    ELSE IF unique_values ≤ 10 AND dtype is categorical:
        → Classify as CATEGORICAL
    ELSE:
        → Classify as CONTINUOUS
```

#### Step 2: Test Selection Logic
```
IF feature is CONTINUOUS:
    Perform normality test (Shapiro-Wilk)
    IF data is normal:
        → Apply Independent Samples T-Test
        → Calculate Cohen's d
    ELSE:
        → Apply Mann-Whitney U Test
        → Calculate rank-biserial correlation (or Cohen's d approximation)

IF feature is BINARY or CATEGORICAL:
    Create contingency table
    Check expected frequencies (all ≥ 5)
    IF assumptions met:
        → Apply Chi-Square Test
        → Calculate Cramér's V
    ELSE:
        → Apply Fisher's Exact Test (small samples)
```

#### Step 3: Significance Interpretation
```
FOR each test result:
    Calculate p-value
    Compare to α = 0.05
    Compare to α_bonferroni = 0.05 / n_tests
    
    IF p < α_bonferroni:
        → TIER 1: Highly significant (robust to multiple testing)
    ELSE IF p < α:
        → TIER 2: Significant but needs validation
    ELSE:
        → Consider excluding (not statistically significant)
```

#### Step 4: Effect Size Evaluation
```
FOR each significant feature:
    IF effect_size is Large:
        → HIGH PRIORITY: Strong practical significance
    ELSE IF effect_size is Medium:
        → MODERATE PRIORITY: Meaningful practical difference
    ELSE IF effect_size is Small/Negligible:
        → CAUTION: May be statistically significant but not practically useful
```

### Key Concepts and Definitions

#### Statistical Significance vs. Practical Significance
- **Statistical Significance (p-value < α)**: 
  - Indicates relationship is unlikely due to chance
  - Influenced by sample size (large samples detect tiny effects)
  - Answers: "Is there an effect?"

- **Practical Significance (effect size)**:
  - Indicates magnitude of the relationship
  - Independent of sample size
  - Answers: "How large is the effect?"
  - More important for business decisions

**Example in AML Context:**
```
Feature: Age difference between SAR and non-SAR customers
- Statistical result: p < 0.001 (highly significant)
- Effect size: Cohen's d = 0.05 (negligible)
- Interpretation: Difference exists but is too small to be useful
- Decision: May not warrant inclusion in model despite significance
```

#### Type I and Type II Errors in AML Context

**Type I Error (False Positive in Testing)**
- **Definition**: Concluding a feature is significant when it's not
- **AML Impact**: Including useless features → noisy models, false alerts
- **Control**: Bonferroni correction, lower α threshold
- **Cost**: Wasted compliance resources investigating false signals

**Type II Error (False Negative in Testing)**
- **Definition**: Missing a truly significant feature
- **AML Impact**: Overlooking valuable signals → missed SARs
- **Control**: Adequate sample size, appropriate test selection
- **Cost**: Regulatory fines, reputational damage, undetected crime

**Balance Strategy:**
- Use two-tier approach (strict Bonferroni + moderate p<0.05)
- Prioritize TIER 1 for production models
- Test TIER 2 in validation to assess incremental value

#### Assumptions and Validation

**For T-Tests:**
1. **Independence**: Each observation is independent
   - ✓ Valid: Different customer-months are independent
   - ✗ Invalid: Same customer across months (consider mixed models)

2. **Normality**: Data should be approximately normal
   - Test with: Shapiro-Wilk, Q-Q plots, histograms
   - Remedy: Use Mann-Whitney U if violated

3. **Homogeneity of Variance**: Equal variances between groups
   - Test with: Levene's test, F-test
   - Remedy: Welch's t-test if violated

**For Chi-Square Tests:**
1. **Expected Frequency**: All cells should have E ≥ 5
   - Check: Calculate expected frequencies
   - Remedy: Combine categories or use Fisher's Exact Test

2. **Independence**: Each observation in one category only
   - Ensure: Proper data structure, no double-counting

3. **Random Sampling**: Data should be randomly sampled
   - Consider: Potential selection biases in SAR reporting

### Feature Selection Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE EVALUATION                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────────┐
                    │  Run Statistical │
                    │      Test        │
                    └──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │   p < α_bonf?    │
                    └──────────────────┘
                         ↙        ↘
                      YES          NO
                       ↓            ↓
                  ┌────────┐   ┌──────────┐
                  │ TIER 1 │   │ p < 0.05?│
                  │INCLUDE │   └──────────┘
                  └────────┘      ↙    ↘
                               YES      NO
                                ↓        ↓
                           ┌────────┐ ┌──────────┐
                           │ TIER 2 │ │ EXCLUDE  │
                           │CONSIDER│ │(No signal)│
                           └────────┘ └──────────┘
                                ↓
                    ┌──────────────────────┐
                    │ Check Effect Size    │
                    └──────────────────────┘
                         ↙      ↓      ↘
                    Large    Medium   Small
                       ↓        ↓        ↓
                  Priority  Include   Caution
                   Feature  Feature  (Low value)
```

### Implementation Results in This Project

**Features Tested**: ~30 engineered features
- Continuous features: Transaction amounts, counts, ratios, turnover
- Binary features: Risk flags, behavioral mismatches, KYC declarations

**Statistical Rigor Applied**:
1. ✅ Automatic normality testing (Shapiro-Wilk)
2. ✅ Appropriate test selection (T-Test vs Mann-Whitney U)
3. ✅ Effect size calculations (Cohen's d, Cramér's V)
4. ✅ Bonferroni correction for multiple testing
5. ✅ Two-tier significance classification

**Key Findings** (Run notebook for actual results):
- TIER 1 Features: High-confidence features passing Bonferroni correction
- TIER 2 Features: Moderate-confidence features (p < 0.05)
- Effect Size Analysis: Identifies features with practical significance
- Behavioral Mismatch Features: Expected to show strong associations with SAR

**Recommended Reading**:
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità

---

## � Hypothesis Testing for Risk Factor Validation

### Why Hypothesis Testing is Necessary in AML

In Anti-Money Laundering systems, we cannot simply assume that commonly cited risk factors (e.g., High-Risk Countries, vulnerable areas, specific transaction patterns) actually predict suspicious activity. **Hypothesis testing provides scientific evidence** to validate or refute these assumptions.

**Critical Reasons for Hypothesis Testing:**

1. **Regulatory Justification**
   - Regulators require **evidence-based** risk assessments, not assumptions
   - Models must be defensible: "Why did you include this feature?"
   - Hypothesis testing provides statistical proof of risk factor validity

2. **Resource Optimization**
   - Compliance teams have limited capacity
   - Testing identifies which risk factors are **truly predictive**
   - Avoids wasting resources on ineffective indicators

3. **Avoiding Bias and Discrimination**
   - Risk factors based on country, demographics, etc., must be **statistically justified**
   - Prevents discriminatory practices based on stereotypes
   - Protects the bank from legal and reputational risk

4. **Model Performance**
   - Including non-predictive features adds noise to ML models
   - Hypothesis testing separates signal from noise
   - Improves model accuracy and generalization

5. **Transparency and Explainability**
   - Statistical tests provide clear, objective evidence
   - P-values and effect sizes are universally understood metrics
   - Facilitates communication with non-technical stakeholders

### Theoretical Framework: Testing Risk Factor Associations

#### Example: "High-Risk Country Customers are More Likely to Have SARs"

This is a classic hypothesis testing scenario in AML contexts.

**Step 1: Formulate Hypotheses**

- **H₀ (Null Hypothesis)**: There is NO association between High-Risk Country status and SAR filing
  - Mathematically: P(SAR | High-Risk) = P(SAR | Non-High-Risk)
  - Assumption: Country risk status does not predict suspicious activity

- **H₁ (Alternative Hypothesis)**: There IS an association between High-Risk Country status and SAR filing
  - Mathematically: P(SAR | High-Risk) ≠ P(SAR | Non-High-Risk)
  - Claim: Country risk status predicts suspicious activity

**Important: Two-Tailed vs. One-Tailed Testing**

We use **two-tailed tests** because:
- We test for **any difference**, not just "higher" risk
- More scientifically rigorous and conservative
- Detects unexpected findings (e.g., if High-Risk countries have *lower* SARs due to enhanced screening)
- Regulatory and peer-review standards favor two-tailed approaches
- Chi-Square test is inherently two-tailed (no directional test possible)

**Step 2: Select Appropriate Statistical Test**

For binary variables (High-Risk: Yes/No, SAR: Yes/No), use:

##### A. Chi-Square Test of Independence (χ²)

**When to Use:**
- Both variables are categorical (binary or multinomial)
- Testing association/independence between two variables
- Sample size is sufficient (expected frequency ≥ 5 in all cells)

**Theory:**
- Compares **observed frequencies** to **expected frequencies** (if no association)
- Test Statistic: χ² = Σ[(O - E)² / E]
  - O = Observed frequency in each cell
  - E = Expected frequency = (row total × column total) / grand total
- Larger χ² indicates greater deviation from independence

**Null Hypothesis:** Variables are independent (no association)

**Degrees of Freedom:** df = (rows - 1) × (columns - 1)

**Decision Rule:** If p-value < α (typically 0.05), reject H₀

**Example Contingency Table:**
```
                    No SAR    SAR     Total
Non-High-Risk       9,500     500    10,000
High-Risk           1,800     200     2,000
Total              11,300     700    12,000
```

**Interpretation:**
- χ² = 15.67, p = 0.0001 → Reject H₀
- Strong evidence of association between High-Risk Country and SAR

##### B. Two-Proportion Z-Test

**When to Use:**
- Directly compare two proportions (SAR rates between groups)
- Large sample sizes (np > 5 and n(1-p) > 5 for both groups)
- Provides directional information

**Theory:**
- Tests if two population proportions are equal
- Under H₀, proportions are equal: p₁ = p₂
- Uses pooled proportion: p̂ = (x₁ + x₂) / (n₁ + n₂)

**Test Statistic:**
```
Z = (p₁ - p₂) / SE

where SE = √[p̂(1 - p̂)(1/n₁ + 1/n₂)]
```

**Decision Rule:**
- Two-tailed: If |Z| > 1.96, reject H₀ (at α = 0.05)
- P-value = 2 × P(Z > |z_observed|)

**Advantage:** More intuitive interpretation of difference in proportions

**Step 3: Calculate Effect Size Metrics**

P-values tell us **IF** there's a difference; effect sizes tell us **HOW MUCH**.

##### Cramér's V (for Chi-Square)

**Formula:**
```
V = √(χ² / (n × min(r-1, c-1)))

where:
- n = total sample size
- r = number of rows
- c = number of columns
```

**Interpretation:**
- V < 0.1: Negligible association
- 0.1 ≤ V < 0.3: Small association
- 0.3 ≤ V < 0.5: Medium association
- V ≥ 0.5: Large association

**Why Important:** Chi-square is sensitive to sample size; Cramér's V standardizes the effect

##### Relative Risk (RR)

**Formula:**
```
RR = P(SAR | High-Risk) / P(SAR | Non-High-Risk)
```

**Interpretation:**
- RR = 1.0: No difference in risk
- RR > 1.0: High-Risk customers have HIGHER SAR probability
- RR < 1.0: High-Risk customers have LOWER SAR probability
- RR = 2.5: High-Risk customers are 2.5× more likely to have SARs

**Business Value:** Most intuitive metric for stakeholders

**95% Confidence Interval:**
- Calculated using logarithmic transformation
- If CI excludes 1.0, the difference is statistically significant
- Example: RR = 2.5, 95% CI [2.1, 2.9] → Significant difference

##### Odds Ratio (OR)

**Formula:**
```
OR = [P(SAR|High-Risk) / P(No SAR|High-Risk)] / [P(SAR|Non-High-Risk) / P(No SAR|Non-High-Risk)]

Simplified: OR = (a×d) / (b×c) from 2×2 table
```

**Interpretation:**
- OR = 1.0: No difference in odds
- OR > 1.0: Higher odds of SAR in High-Risk group
- OR < 1.0: Lower odds of SAR in High-Risk group

**When to Use:**
- Logistic regression contexts
- Case-control studies
- When outcome is rare (SAR rate < 10%), OR ≈ RR

**RR vs. OR:**
- RR: Compares probabilities (more intuitive)
- OR: Compares odds (used in regression models)
- For rare outcomes, RR and OR are similar
- For common outcomes, OR overestimates RR

**Step 4: Interpret Results and Make Decisions**

**Decision Framework:**
```
┌─────────────────────────────────────────────────────────┐
│             HYPOTHESIS TEST INTERPRETATION              │
└─────────────────────────────────────────────────────────┘
                         ↓
              ┌──────────────────┐
              │  p-value < 0.05? │
              └──────────────────┘
                   ↙         ↘
                YES           NO
                 ↓             ↓
        ┌──────────────┐  ┌────────────────┐
        │  SIGNIFICANT │  │ NOT SIGNIFICANT│
        │  Association │  │ No Evidence    │
        └──────────────┘  └────────────────┘
                 ↓             ↓
        ┌──────────────┐  ┌────────────────┐
        │ Check Effect │  │ DO NOT include │
        │    Size      │  │ feature in     │
        └──────────────┘  │ priority list  │
                 ↓         └────────────────┘
         ┌───────┴────────┐
         ↓                ↓
    Large/Medium      Small/Negligible
         ↓                ↓
    HIGH PRIORITY    Consider excluding
    Include feature  (statistically sig.
    in ML model     but not practical)
```

**Best Practice Recommendation**

✅ Keep the two-tailed approach for the main hypothesis test

✅ Report the directionality using descriptive statistics:

**Example Interpretation:**

**Scenario A: Confirmed Risk Factor**
- χ² = 45.3, p < 0.001, V = 0.28 (small-medium effect)
- RR = 2.8, 95% CI [2.4, 3.3]

- **Conclusion:** High-Risk Country is a valid SAR predictor
    -   "High-Risk Country customers have a SAR rate of X% vs. Y% for non-High-Risk"
    -   "RR = 2.8 means High-Risk customers are 2.8x more likely to have SARs"
- **Action:** Include in model, allocate enhanced monitoring resources



"High-Risk Country customers have a SAR rate of X% vs. Y% for non-High-Risk"
"RR = 2.5 means High-Risk customers are 2.5x more likely to have SARs"

**Scenario B: Unexpected Finding**
- χ² = 32.1, p < 0.001, V = 0.24 (small effect)
- RR = 0.6, 95% CI [0.5, 0.7] (LOWER risk in High-Risk countries)
- **Conclusion:** High-Risk Country customers have fewer SARs
- **Action:** Investigate why (effective screening? under-reporting?)

**Scenario C: No Evidence**
- χ² = 2.1, p = 0.15, V = 0.06 (negligible)
- RR = 1.1, 95% CI [0.9, 1.4]
- **Conclusion:** No statistical evidence of association
- **Action:** Do not prioritize this feature; focus on transaction behaviors

### Practical Implementation in This Project

**Hypothesis Tests Conducted:**
1. ✅ High-Risk Country → SAR association
2. ✅ Vulnerable Area → SAR association
3. ✅ Cash Mismatch → SAR association
4. ✅ International Payment Mismatch → SAR association
5. ✅ Monthly SAR distribution (temporal patterns)
6. ✅ All continuous features (T-test/Mann-Whitney U)

**Outputs Generated:**
- Contingency tables with observed/expected frequencies
- Statistical test results (χ², Z, p-values)
- Effect sizes (Cramér's V, Cohen's d, RR, OR)
- 95% Confidence intervals
- Visual comparisons (heatmaps, bar charts, summary boxes)
- Business interpretations and recommendations

**Model Impact:**
- Features passing hypothesis tests → TIER 1 (high priority)
- Features with p < 0.05 but weak effect → TIER 2 (consider)
- Features failing tests → excluded or deprioritized

### Statistical Assumptions and Validation

**For Chi-Square Test:**
1. **Independence of Observations**
   - Each customer-month record is independent
   - Valid: Different customers are independent
   - Caution: Same customer across months (consider mixed models if needed)

2. **Expected Frequency Rule**
   - All cells should have expected frequency ≥ 5
   - Check: Calculate expected frequencies before testing
   - Solution: Combine categories or use Fisher's Exact Test if violated

3. **Random Sampling**
   - Data should represent the population
   - Consider: Potential selection bias in SAR reporting

**For Two-Proportion Z-Test:**
1. **Large Sample Approximation**
   - np > 5 and n(1-p) > 5 for both groups
   - Use exact tests for small samples

2. **Independent Samples**
   - Groups must be independent (not paired)

### Why Two-Tailed Tests Are Standard

**Question:** Why not use one-tailed tests if we hypothesize High-Risk customers have *higher* SARs?

**Answer:**

1. **Scientific Rigor**
   - Two-tailed tests are more conservative
   - Don't assume direction a priori
   - Less susceptible to confirmation bias

2. **Unexpected Findings**
   - What if High-Risk customers have *lower* SARs?
   - Could indicate effective screening or under-reporting
   - One-tailed test would miss this critical finding

3. **Regulatory Standards**
   - Auditors expect unbiased testing
   - Two-tailed tests are harder to challenge
   - Industry best practice for AML validation

4. **Chi-Square is Inherently Two-Tailed**
   - χ² statistic is always positive
   - No directional version exists
   - Consistency: Use two-tailed for all tests

5. **Effect Size Provides Direction**
   - RR and OR tell us the direction
   - No need to pre-specify direction in hypothesis test
   - More informative approach

**Exception:** One-tailed tests justified only with:
- Strong prior evidence from literature
- Theoretical impossibility of opposite direction
- Explicit regulatory/business requirement

### Integration with Machine Learning Pipeline

**Pre-Modeling:**
```
1. Hypothesis Testing → Identify valid risk factors
2. Feature Selection → Include only significant features
3. Model Training → Better signal-to-noise ratio
4. Model Validation → Confirmed features improve generalization
```

**Benefits:**
- ✅ Faster training (fewer features)
- ✅ Better interpretability (each feature is justified)
- ✅ Regulatory compliance (evidence-based approach)
- ✅ Reduced overfitting (no spurious correlations)

**Documentation Trail:**
- Hypothesis test results → Model documentation
- Effect sizes → Feature importance justification
- Confidence intervals → Uncertainty quantification
- Business interpretations → Stakeholder reports

---

## �🔧 Setup and Installation

### Prerequisites
- Python 3.9+
- UV package manager (or pip)

### Installation

```bash
# Clone the repository
git clone https://github.com/masudpervez27/home-assignment.git
cd home-assignment

# Install dependencies using UV
uv sync

# Or using pip
pip install -r requirements.txt
```

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

---

## 🚀 Usage

### Option 1: Run Python Script
```bash
python main.py
```
This will:
- Load and process data
- Engineer features
- Train Random Forest model
- Display performance metrics
- Save trained model to `rf_sar_model.pkl`

### Option 2: Jupyter Notebook (Recommended for Analysis)
```bash
jupyter notebook "Exploration and Data Analysis.ipynb"
```
The notebook contains:
- Comprehensive data exploration
- **Statistical Testing Framework for Feature Selection**
  - Automatic feature type classification
  - Appropriate statistical test selection (T-Test, Mann-Whitney U, Chi-Square)
  - Effect size calculations (Cohen's d, Cramér's V)
  - Bonferroni correction for multiple testing
  - Two-tier feature significance classification
- Feature engineering (including behavioral mismatch features)
- Model development and comparison
- Threshold optimization
- Detailed recommendations

**Key Statistical Outputs:**
- P-values and significance testing results for all features
- Effect size metrics (practical vs. statistical significance)
- TIER 1 features: Bonferroni-corrected (highest confidence)
- TIER 2 features: Significant at p<0.05 (validation recommended)
- Visual analysis: Significance rankings, effect size distributions
- Export-ready feature lists for modeling

---

## 🎯 Key Innovations and Contributions

### 1. Behavioral Mismatch Features
**Novel AML Red Flag Detection**
- Captures discrepancies between KYC declarations and actual transaction behavior
- **Cash Mismatch**: Customer said "No" to cash deposits but made cash transactions
- **International Mismatch**: Customer said "No" to international payments but made them
- **Why Important**: Lying on KYC forms is a strong AML indicator
- Features engineered: Binary flags, transaction counts, amounts when mismatches occur

### 2. Rigorous Statistical Feature Selection
**Multi-Test Framework with Effect Sizes**
- Goes beyond simple correlation analysis
- Applies appropriate statistical tests based on feature type
- Calculates practical significance (effect sizes) not just p-values
- Controls for multiple testing (Bonferroni correction)
- Creates defensible, explainable feature selection for regulatory compliance

### 3. Transaction Type Filtering
**Correct Implementation of Business Rule**
- Excludes salary transactions from turnover calculations
- Aligns with regulatory definition: "turnover > €9,000 (excluding salary)"
- Ensures baseline rule is correctly evaluated
- Critical for fair model comparison

### 4. Comprehensive Model Evaluation
**Beyond Accuracy Metrics**
- Precision-Recall analysis (appropriate for imbalanced data)
- ROC-AUC and PR-AUC curves
- Confusion matrix interpretation
- Business impact analysis (false positives vs. missed SARs)
- Threshold optimization for operational constraints

### 5. Statistical Significance Testing for Temporal Patterns
**Chi-Square Analysis of Monthly SAR Distribution**
- Tests whether month-to-month variation is significant or random
- Informs operational planning (resource allocation)
- Identifies seasonal patterns in suspicious activity
- Validates whether temporal features should be included in models

---

## 🚀 Model Deployment

   - Batch scoring pipeline with monthly scheduling
   - Alert generation
   - Model monitoring, retraining, and governance workflows
   - Integration with existing case management systems

---

## 🔍 Model Artifacts

After running the analysis, the following files are generated:

- `rf_sar_model.pkl` - Trained Random Forest model
- `feature_scaler.pkl` - StandardScaler for feature normalization
- `feature_names.pkl` - List of feature column names

These can be loaded for deployment in production:

```python
import pickle

# Load model
with open('rf_sar_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```
