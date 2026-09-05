# Augmented Dickey–Fuller Test — Stationarity of Equity Prices

Runs an Augmented Dickey–Fuller unit root test on four years of Apple (AAPL)
daily closing prices to establish whether the price series is stationary. This
is the standard pre-condition check before fitting ARIMA, running a
cointegration/pairs-trading analysis, or applying any time-series model whose
estimators assume a stable mean and variance.

## Why stationarity matters

Most time-series econometrics assumes the series is **stationary** — constant
mean, constant variance, and an autocovariance that depends only on lag, not on
calendar time. Equity *price* levels almost never satisfy this: they behave like
a random walk with drift, where the variance grows with the horizon. Regressing
one non-stationary series on another produces *spurious regression* — large
R² and significant t-statistics between series that have no genuine relationship.
Testing first, and differencing when the test fails, is what prevents that.

## Method

**Data** — `yfinance` pulls AAPL daily bars from 2020-01-01 to 2023-12-31, and the
`Close` column is taken as the series under test.

**Test** — `statsmodels.tsa.stattools.adfuller` estimates the ADF regression

```
Δy(t) = α + β·y(t−1) + Σᵏ γᵢ·Δy(t−i) + ε(t)
```

and tests `H₀: β = 0` (a unit root is present → non-stationary) against
`H₁: β < 0` (mean-reverting → stationary). The lagged difference terms are the
"augmented" part: they soak up serial correlation in the residuals so the test
statistic keeps its intended distribution. With `maxlag` left unspecified,
statsmodels selects the lag order automatically by AIC.

**Interpretation** — the ADF statistic does not follow a standard *t*
distribution, so it is compared against Dickey–Fuller critical values (returned
in `result[4]`). The script uses the p-value with a 5% threshold: below 0.05,
reject the unit root and call the series stationary; above, conclude that
differencing is required.

**Expected outcome** — for raw price *levels* over a trending four-year window,
the test should fail to reject `H₀`. That is the correct and informative result:
it says the modelling should be done on **log returns**, `Δ log y`, which are
stationary, rather than on prices.

The script also plots the closing-price series before testing, so the visual
trend can be read alongside the formal test.

## Usage

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib statsmodels yfinance

python3 "ADF Test"
```

Output is the ADF statistic, the p-value, and a one-line stationary /
non-stationary verdict, plus an interactive price chart.

## Repo structure

```
ADF Test    yfinance download -> price plot -> adfuller unit root test (Python; no .py extension)
```

## Known caveats

Documented here rather than left to be discovered at runtime:

- **yfinance column shape.** Recent `yfinance` versions return a MultiIndex
  column frame even for a single ticker, so `stock_data['Close']` yields a
  one-column *DataFrame* rather than a Series, and `adfuller` will reject it.
  Fix with `yf.download("AAPL", ..., auto_adjust=True)['Close'].squeeze()`, or
  index explicitly as `stock_data['Close']['AAPL']`.
- **Levels only.** The script tests prices but never re-tests the differenced
  series, so it stops one step short of demonstrating that `Δ log price` *is*
  stationary. Running `adfuller(np.log(prices).diff().dropna())` completes the
  argument.
- **Deterministic trend.** The default `regression='c'` includes a constant but
  no time trend. For a series with visible drift, `regression='ct'` is the more
  appropriate specification, and the two can disagree.
- **Low power.** ADF is known to have weak power against a highly persistent but
  genuinely mean-reverting alternative. Pairing it with a KPSS test — which
  reverses the null hypothesis — gives a much more robust conclusion than either
  test alone.
