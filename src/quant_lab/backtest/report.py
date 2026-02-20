"""HTML backtest report generation using Plotly + Jinja2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Template

import structlog

logger = structlog.get_logger(__name__)


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: #1a1a2e; color: white; padding: 20px 30px; border-radius: 8px;
                  margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 24px; }
        .header p { margin: 5px 0 0; opacity: 0.8; font-size: 14px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                       gap: 15px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 15px 20px; border-radius: 8px;
                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .metric-card .label { font-size: 12px; color: #666; text-transform: uppercase;
                             letter-spacing: 1px; }
        .metric-card .value { font-size: 24px; font-weight: 600; color: #1a1a2e; margin-top: 5px; }
        .metric-card .value.positive { color: #2ecc71; }
        .metric-card .value.negative { color: #e74c3c; }
        .chart-container { background: white; border-radius: 8px; padding: 20px;
                          box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section-title { font-size: 18px; font-weight: 600; color: #1a1a2e;
                        margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th { background: #f8f9fa; padding: 10px 12px; text-align: left; font-weight: 600;
             border-bottom: 2px solid #dee2e6; }
        td { padding: 8px 12px; border-bottom: 1px solid #eee; }
        .footer { text-align: center; color: #999; font-size: 12px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generated_at }} | Period: {{ period }}</p>
    </div>

    <div class="metrics-grid">
        {% for m in metrics %}
        <div class="metric-card">
            <div class="label">{{ m.label }}</div>
            <div class="value {{ m.css_class }}">{{ m.value }}</div>
        </div>
        {% endfor %}
    </div>

    {% for chart in charts %}
    <div class="chart-container">
        <div class="section-title">{{ chart.title }}</div>
        {{ chart.html }}
    </div>
    {% endfor %}

    {% if regime_table %}
    <div class="chart-container">
        <div class="section-title">Regime Analysis</div>
        {{ regime_table }}
    </div>
    {% endif %}

    {% if regime_performance_table %}
    <div class="chart-container">
        <div class="section-title">Performance by Regime</div>
        {{ regime_performance_table }}
    </div>
    {% endif %}

    <div class="footer">
        Quant Lab Backtest Report &bull; AI Quant Research Lab
    </div>
</body>
</html>
"""


@dataclass
class ReportConfig:
    """Report generation configuration."""

    title: str = "Backtest Report"
    output_dir: str = "outputs/backtests"
    include_regime_analysis: bool = True


class BacktestReport:
    """Generate HTML backtest reports with interactive Plotly charts."""

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()

    def generate(
        self,
        portfolio_values: np.ndarray,
        dates: np.ndarray | pd.DatetimeIndex | None = None,
        metrics: dict[str, float] | None = None,
        weights_history: np.ndarray | None = None,
        regime_labels: np.ndarray | None = None,
        regime_summary: pd.DataFrame | None = None,
        benchmark_values: np.ndarray | None = None,
        ticker_names: list[str] | None = None,
    ) -> str:
        """Generate HTML report.

        Args:
            portfolio_values: (n_steps,) portfolio value series.
            dates: Optional date index for x-axis.
            metrics: Dict of performance metrics (Sharpe, CAGR, etc.).
            weights_history: Optional (n_steps, n_assets) weight matrix.
            regime_labels: Optional (n_steps,) regime labels for coloring.
            regime_summary: Optional regime characteristics DataFrame.
            benchmark_values: Optional benchmark value series for comparison.

        Returns:
            Path to the generated HTML file.
        """
        if dates is None:
            dates = np.arange(len(portfolio_values))

        # Build metric cards
        metric_cards = self._build_metric_cards(metrics or {})

        # Build charts
        charts = []

        # 1. Equity curve
        equity_fig = self._equity_curve(portfolio_values, dates, benchmark_values, regime_labels)
        charts.append({"title": "Equity Curve", "html": equity_fig.to_html(full_html=False, include_plotlyjs="cdn")})

        # 2. Drawdown chart
        dd_fig = self._drawdown_chart(portfolio_values, dates)
        charts.append({"title": "Drawdown", "html": dd_fig.to_html(full_html=False, include_plotlyjs=False)})

        # 3. Rolling returns
        rolling_fig = self._rolling_returns(portfolio_values, dates)
        charts.append({"title": "Rolling 21-Day Returns", "html": rolling_fig.to_html(full_html=False, include_plotlyjs=False)})

        # 4. Weight allocation (if provided)
        if weights_history is not None:
            weight_fig = self._weight_allocation(weights_history, dates, ticker_names)
            charts.append({"title": "Portfolio Allocation", "html": weight_fig.to_html(full_html=False, include_plotlyjs=False)})

        # Regime table
        regime_table_html = ""
        if regime_summary is not None and len(regime_summary) > 0:
            regime_table_html = regime_summary.to_html(index=False, classes="regime-table")

        # Regime-conditional performance breakdown
        regime_perf_html = ""
        if regime_labels is not None and len(regime_labels) == len(portfolio_values):
            regime_perf_html = self._regime_performance_breakdown(
                portfolio_values, regime_labels
            )

        # Render template
        template = Template(REPORT_TEMPLATE)
        period = f"{dates[0]} to {dates[-1]}" if hasattr(dates[0], "strftime") else f"Step 0 to {len(dates) - 1}"

        html_content = template.render(
            title=self.config.title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            period=period,
            metrics=metric_cards,
            charts=charts,
            regime_table=regime_table_html,
            regime_performance_table=regime_perf_html,
        )

        # Save
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "report.html"
        output_path.write_text(html_content, encoding="utf-8")

        logger.info("report_generated", path=str(output_path))
        return str(output_path)

    def _build_metric_cards(self, metrics: dict) -> list[dict]:
        """Build metric card data for the template."""
        cards = []
        format_map = {
            "cagr": ("{:.2%}", True),
            "sharpe": ("{:.2f}", True),
            "sortino": ("{:.2f}", True),
            "max_drawdown": ("{:.2%}", False),
            "calmar": ("{:.2f}", True),
            "total_return": ("{:.2%}", True),
            "annual_volatility": ("{:.2%}", False),
            "turnover": ("{:.4f}", False),
        }

        for key, value in metrics.items():
            fmt, higher_better = format_map.get(key, ("{:.4f}", True))
            formatted = fmt.format(value)
            css_class = ""
            if higher_better:
                css_class = "positive" if value > 0 else "negative"

            cards.append({
                "label": key.replace("_", " ").title(),
                "value": formatted,
                "css_class": css_class,
            })

        return cards

    def _equity_curve(
        self,
        values: np.ndarray,
        dates: np.ndarray,
        benchmark: np.ndarray | None = None,
        regimes: np.ndarray | None = None,
    ) -> go.Figure:
        """Create equity curve chart."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode="lines", name="Portfolio",
            line=dict(color="#2196F3", width=2),
        ))

        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=dates, y=benchmark,
                mode="lines", name="Benchmark",
                line=dict(color="#9E9E9E", width=1, dash="dash"),
            ))

        if regimes is not None:
            colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
            unique_regimes = sorted(set(regimes[regimes >= 0]))
            for rid in unique_regimes:
                mask = regimes == rid
                fig.add_trace(go.Scatter(
                    x=dates[mask], y=values[mask],
                    mode="markers", name=f"Regime {rid}",
                    marker=dict(color=colors[rid % len(colors)], size=3, opacity=0.5),
                ))

        fig.update_layout(
            height=400, template="plotly_white",
            xaxis_title="Date", yaxis_title="Portfolio Value",
            hovermode="x unified",
        )
        return fig

    def _drawdown_chart(self, values: np.ndarray, dates: np.ndarray) -> go.Figure:
        """Create drawdown chart."""
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown,
            mode="lines", name="Drawdown",
            fill="tozeroy",
            line=dict(color="#e74c3c", width=1),
            fillcolor="rgba(231, 76, 60, 0.3)",
        ))
        fig.update_layout(
            height=250, template="plotly_white",
            xaxis_title="Date", yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
        )
        return fig

    def _rolling_returns(self, values: np.ndarray, dates: np.ndarray) -> go.Figure:
        """Create rolling 21-day return chart."""
        returns = np.diff(values) / values[:-1]
        window = min(21, len(returns))
        if window < 2:
            rolling = returns
        else:
            rolling = pd.Series(returns).rolling(window).mean().values

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates[1:], y=rolling,
            name="Rolling Return",
            marker_color=np.where(rolling >= 0, "#2ecc71", "#e74c3c"),
        ))
        fig.update_layout(
            height=250, template="plotly_white",
            xaxis_title="Date", yaxis_title="Rolling Return",
            yaxis_tickformat=".2%",
        )
        return fig

    def _weight_allocation(
        self, weights: np.ndarray, dates: np.ndarray,
        ticker_names: list[str] | None = None,
    ) -> go.Figure:
        """Create stacked area chart of portfolio weights."""
        fig = go.Figure()
        n_assets = weights.shape[1]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336",
                  "#00BCD4", "#CDDC39", "#795548", "#607D8B", "#E91E63"]

        for i in range(min(n_assets, 10)):  # Show at most 10 assets
            name = ticker_names[i] if ticker_names and i < len(ticker_names) else f"Asset {i}"
            fig.add_trace(go.Scatter(
                x=dates[:len(weights)],
                y=weights[:, i],
                mode="lines",
                stackgroup="one",
                name=name,
                line=dict(width=0),
                fillcolor=colors[i % len(colors)],
            ))

        fig.update_layout(
            height=300, template="plotly_white",
            xaxis_title="Date", yaxis_title="Weight",
            yaxis_range=[0, 1],
        )
        return fig

    def _regime_performance_breakdown(
        self,
        portfolio_values: np.ndarray,
        regime_labels: np.ndarray,
    ) -> str:
        """Compute per-regime performance metrics and return HTML table."""
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        # Align labels with returns (returns has length n-1)
        labels = regime_labels[1:]

        unique_regimes = sorted(set(labels[labels >= 0]))
        if len(unique_regimes) == 0:
            return ""

        rows = []
        for rid in unique_regimes:
            mask = labels == rid
            r = returns[mask]
            if len(r) == 0:
                continue

            ann_return = float(np.mean(r)) * 252
            ann_vol = float(np.std(r)) * np.sqrt(252) if len(r) > 1 else 0.0
            sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0.0

            # Max drawdown within this regime's periods
            cum = np.cumprod(1 + r)
            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

            rows.append({
                "Regime": int(rid),
                "Days": int(mask.sum()),
                "Pct of Time": f"{mask.sum() / len(labels) * 100:.1f}%",
                "Ann. Return": f"{ann_return:.2%}",
                "Ann. Volatility": f"{ann_vol:.2%}",
                "Sharpe": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.2%}",
            })

        df = pd.DataFrame(rows)
        return df.to_html(index=False, classes="regime-perf-table")
