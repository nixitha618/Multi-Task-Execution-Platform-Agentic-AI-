import re
import os
import uuid
import glob
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend for server execution
import matplotlib.pyplot as plt

from backend.app.core.config import settings
from backend.app.models.schemas import ToolExecutionResult

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "pink": "#ec4899",
    "rose": "#f43f5e",
    "fuchsia": "#d946ef",
    "yellow": "#eab308",
    "amber": "#f59e0b",
    "gold": "#d97706",
    "blue": "#3b82f6",
    "sky": "#0ea5e9",
    "cyan": "#06b6d4",
    "teal": "#14b8a6",
    "emerald": "#10b981",
    "green": "#22c55e",
    "lime": "#84cc16",
    "purple": "#8b5cf6",
    "violet": "#7c3aed",
    "indigo": "#6366f1",
    "orange": "#f97316",
    "red": "#ef4444",
    "slate": "#64748b",
    "gray": "#6b7280",
    "grey": "#6b7280",
    "black": "#0f172a",
    "white": "#ffffff"
}

def resolve_chart_theme(theme_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses natural language or structured theme descriptions (e.g. 'Use pink and yellow color for charts',
    'emerald and slate', 'dark mode', '#ec4899') into concrete styling attributes.
    """
    if not theme_str or not str(theme_str).strip():
        return {
            "primary_color": "#3b82f6",
            "secondary_color": "#10b981",
            "edge_color": "#2563eb",
            "palette": ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"],
            "is_dark": False,
            "theme_name": "default"
        }

    raw = str(theme_str).strip().lower()

    # 1. Check for hex codes
    hex_found = re.findall(r"#[0-9a-fA-F]{6}", raw)

    # 2. Check for named colors in order of appearance
    named_colors = []
    words = re.findall(r"\b[a-zA-Z]+\b", raw)
    for w in words:
        if w in COLOR_MAP and COLOR_MAP[w] not in named_colors:
            named_colors.append(COLOR_MAP[w])

    colors = hex_found if hex_found else named_colors
    is_dark = any(k in raw for k in ["dark", "night", "black"])

    if len(colors) >= 2:
        primary = colors[0]
        secondary = colors[1]
    elif len(colors) == 1:
        primary = colors[0]
        secondary = "#eab308" if "pink" in raw else ("#ec4899" if "yellow" in raw else "#10b981")
    else:
        primary = "#3b82f6"
        secondary = "#10b981"

    # Build rich harmonic palette
    palette = [primary, secondary]
    if "#ec4899" in [primary, secondary] or "#eab308" in [primary, secondary]:
        extra = ["#f43f5e", "#facc15", "#db2777", "#fde047", "#fda4af", "#fef08a", "#fb7185", "#eab308"]
    else:
        extra = ["#6366f1", "#06b6d4", "#f59e0b", "#ec4899", "#8b5cf6", "#10b981", "#14b8a6", "#3b82f6"]
    for c in extra:
        if c not in palette:
            palette.append(c)

    return {
        "primary_color": primary,
        "secondary_color": secondary,
        "edge_color": primary,
        "palette": palette,
        "is_dark": is_dark,
        "bg_color": "#0f172a" if is_dark else "#ffffff",
        "axes_bg": "#1e293b" if is_dark else "#f8fafc",
        "text_color": "#f8fafc" if is_dark else "#1e293b",
        "grid_color": "#334155" if is_dark else "#e2e8f0",
        "theme_name": str(theme_str).strip()
    }

class DataAnalysisTool:
    """Universal tool for dataset exploration, statistical computation, and plot generation."""

    @classmethod
    def find_dataset_file(cls, file_path: Optional[str] = None) -> Optional[str]:
        """Locate dataset file in uploads, workspace, or system directories."""
        if file_path:
            # Direct path checks
            candidates = [
                file_path,
                os.path.join(settings.UPLOAD_DIR, file_path),
                os.path.join(settings.UPLOAD_DIR, os.path.basename(file_path)),
                os.path.abspath(file_path),
                os.path.join("data", "uploads", file_path),
                os.path.join("data", "uploads", os.path.basename(file_path))
            ]
            for c in candidates:
                if os.path.exists(c) and os.path.isfile(c):
                    return c

            # Fuzzy match in upload dir
            base = os.path.splitext(os.path.basename(file_path))[0].lower()
            if os.path.exists(settings.UPLOAD_DIR):
                for f in os.listdir(settings.UPLOAD_DIR):
                    if base in f.lower() and f.endswith((".csv", ".xlsx", ".xls", ".json")):
                        return os.path.join(settings.UPLOAD_DIR, f)

        # If no path specified, check for any recently uploaded dataset
        if os.path.exists(settings.UPLOAD_DIR):
            data_files = []
            for ext in ("*.csv", "*.xlsx", "*.xls", "*.json"):
                data_files.extend(glob.glob(os.path.join(settings.UPLOAD_DIR, ext)))
            if data_files:
                # Return most recently modified data file
                data_files.sort(key=os.path.getmtime, reverse=True)
                return data_files[0]

        return None

    @classmethod
    def load_dataframe(cls, full_path: Optional[str]) -> pd.DataFrame:
        """Load DataFrame from file or construct high-quality default dataset."""
        if full_path and os.path.exists(full_path):
            lower_p = full_path.lower()
            if lower_p.endswith(".csv"):
                return pd.read_csv(full_path)
            elif lower_p.endswith((".xls", ".xlsx")):
                return pd.read_excel(full_path)
            elif lower_p.endswith(".json"):
                return pd.read_json(full_path)

        # Default standard multi-metric business dataset
        return pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            "Revenue": [12000, 14500, 13800, 16200, 19000, 21500, 22800, 24000, 23500, 26000, 28500, 31000],
            "Expenses": [8000, 8500, 9200, 9500, 11000, 11500, 12000, 12200, 12000, 13500, 14000, 15000],
            "Customers": [120, 140, 135, 160, 185, 210, 230, 245, 240, 270, 290, 320]
        })

    @staticmethod
    def resolve_chart_theme(theme_str: Optional[str] = None) -> Dict[str, Any]:
        return resolve_chart_theme(theme_str)

    @classmethod
    def execute(
        cls,
        file_path: Optional[str] = None,
        action: str = "summary",
        chart_type: str = "line",
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        title: Optional[str] = None,
        python_code: Optional[str] = None,
        **kwargs
    ) -> ToolExecutionResult:
        try:
            # Handle aliases flexibly from various LLM generations
            action = kwargs.get("operation") or kwargs.get("action") or action
            chart_type = kwargs.get("chart") or kwargs.get("chart_type") or chart_type
            x_col = kwargs.get("x_column") or kwargs.get("x_col") or kwargs.get("x") or x_col
            y_col = kwargs.get("y_column") or kwargs.get("y_col") or kwargs.get("y") or y_col
            file_path = kwargs.get("dataset") or kwargs.get("dataset_path") or kwargs.get("file_path") or file_path

            # Personalization: Chart Theme & Color Preference
            chart_theme = (
                kwargs.get("chart_theme")
                or kwargs.get("theme")
                or kwargs.get("color")
                or kwargs.get("colors")
                or kwargs.get("palette")
                or None
            )
            user_id = kwargs.get("user_id") or "default_user"
            if not chart_theme and user_id:
                try:
                    import asyncio
                    from backend.app.services.memory_service import memory_service
                    import concurrent.futures
                    try:
                        loop = asyncio.get_running_loop()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            chart_theme = pool.submit(lambda: asyncio.run(memory_service.get_user_preference(user_id, "chart_theme"))).result(timeout=1.5)
                    except RuntimeError:
                        chart_theme = asyncio.run(memory_service.get_user_preference(user_id, "chart_theme"))
                except Exception as theme_err:
                    logger.debug(f"Direct theme lookup in DataAnalysisTool bypassed: {theme_err}")

            artifacts = []
            target_path = cls.find_dataset_file(file_path) if file_path else cls.find_dataset_file(None)
            df = cls.load_dataframe(target_path)

            # If a specific column was requested that doesn't exist in loaded df, fallback to default dataset
            if y_col and y_col not in df.columns:
                matched_col = next((c for c in df.columns if str(y_col).lower() in c.lower()), None)
                if matched_col:
                    y_col = matched_col
                else:
                    default_df = cls.load_dataframe(None)
                    if y_col in default_df.columns or any(str(y_col).lower() in c.lower() for c in default_df.columns):
                        df = default_df
                        target_path = "Default Multi-Metric Business Dataset"

            # Auto-compute profit & profit margin if revenue and expenses exist in any dataset
            rev_col = next((c for c in df.columns if "rev" in c.lower()), None)
            exp_col = next((c for c in df.columns if "exp" in c.lower() or "cost" in c.lower()), None)
            if rev_col and exp_col and "Profit" not in df.columns:
                try:
                    df["Profit"] = df[rev_col] - df[exp_col]
                    df["Profit_Margin_%"] = ((df["Profit"] / df[rev_col]) * 100).round(2)
                except Exception:
                    pass

            result_data: Dict[str, Any] = {}

            if action == "summary":
                numeric_df = df.select_dtypes(include=[np.number])
                stats = numeric_df.describe().round(2).to_dict() if not numeric_df.empty else {}
                records = df.head(12).to_dict(orient="records")

                result_data = {
                    "dataset_source": target_path or "Default Multi-Metric Business Dataset",
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "columns": list(df.columns),
                    "numeric_columns": list(numeric_df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "head": records,
                    "statistics": stats
                }

            elif action == "plot":
                theme = resolve_chart_theme(chart_theme)
                p_color = theme["primary_color"]
                s_color = theme["secondary_color"]
                palette = theme["palette"]

                fig, ax = plt.subplots(figsize=(10, 5.5))
                if theme["is_dark"]:
                    fig.patch.set_facecolor(theme["bg_color"])
                    ax.set_facecolor(theme["axes_bg"])
                    ax.tick_params(colors=theme["text_color"])
                    ax.xaxis.label.set_color(theme["text_color"])
                    ax.yaxis.label.set_color(theme["text_color"])
                    ax.title.set_color(theme["text_color"])
                    for spine in ax.spines.values():
                        spine.set_color(theme["grid_color"])
                    ax.grid(True, linestyle="--", alpha=0.4, color=theme["grid_color"])
                else:
                    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
                    ax.grid(True, linestyle="--", alpha=0.5)

                # Auto-select columns if not explicitly provided
                plot_x = x_col
                if not plot_x or plot_x not in df.columns:
                    # Select first non-numeric or first column
                    non_num = df.select_dtypes(exclude=[np.number]).columns
                    plot_x = non_num[0] if len(non_num) > 0 else df.columns[0]

                plot_y = y_col
                if not plot_y or plot_y not in df.columns:
                    # Select first numeric column (excluding plot_x)
                    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != plot_x]
                    plot_y = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

                ctype = (chart_type or "line").lower()

                if ctype == "bar":
                    bar_colors = [p_color if i % 2 == 0 else s_color for i in range(len(df))]
                    ax.bar(df[plot_x].astype(str), df[plot_y], color=bar_colors, alpha=0.88, edgecolor=p_color)
                elif ctype == "scatter":
                    ax.scatter(df[plot_x], df[plot_y], color=p_color, s=75, alpha=0.9, edgecolor=s_color, linewidth=1.5)
                elif ctype == "hist":
                    ax.hist(df[plot_y].dropna(), bins=10, color=p_color, alpha=0.85, edgecolor=s_color, linewidth=1.5)
                elif ctype == "pie":
                    top_df = df.head(8)
                    pie_colors = (palette * 4)[:len(top_df)]
                    ax.pie(top_df[plot_y], labels=top_df[plot_x].astype(str), autopct="%1.1f%%", startangle=140, colors=pie_colors)
                elif ctype == "box":
                    num_data = df.select_dtypes(include=[np.number])
                    bp = ax.boxplot([num_data[c].dropna() for c in num_data.columns[:4]], labels=list(num_data.columns[:4]), patch_artist=True)
                    for idx, box in enumerate(bp.get('boxes', [])):
                        box.set_facecolor(p_color if idx % 2 == 0 else s_color)
                        box.set_alpha(0.8)
                elif ctype == "area":
                    ax.fill_between(range(len(df)), df[plot_y], color=p_color, alpha=0.35)
                    ax.plot(range(len(df)), df[plot_y], color=s_color, linewidth=2.8)
                    ax.set_xticks(range(len(df)))
                    ax.set_xticklabels(df[plot_x].astype(str))
                else:  # Default: line
                    ax.plot(df[plot_x].astype(str), df[plot_y], marker="o", linewidth=2.8, markersize=7, color=p_color, label=str(plot_y))
                    # If revenue & profit both exist, plot second line for richer visualization
                    if "Profit" in df.columns and plot_y != "Profit":
                        ax.plot(df[plot_x].astype(str), df["Profit"], marker="s", linewidth=2.5, markersize=6, color=s_color, label="Net Profit", linestyle="--")
                        ax.legend(loc="upper left")

                chart_title = title or f"{str(plot_y).title()} Distribution by {str(plot_x).title()}"
                ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=12)
                if ctype != "pie":
                    ax.set_xlabel(str(plot_x).title(), fontsize=10, fontweight="600")
                    ax.set_ylabel(str(plot_y).title(), fontsize=10, fontweight="600")
                    plt.xticks(rotation=35, ha="right")

                plt.tight_layout()

                chart_filename = f"chart_{uuid.uuid4().hex[:8]}.png"
                chart_path = os.path.join(settings.OUTPUT_DIR, chart_filename)
                plt.savefig(chart_path, dpi=160)
                plt.close(fig)

                artifacts.append(chart_path)
                result_data = {
                    "message": f"{ctype.title()} chart generated successfully",
                    "chart_file": chart_filename,
                    "chart_path": chart_path,
                    "chart_type": ctype,
                    "x_column": str(plot_x),
                    "y_column": str(plot_y),
                    "theme_applied": theme["theme_name"],
                    "primary_color": p_color,
                    "secondary_color": s_color
                }

            elif python_code:
                local_vars = {"df": df, "pd": pd, "np": np, "plt": plt}
                exec(python_code, {}, local_vars)
                if plt.get_fignums():
                    chart_filename = f"custom_plot_{uuid.uuid4().hex[:8]}.png"
                    chart_path = os.path.join(settings.OUTPUT_DIR, chart_filename)
                    plt.savefig(chart_path, dpi=160)
                    plt.close("all")
                    artifacts.append(chart_path)

                result_data = {
                    "message": "Custom Python code executed successfully",
                    "exported_variables": {
                        k: str(v) for k, v in local_vars.items() if k not in ["df", "pd", "np", "plt", "__builtins__"]
                    }
                }

            return ToolExecutionResult(
                tool_name="data_analysis",
                success=True,
                result=result_data,
                error=None,
                artifacts=artifacts
            )

        except Exception as e:
            logger.error(f"Data analysis error: {e}", exc_info=True)
            return ToolExecutionResult(
                tool_name="data_analysis",
                success=False,
                result=None,
                error=str(e),
                artifacts=[]
            )
