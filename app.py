import os
import time
from typing import Literal

import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# 1. PAGE SETUP + STYLING
# ==========================================
# Colors are set explicitly everywhere (not "inherit" / "default") on
# purpose. Relying on the browser/OS light-vs-dark preference is exactly
# what caused the invisible-text problem, since text and background could
# independently flip. Pairing them with .streamlit/config.toml (also
# provided) locks the whole app to one deliberate dark theme.

BG = "#0B1220"          # page background
CARD_BG = "#131C2E"     # card background (slightly lighter than page)
CARD_BORDER = "#1E293B"
TEXT_PRIMARY = "#E2E8F0"
TEXT_MUTED = "#94A3B8"
ACCENT = "#14B8A6"      # teal — active/done states, buttons
ACCENT_2 = "#38BDF8"    # sky blue — secondary accent

st.set_page_config(
    page_title="Sales Predictor",
    page_icon="📊",
    layout="centered"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .app-header {{
        margin-bottom: 0.25rem;
        color: {TEXT_PRIMARY} !important;
        font-weight: 700;
    }}
    .app-subtitle {{ color: {TEXT_MUTED}; font-size: 0.95rem; margin-bottom: 1.5rem; }}

    .step-card {{
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        background: {CARD_BG};
    }}
    .step-card h4 {{ margin-top: 0; color: {TEXT_PRIMARY} !important; }}

    .stepper {{ display: flex; align-items: center; margin-bottom: 2rem; }}
    .step-circle {{
        width: 26px; height: 26px; min-width: 26px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700;
    }}
    .step-circle.done {{ background: {ACCENT}; color: {BG}; }}
    .step-circle.active {{ background: {ACCENT_2}; color: {BG}; }}
    .step-circle.todo {{ background: {CARD_BORDER}; color: {TEXT_MUTED}; }}
    .step-text {{ margin: 0 0.6rem; font-size: 0.85rem; font-weight: 500; color: {TEXT_MUTED}; white-space: nowrap; }}
    .step-text.active {{ color: {TEXT_PRIMARY}; }}
    .step-line {{ flex: 1; height: 2px; background: {CARD_BORDER}; margin: 0 0.4rem; }}
    .step-line.done {{ background: {ACCENT}; }}

    .req-item {{
        border-left: 3px solid {ACCENT};
        background: {BG};
        padding: 0.5rem 0.85rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
        color: {TEXT_PRIMARY};
    }}

    .stButton>button {{
        background-color: {ACCENT};
        color: {BG};
        border: none;
        border-radius: 6px;
        font-weight: 700;
        padding: 0.5rem 1.3rem;
    }}
    .stButton>button:hover {{ background-color: {ACCENT_2}; color: {BG}; }}
    .stButton>button:disabled {{ background-color: {CARD_BORDER}; color: {TEXT_MUTED}; }}

    .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; }}

    .stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="app-header">Sales Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Explore the historical data, then describe what you want '
    'predicted in plain English.</div>',
    unsafe_allow_html=True
)


# ==========================================
# 2. LOAD MODELS + DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "sales.csv")

MODEL_FILES = {
    "General Model": "model_general.pkl",
    "Recent Year Model": "model_recent_year.pkl",
    "High Profit Model": "model_high_profit.pkl",
    "Best Seller Model": "model_best_seller.pkl",
    "Loss Risk Model": "model_loss_risk.pkl",
    "High Discount Model": "model_high_discount.pkl",
}

# Vibrant enough to read clearly against the dark card background
BRAND_COLORS = ["#14B8A6", "#38BDF8", "#A78BFA", "#FBBF24", "#FB7185", "#34D399"]


@st.cache_resource
def load_models():
    return {label: joblib.load(os.path.join(MODELS_DIR, f)) for label, f in MODEL_FILES.items()}


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Order_Year"] = df["Order Date"].dt.year
    df["Order_Month"] = df["Order Date"].dt.month
    return df


def build_dataset_context(df: pd.DataFrame) -> dict:
    return {
        "min_year": int(df["Order_Year"].min()),
        "max_year": int(df["Order_Year"].max()),
        "median_quantity": float(df["Quantity"].median()),
        "p90_quantity": float(df["Quantity"].quantile(0.90)),
        "median_discount": float(df["Discount"].median()),
        "p90_discount": float(df["Discount"].quantile(0.90)),
        "median_profit": float(df["Profit"].median()),
        "most_common_segment": df["Segment"].mode()[0],
        "most_common_region": df["Region"].mode()[0],
        "most_common_ship_mode": df["Ship Mode"].mode()[0],
        "product_catalog": df[["Sub-Category", "Product Name"]].drop_duplicates(),
    }


try:
    models = load_models()
    df = load_raw_data()
    data_ctx = build_dataset_context(df)
except FileNotFoundError as e:
    st.error(
        f"A required file was not found:\n{e}\n\n"
        "Run multi_train.py first (from inside src/) to create the models."
    )
    st.stop()


@st.cache_resource
def load_genai_client():
    return genai.Client()

client = load_genai_client()
MODEL_FALLBACK_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]


def call_gemini_with_retry(prompt: str, schema, max_attempts_per_model: int = 2):
    last_error = None
    for model_name in MODEL_FALLBACK_CHAIN:
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)
    raise last_error


# ==========================================
# 3. STRUCTURED OUTPUT SCHEMAS
# ==========================================

class RequirementInterpretation(BaseModel):
    features: list[str] = Field(description="Dataset feature(s) this requirement refers to")
    condition: str = Field(description="Condition implied, e.g. 'high quantity', or 'none'")
    purpose: str = Field(description="One short sentence: what the user is trying to predict")


class InterpretationBatch(BaseModel):
    interpretations: list[RequirementInterpretation]


class PredictionInput(BaseModel):
    quantity: int = Field(ge=1, le=50)
    discount: float = Field(ge=0.0, le=0.8)
    profit: float = Field(ge=-2000.0, le=1500.0)
    segment: Literal["Consumer", "Corporate", "Home Office"]
    region: Literal["Central", "East", "South", "West"]
    ship_mode: Literal["Standard Class", "Second Class", "First Class", "Same Day"]
    sub_category: Literal["Bookcases", "Chairs", "Furnishings", "Tables"]
    product_name: str
    order_year: int
    order_month: int = Field(ge=1, le=12)
    order_day: int = Field(ge=1, le=28)
    model_choice: Literal[
        "General Model", "Recent Year Model", "High Profit Model",
        "Best Seller Model", "Loss Risk Model", "High Discount Model"
    ]
    reasoning: str = Field(description="Which values were explicit vs inferred, and why this model was chosen.")


# ==========================================
# 4. SESSION STATE
# ==========================================

if "requirements" not in st.session_state:
    st.session_state.requirements = []
if "interpretations" not in st.session_state:
    st.session_state.interpretations = []
if "show_selection" not in st.session_state:
    st.session_state.show_selection = False
if "final_result" not in st.session_state:
    st.session_state.final_result = None


# ==========================================
# 5. DATASET OVERVIEW DASHBOARD
# ==========================================

def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        showlegend=False,
        xaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
        yaxis=dict(gridcolor=CARD_BORDER, zerolinecolor=CARD_BORDER),
    )
    return fig


def render_dashboard(df: pd.DataFrame):
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### At a glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total orders", f"{len(df):,}")
    col3.metric("Avg order value", f"${df['Sales'].mean():,.2f}")
    col4.metric("Total profit", f"${df['Profit'].sum():,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Top sellers and regional split")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Top 10 products by sales")
        top_products = (
            df.groupby("Product Name")["Sales"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(
            top_products, x="Sales", y="Product Name", orientation="h",
            color_discrete_sequence=[BRAND_COLORS[0]]
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with c2:
        st.caption("Sales by region")
        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig = px.bar(
            region_sales, x="Region", y="Sales",
            color_discrete_sequence=[BRAND_COLORS[1]]
        )
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(style_chart(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Sales trend over time")
    trend = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
    fig = px.line(trend, x="Order Date", y="Sales", color_discrete_sequence=[BRAND_COLORS[0]])
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style_chart(fig, height=300), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Customer segments and product categories")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Sales by segment")
        segment_sales = df.groupby("Segment")["Sales"].sum().reset_index()
        fig = px.pie(
            segment_sales, names="Segment", values="Sales", hole=0.55,
            color_discrete_sequence=BRAND_COLORS
        )
        fig.update_traces(textinfo="percent+label", textfont_color=TEXT_PRIMARY)
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, showlegend=False,
            font=dict(color=TEXT_PRIMARY)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.caption("Sales by sub-category")
        subcat = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(subcat, x="Sub-Category", y="Sales", color_discrete_sequence=[BRAND_COLORS[2]])
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style_chart(fig, height=320), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Discount vs. profit")
    st.caption("Each dot is one order. This is the pattern the High Discount and Loss Risk models specialize in.")
    fig = px.scatter(df, x="Discount", y="Profit", opacity=0.45, color_discrete_sequence=[ACCENT_2])
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED)
    st.plotly_chart(style_chart(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 6. PREDICTION WIZARD (Steps 1-3)
# ==========================================

def interpret_requirements(requirements: list[str]) -> list[RequirementInterpretation]:
    prompt = f"""
You are helping translate natural-language prediction requests into
structured machine learning terms for a furniture sales dataset.

Available dataset features: Quantity, Discount, Profit, Segment, Region,
Ship Mode, Sub-Category, Product Name, Order_Year, Order_Month, Order_Day.
The target being predicted is always: Sales.

Here is a numbered list of user requirements, in order:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(requirements))}

For EACH requirement, in the SAME ORDER, identify which feature(s) it
refers to, any condition implied (e.g. "high", "recent", "discounted"),
and the purpose in one short sentence.
"""
    try:
        response = call_gemini_with_retry(prompt, InterpretationBatch)
    except Exception as e:
        st.error(f"Couldn't reach Gemini after multiple attempts: {e}")
        st.stop()

    if response.parsed is None:
        st.error("Gemini's response didn't match the expected format. Try rephrasing your requirement.")
        st.stop()

    return response.parsed.interpretations


def build_prediction_input(selected: list[str]) -> PredictionInput:
    combined_text = "; ".join(selected)
    prompt = f"""
You are configuring a single prediction for a furniture sales ML model.

The user's combined requirements (already selected by them) are:
"{combined_text}"

Real statistics from the historical dataset, to ground your defaults:
- Years available: {data_ctx['min_year']} to {data_ctx['max_year']}
- Median order quantity: {data_ctx['median_quantity']}, 90th percentile ("high quantity"): {data_ctx['p90_quantity']}
- Median discount: {data_ctx['median_discount']}, 90th percentile ("high discount"): {data_ctx['p90_discount']}
- Median profit: {data_ctx['median_profit']}
- Most common segment: {data_ctx['most_common_segment']}
- Most common region: {data_ctx['most_common_region']}
- Most common ship mode: {data_ctx['most_common_ship_mode']}

Instructions:
- If the requirements mention a specific value or clear condition, use the
  matching statistic above.
- For anything NOT mentioned, use the most common / median value instead of
  guessing randomly.
- Choose exactly one model from the allowed list that best matches the
  requirements ("Recent Year Model" for recent/latest year emphasis,
  "High Profit Model" for high profit, "Best Seller Model" for best-selling
  product/category, "Loss Risk Model" for loss-making orders, "High Discount
  Model" for heavy discounts, "General Model" otherwise).
- Explain your reasoning, including which values you inferred vs which came
  directly from the user's text.
"""
    try:
        response = call_gemini_with_retry(prompt, PredictionInput)
    except Exception as e:
        st.error(f"Couldn't reach Gemini after multiple attempts: {e}")
        st.stop()

    if response.parsed is None:
        st.error("Gemini couldn't produce valid model inputs from these requirements. Try different wording.")
        st.stop()

    return response.parsed


def resolve_product_name(sub_category: str, suggested_name: str) -> str:
    catalog = data_ctx["product_catalog"]
    matches = catalog[catalog["Sub-Category"] == sub_category]["Product Name"]
    if suggested_name in matches.values:
        return suggested_name
    return matches.iloc[0] if len(matches) else suggested_name


def render_prediction_wizard():
    if st.session_state.final_result:
        stage = 3
    elif st.session_state.show_selection:
        stage = 2
    else:
        stage = 1

    def step_state(n):
        if n < stage:
            return "done"
        if n == stage:
            return "active"
        return "todo"

    def circle(n):
        s = step_state(n)
        content = "✓" if s == "done" else str(n)
        return f'<div class="step-circle {s}">{content}</div>'

    def label(n, text):
        s = "active" if step_state(n) == "active" else ""
        return f'<div class="step-text {s}">{text}</div>'

    def line(n):
        s = "done" if n < stage else ""
        return f'<div class="step-line {s}"></div>'

    st.markdown(
        '<div class="stepper">'
        + circle(1) + label(1, "Describe")
        + line(1)
        + circle(2) + label(2, "Select")
        + line(2)
        + circle(3) + label(3, "Result")
        + '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Step 1 — Describe what you want predicted")

    with st.form("add_requirement_form", clear_on_submit=True):
        new_requirement = st.text_area(
            "Requirement",
            placeholder="e.g. Predict sales for a high quantity order",
            height=70,
            label_visibility="collapsed"
        )
        add_clicked = st.form_submit_button("Add requirement")

    if add_clicked and new_requirement.strip():
        st.session_state.requirements.append(new_requirement.strip())
        st.session_state.interpretations = []
        st.session_state.show_selection = False
        st.session_state.final_result = None

    continue_clicked = False
    if st.session_state.requirements:
        for req in st.session_state.requirements:
            st.markdown(f'<div class="req-item">{req}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            continue_clicked = st.button("Continue", use_container_width=True)
        with col2:
            if st.button("Clear all", use_container_width=True):
                st.session_state.requirements = []
                st.session_state.interpretations = []
                st.session_state.show_selection = False
                st.session_state.final_result = None
                st.rerun()
    else:
        st.caption("Add at least one requirement to continue.")
    st.markdown('</div>', unsafe_allow_html=True)

    if continue_clicked:
        with st.spinner("Analyzing your requirements..."):
            st.session_state.interpretations = interpret_requirements(st.session_state.requirements)
        st.session_state.show_selection = True
        st.rerun()

    selected_requirements = []
    submit_clicked = False

    if st.session_state.show_selection:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 2 — Choose which requirements to use")

        for i, req in enumerate(st.session_state.requirements):
            checked = st.checkbox(req, value=True, key=f"select_{i}")
            if checked:
                selected_requirements.append(req)

        submit_clicked = st.button("Get prediction", disabled=(len(selected_requirements) == 0))
        st.markdown('</div>', unsafe_allow_html=True)

    if submit_clicked and selected_requirements:
        with st.spinner("Computing prediction..."):
            pred_input = build_prediction_input(selected_requirements)

            safe_product_name = resolve_product_name(pred_input.sub_category, pred_input.product_name)
            safe_year = min(max(pred_input.order_year, data_ctx["min_year"]), data_ctx["max_year"])

            new_order = pd.DataFrame({
                "Quantity": [pred_input.quantity],
                "Discount": [pred_input.discount],
                "Profit": [pred_input.profit],
                "Segment": [pred_input.segment],
                "Region": [pred_input.region],
                "Ship Mode": [pred_input.ship_mode],
                "Sub-Category": [pred_input.sub_category],
                "Product Name": [safe_product_name],
                "Order_Year": [safe_year],
                "Order_Month": [pred_input.order_month],
                "Order_Day": [pred_input.order_day],
                "Order_DayOfWeek": [pd.Timestamp(
                    year=safe_year, month=pred_input.order_month, day=pred_input.order_day
                ).dayofweek],
            })

            chosen_pipeline = models[pred_input.model_choice]
            prediction = chosen_pipeline.predict(new_order)[0]

            st.session_state.final_result = {
                "prediction": prediction,
                "model_used": pred_input.model_choice,
                "reasoning": pred_input.reasoning,
                "inputs": new_order,
                "used_requirements": selected_requirements,
            }
        st.rerun()

    if st.session_state.final_result:
        result = st.session_state.final_result

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 3 — Result")

        col1, col2 = st.columns(2)
        col1.metric("Predicted sales", f"${result['prediction']:,.2f}")
        col2.metric("Model used", result["model_used"])

        st.markdown("**Requirements used**")
        for r in result["used_requirements"]:
            st.markdown(f'<div class="req-item">{r}</div>', unsafe_allow_html=True)

        with st.expander("Why these values were chosen"):
            st.write(result["reasoning"])

        with st.expander("Exact input sent to the model"):
            st.dataframe(result["inputs"], use_container_width=True)

        if st.button("Start a new prediction"):
            st.session_state.requirements = []
            st.session_state.interpretations = []
            st.session_state.show_selection = False
            st.session_state.final_result = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 7. AI ANALYST — TOOLS GEMINI CAN CALL FOR ITSELF
# ==========================================
# Each function below is a real Python function that computes something
# from your actual data or actual trained model. We hand Gemini the
# LIST of functions (not their results) as "tools". Gemini reads each
# function's name, type hints, and docstring to decide when a question
# needs it, calls it with arguments it chooses, and the SDK runs the
# real Python code and feeds the real return value back to Gemini
# automatically. This is what stops it from inventing numbers.

def get_top_products(metric: str, top_n: int) -> str:
    """Get the top-performing products ranked by a metric, computed from
    the full historical order dataset.

    Args:
        metric: Which metric to rank by. Must be one of: "sales", "profit", "quantity".
        top_n: How many top products to return, e.g. 5.
    """
    sort_col = {"sales": "total_sales", "profit": "total_profit", "quantity": "total_quantity"}.get(
        metric.lower(), "total_sales"
    )
    grouped = df.groupby("Product Name").agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_quantity=("Quantity", "sum"),
    ).reset_index()
    top = grouped.sort_values(sort_col, ascending=False).head(top_n)

    lines = []
    for _, r in top.iterrows():
        prod = r["Product Name"]
        yearly = df[df["Product Name"] == prod].groupby("Order_Year")["Sales"].sum().sort_index()
        growth_note = ""
        if len(yearly) >= 2 and yearly.iloc[-2] != 0:
            pct = (yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100
            growth_note = f", year-over-year growth {pct:+.1f}%"
        lines.append(
            f"- {prod}: total sales ${r['total_sales']:.2f}, "
            f"total profit ${r['total_profit']:.2f}, units sold {int(r['total_quantity'])}{growth_note}"
        )
    return f"Top {top_n} products by {metric}:\n" + "\n".join(lines)


def get_sales_trend(product_name: str) -> str:
    """Get year-by-year historical sales for one specific product, to see
    if it's growing, shrinking, or flat over time.

    Args:
        product_name: The exact product name to look up.
    """
    subset = df[df["Product Name"] == product_name]
    if subset.empty:
        return f"No historical data found for a product named '{product_name}'."
    yearly = subset.groupby("Order_Year")["Sales"].sum().sort_index()
    lines = [f"{yr}: ${val:.2f}" for yr, val in yearly.items()]
    return f"Yearly sales for '{product_name}':\n" + "\n".join(lines)


def get_category_performance() -> str:
    """Get total sales, total profit, and average discount for every
    product Sub-Category, sorted from best to worst by sales."""
    grouped = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        avg_discount=("Discount", "mean"),
    ).reset_index().sort_values("total_sales", ascending=False)

    lines = [
        f"- {r['Sub-Category']}: sales ${r['total_sales']:.2f}, "
        f"profit ${r['total_profit']:.2f}, avg discount {r['avg_discount']*100:.1f}%"
        for _, r in grouped.iterrows()
    ]
    return "Category performance:\n" + "\n".join(lines)


def get_region_performance() -> str:
    """Get total sales and total profit for each geographic Region."""
    grouped = df.groupby("Region").agg(
        total_sales=("Sales", "sum"), total_profit=("Profit", "sum")
    ).reset_index().sort_values("total_sales", ascending=False)
    lines = [
        f"- {r['Region']}: sales ${r['total_sales']:.2f}, profit ${r['total_profit']:.2f}"
        for _, r in grouped.iterrows()
    ]
    return "Regional performance:\n" + "\n".join(lines)


def get_overall_growth_trend() -> str:
    """Get total company-wide sales for every year on record, to show the
    overall business growth trend (not specific to any one product)."""
    yearly = df.groupby("Order_Year")["Sales"].sum().sort_index()
    lines = [f"{yr}: ${val:.2f}" for yr, val in yearly.items()]
    return "Company-wide sales by year:\n" + "\n".join(lines)


def predict_next_year_sales(product_name: str) -> str:
    """Uses the trained General Model to predict the Sales value for ONE
    typical/representative order of the given product next year. This is
    a model estimate for a single order line, NOT total annual revenue
    for the product — make that distinction clear when using this result.

    Args:
        product_name: The exact product name to predict for.
    """
    matches = df[df["Product Name"] == product_name]
    if matches.empty:
        return f"No historical data found for a product named '{product_name}'. Cannot predict."

    sub_category = matches["Sub-Category"].mode()[0]
    next_year = data_ctx["max_year"] + 1
    row = pd.DataFrame({
        "Quantity": [data_ctx["median_quantity"]],
        "Discount": [data_ctx["median_discount"]],
        "Profit": [data_ctx["median_profit"]],
        "Segment": [data_ctx["most_common_segment"]],
        "Region": [data_ctx["most_common_region"]],
        "Ship Mode": [data_ctx["most_common_ship_mode"]],
        "Sub-Category": [sub_category],
        "Product Name": [product_name],
        "Order_Year": [next_year],
        "Order_Month": [6],
        "Order_Day": [15],
        "Order_DayOfWeek": [pd.Timestamp(year=next_year, month=6, day=15).dayofweek],
    })
    prediction = models["General Model"].predict(row)[0]
    return (
        f"Predicted Sales for ONE typical order of '{product_name}' in {next_year}: "
        f"${prediction:.2f} (model estimate for a single representative order, "
        f"not the product's total annual revenue)."
    )


ANALYST_TOOLS = [
    get_top_products,
    get_sales_trend,
    get_category_performance,
    get_region_performance,
    get_overall_growth_trend,
    predict_next_year_sales,
]

ANALYST_SYSTEM_INSTRUCTION = """
You are an AI data analyst for a furniture retail company. Business
stakeholders ask you natural-language questions about products, sales,
trends, and where to focus effort next.

Rules:
- ALWAYS call one or more of the available tools to retrieve real numbers
  before answering. Never invent statistics.
- Ground every specific claim (a number, a trend, a ranking) in data a
  tool actually returned.
- When recommending a product or action, explain WHY using the specific
  numbers you retrieved (sales, profit, growth, trend) — don't just state
  a conclusion.
- Be clear about the difference between historical totals and model
  predictions for a single order — don't conflate the two.
- Keep the tone like a helpful analyst briefing a stakeholder: clear,
  structured, roughly 4-8 sentences. No raw code, JSON, or bullet-dumping
  of tool output — synthesize it into prose.
"""


def create_analyst_chat():
    last_error = None
    for model_name in MODEL_FALLBACK_CHAIN:
        try:
            return client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=ANALYST_SYSTEM_INSTRUCTION,
                    tools=ANALYST_TOOLS,
                ),
            )
        except Exception as e:
            last_error = e
    raise last_error


def send_analyst_message(chat, question: str, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chat.send_message(question)
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)
    raise last_error


def render_analyst_tab():
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown("#### Ask the AI analyst")
    st.caption(
        "Ask a business question, e.g. \"which product should we focus on next year?\" "
        "The analyst looks up real numbers from your data before answering."
    )

    if "analyst_chat" not in st.session_state:
        st.session_state.analyst_chat = create_analyst_chat()
        st.session_state.analyst_messages = []

    for msg in st.session_state.analyst_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about your sales data...")
    if question:
        st.session_state.analyst_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Looking up the data and analyzing..."):
                try:
                    response = send_analyst_message(st.session_state.analyst_chat, question)
                    st.write(response.text)
                    st.session_state.analyst_messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Couldn't get an answer: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 8. TABS
# ==========================================

tab_dashboard, tab_predict, tab_analyst = st.tabs(
    ["Dataset overview", "Make a prediction", "Ask the analyst"]
)

with tab_dashboard:
    render_dashboard(df)

with tab_predict:
    render_prediction_wizard()

with tab_analyst:
    render_analyst_tab()