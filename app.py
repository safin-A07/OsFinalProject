"""
Intelligent CPU Scheduling Advisor and Simulator
Main Streamlit Application Entrypoint.
Handles the 7 core sections of the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import custom modules
from data.sample_data import get_sample_datasets
from utils.helpers import validate_process_data
from algorithms.fcfs import schedule_fcfs
from algorithms.sjf import schedule_sjf
from algorithms.priority import schedule_priority
from algorithms.round_robin import schedule_round_robin
from metrics.calculator import compute_metrics, compare_all
from recommendation.engine import get_recommendations
from visualization.gantt import plot_gantt
from visualization.charts import plot_comparison_bar, plot_radar_comparison
from ai.gemini_explainer import get_explanation

# --- Page Configuration ---
st.set_page_config(
    page_title="CPU Scheduling Advisor & Simulator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Section Divider Utility ---
# Used for dividing sections visually and keeping long script readable.

# --- Styling & CSS ---
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700;800&display=swap');

/* Typography override */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Custom dark card panels */
.custom-card {
    background-color: #161616;
    border: 1px solid #2d2d2d;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}

/* Recommendation card styling */
.recommendation-card {
    background-color: #1c1c1e;
    border: 1px solid #3a3a3c;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 15px;
    min-height: 180px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s;
}
.recommendation-card:hover {
    transform: translateY(-2px);
    border-color: #00CC96;
}
.recommendation-title {
    font-size: 13px;
    color: #98989d;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.recommendation-algo {
    font-size: 22px;
    color: #00CC96;
    font-weight: 700;
    margin: 8px 0;
}
.recommendation-reason {
    font-size: 13px;
    color: #aeaea2;
    line-height: 1.4;
}

/* AI Explanation Box */
.ai-box {
    background: linear-gradient(135deg, #111827 0%, #030712 100%);
    border: 1px solid #1f2937;
    border-left: 5px solid #00CC96;
    border-radius: 10px;
    padding: 25px;
    margin-top: 20px;
    color: #f3f4f6;
    box-shadow: 0 4px 20px rgba(0, 204, 150, 0.1);
    line-height: 1.6;
}

/* Metric styling */
div[data-testid="stMetric"] {
    background-color: #1c1c1e;
    border: 1px solid #2c2c2e;
    padding: 12px 18px;
    border-radius: 8px;
    box-shadow: inset 0 1px 1px rgba(255,255,255,0.05);
}
div[data-testid="stMetric"] label {
    font-size: 11px !important;
    color: #aeaeae !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: bold;
    color: #ffffff;
}

/* Streamlit button style overrides */
div.stButton > button {
    border-radius: 8px;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# --- Section 1 — Header ---
# ==============================================================================
st.title("💻 Intelligent CPU Scheduling Advisor & Simulator")
st.caption("A university-level Decision Support System & simulation advisor designed to explain and compare process execution scheduling models.")
st.write(
    "This system simulates four core CPU scheduling algorithms, evaluates their performance under different "
    "workload characteristics, recommends optimal strategies for various environments, and generates natural language explanations using AI."
)
st.markdown("---")


# ==============================================================================
# --- Section 2 — Process Input Panel ---
# ==============================================================================
st.header("📥 Process Configuration Panel")

# Initialize default process dataset in session state if not present
if "processes_df" not in st.session_state:
    samples = get_sample_datasets()
    default_data = samples["Basic (4 Processes)"]
    st.session_state.processes_df = pd.DataFrame([
        {
            "PID": p["pid"],
            "Arrival Time": float(p["arrival"]),
            "Burst Time": float(p["burst"]),
            "Priority": int(p["priority"])
        } for p in default_data
    ])

# Input Layout columns
col_editor, col_controls = st.columns([2, 1])

with col_editor:
    st.subheader("Process Queue Table")
    # Data editor with safety inputs configuration
    edited_df = st.data_editor(
        st.session_state.processes_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "PID": st.column_config.TextColumn("Process ID (PID)", required=True),
            "Arrival Time": st.column_config.NumberColumn("Arrival Time", min_value=0.0, format="%.2f", required=True),
            "Burst Time": st.column_config.NumberColumn("Burst Time (CPU Run)", min_value=0.1, format="%.2f", required=True),
            "Priority": st.column_config.NumberColumn("Priority (Lower = Higher)", min_value=1, step=1, required=True),
        }
    )
    # Save edits back to session state to prevent loss on re-runs
    st.session_state.processes_df = edited_df

with col_controls:
    st.subheader("Scheduling Parameters")
    # Round Robin Quantum Slider
    quantum = st.slider(
        "Round Robin Quantum (Time Slice)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="The time slice length allocated for Round Robin scheduling."
    )
    
    # Priority Direction Toggle
    priority_direction = st.checkbox(
        "Lower Priority Number = Higher Priority",
        value=True,
        help="Standard UNIX schedules treat priority 1 as highest, while other systems treat larger numbers as higher."
    )

    # Actions buttons
    st.write("Table Control Actions:")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("➕ Add Process", use_container_width=True):
            next_idx = len(st.session_state.processes_df) + 1
            new_row = pd.DataFrame([{
                "PID": f"P{next_idx}",
                "Arrival Time": 0.0,
                "Burst Time": 5.0,
                "Priority": 3
            }])
            st.session_state.processes_df = pd.concat([st.session_state.processes_df, new_row], ignore_index=True)
            st.rerun()
            
    with btn_col2:
        if st.button("❌ Delete Last Row", use_container_width=True):
            if len(st.session_state.processes_df) > 1:
                st.session_state.processes_df = st.session_state.processes_df.iloc[:-1]
                st.rerun()
            else:
                st.warning("Cannot delete the last remaining process.")

    st.write("Load Simulation Presets:")
    samples = get_sample_datasets()
    sample_choice = st.selectbox("Choose pre-defined dataset:", list(samples.keys()))
    if st.button("📂 Load Preset Dataset", use_container_width=True):
        selected_data = samples[sample_choice]
        st.session_state.processes_df = pd.DataFrame([
            {
                "PID": p["pid"],
                "Arrival Time": float(p["arrival"]),
                "Burst Time": float(p["burst"]),
                "Priority": int(p["priority"])
            } for p in selected_data
        ])
        st.success(f"Successfully loaded '{sample_choice}' dataset.")
        st.rerun()

# Run validation before calculations
process_list = []
for _, row in st.session_state.processes_df.iterrows():
    process_list.append({
        "pid": str(row["PID"]).strip(),
        "arrival": float(row["Arrival Time"]),
        "burst": float(row["Burst Time"]),
        "priority": int(row["Priority"])
    })

is_valid, validation_error = validate_process_data(process_list)

if not is_valid:
    st.error(f"❌ Input Validation Error: {validation_error}")
    st.info("Please correct the entries in the Process Queue Table before continuing.")
else:
    st.markdown("---")

    # ==============================================================================
    # --- Section 3 — Workload Summary ---
    # ==============================================================================
    st.header("📊 Input Workload Analysis Summary")
    
    total_processes = len(process_list)
    bursts = [p["burst"] for p in process_list]
    arrivals = [p["arrival"] for p in process_list]
    priorities = [p["priority"] for p in process_list]

    # Display Metrics Row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Total Processes", total_processes)
    metric_col2.metric("Average Burst Time", f"{np.mean(bursts):.2f} units")
    metric_col3.metric("Burst Range (Min/Max)", f"{np.min(bursts):.1f} / {np.max(bursts):.1f}")
    metric_col4.metric("Arrival Interval", f"{np.min(arrivals):.1f} - {np.max(arrivals):.1f}")

    # Display Workload Visualizations
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Burst time per process bar chart
        fig_burst = px.bar(
            x=[p["pid"] for p in process_list],
            y=bursts,
            labels={"x": "Process PID", "y": "Burst Time"},
            title="Burst Time Profile (CPU Cycle Demands)"
        )
        fig_burst.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            xaxis=dict(gridcolor="#222222"),
            yaxis=dict(gridcolor="#222222"),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_burst, use_container_width=True)

    with chart_col2:
        # Priority distribution pie chart
        df_prio = pd.DataFrame(process_list)
        prio_counts = df_prio["priority"].value_counts().reset_index()
        prio_counts.columns = ["Priority Level", "Count"]
        prio_counts["Priority Level"] = prio_counts["Priority Level"].astype(str)
        
        fig_prio = px.pie(
            prio_counts,
            values="Count",
            names="Priority Level",
            title="Process Priority Composition",
            hole=0.4
        )
        fig_prio.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_prio, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # --- Execute Simulations ---
    # ==============================================================================
    # Run all 4 algorithms to collect results and timelines
    results_fcfs, timeline_fcfs = schedule_fcfs(process_list)
    results_sjf, timeline_sjf = schedule_sjf(process_list)
    results_priority, timeline_priority = schedule_priority(process_list, lower_is_higher=priority_direction)
    results_rr, timeline_rr = schedule_round_robin(process_list, time_quantum=quantum)

    # Compute metrics for all runs
    metrics_fcfs = compute_metrics(results_fcfs)
    metrics_sjf = compute_metrics(results_sjf)
    metrics_priority = compute_metrics(results_priority)
    metrics_rr = compute_metrics(results_rr)

    # Build comparison map
    all_runs = {
        "FCFS": metrics_fcfs,
        "SJF": metrics_sjf,
        "Priority": metrics_priority,
        "Round Robin": metrics_rr
    }

    # ==============================================================================
    # --- Section 4 — Scheduling Results ---
    # ==============================================================================
    st.header("⚙️ Simulation Execution Details")
    
    # Set up 4 tabs for each algorithm
    tab_fcfs, tab_sjf, tab_priority, tab_rr = st.tabs([
        "🟢 First-Come, First-Served (FCFS)",
        "🔵 Shortest Job First (SJF)",
        "🟣 Priority Scheduling",
        "🟠 Round Robin (RR)"
    ])

    def render_algorithm_tab(results: list[dict], timeline: list[tuple], metrics: dict, title: str):
        """Helper to render standardized view inside each tab."""
        # Key metrics cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Waiting Time", f"{metrics['avg_waiting_time']:.2f}")
        c2.metric("Average Turnaround Time", f"{metrics['avg_turnaround_time']:.2f}")
        c3.metric("Throughput (Proc/Unit)", f"{metrics['throughput']:.4f}")
        c4.metric("CPU Utilization", f"{metrics['cpu_utilization']:.1f}%")

        # Per process results table
        st.subheader("Process Performance Log")
        df_table = pd.DataFrame(results)[[
            "pid", "arrival", "burst", "priority", "start_time", "completion_time", "waiting_time", "turnaround_time", "response_time"
        ]]
        df_table.columns = [
            "PID", "Arrival Time", "Burst Time", "Priority", "Start Time", "Completion Time", "Waiting Time", "Turnaround Time", "Response Time"
        ]
        
        # Color code by waiting time to highlight delayed processes
        styled_df = df_table.style.background_gradient(subset=["Waiting Time"], cmap="Reds")
        st.dataframe(styled_df, use_container_width=True)

        # Gantt chart Plotly
        st.subheader("Execution Timeline (Gantt Chart)")
        fig_gantt = plot_gantt(timeline, f"{title} Scheduling Gantt Chart")
        st.plotly_chart(fig_gantt, use_container_width=True)

    with tab_fcfs:
        render_algorithm_tab(results_fcfs, timeline_fcfs, metrics_fcfs, "FCFS")

    with tab_sjf:
        render_algorithm_tab(results_sjf, timeline_sjf, metrics_sjf, "SJF")

    with tab_priority:
        render_algorithm_tab(results_priority, timeline_priority, metrics_priority, "Priority")

    with tab_rr:
        render_algorithm_tab(results_rr, timeline_rr, metrics_rr, f"Round Robin (q={quantum})")

    st.markdown("---")


    # ==============================================================================
    # --- Section 5 — Performance Comparison ---
    # ==============================================================================
    st.header("⚖️ Comparative Performance Dashboard")
    
    # Build comparison DataFrame
    df_compare = compare_all(all_runs)

    # Display comparison DataFrame
    st.subheader("Comparative Metric Summary Table")
    st.dataframe(df_compare.style.highlight_min(subset=["Avg Waiting Time", "Avg Turnaround Time", "Avg Response Time"], color="#1f402b")
                                 .highlight_max(subset=["Throughput", "CPU Utilization (%)"], color="#1f402b"),
                 use_container_width=True)

    # Metrics Side-by-Side plots
    st.subheader("Visual Metric Comparison")
    c_bar1, c_bar2 = st.columns(2)
    with c_bar1:
        st.plotly_chart(
            plot_comparison_bar(df_compare, "Avg Waiting Time", "Avg Waiting Time (Lower is Better)", lower_is_better=True), 
            use_container_width=True
        )
        st.plotly_chart(
            plot_comparison_bar(df_compare, "Avg Response Time", "Avg Response Time (Lower is Better)", lower_is_better=True), 
            use_container_width=True
        )

    with c_bar2:
        st.plotly_chart(
            plot_comparison_bar(df_compare, "Avg Turnaround Time", "Avg Turnaround Time (Lower is Better)", lower_is_better=True), 
            use_container_width=True
        )
        st.plotly_chart(
            plot_comparison_bar(df_compare, "Throughput", "Throughput (Higher is Better)", lower_is_better=False), 
            use_container_width=True
        )

    # Bottom Radar comparison
    st.plotly_chart(plot_comparison_bar(df_compare, "CPU Utilization (%)", "CPU Utilization % (Higher is Better)", lower_is_better=False), use_container_width=True)
    st.plotly_chart(plot_radar_comparison(df_compare), use_container_width=True)

    st.markdown("---")


    # ==============================================================================
    # --- Section 6 — Recommendation Dashboard ---
    # ==============================================================================
    st.header("🎯 Rule-Based Optimal Scheduling Recommendations")
    st.write(
        "Based on theoretical operating systems principles and simulation results, "
        "here are the recommended algorithms for different scenarios:"
    )

    recommendations = get_recommendations(all_runs)

    # 4x2 grid of columns
    scenarios_list = list(recommendations.keys())
    
    # Row 1
    col_r1_1, col_r1_2, col_r1_3, col_r1_4 = st.columns(4)
    row1_cols = [col_r1_1, col_r1_2, col_r1_3, col_r1_4]
    for i in range(4):
        sname = scenarios_list[i]
        rec = recommendations[sname]
        with row1_cols[i]:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-title">{sname}</div>
                <div class="recommendation-algo">{rec['algorithm']}</div>
                <div class="recommendation-reason">{rec['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Row 2
    col_r2_1, col_r2_2, col_r2_3, col_r2_4 = st.columns(4)
    row2_cols = [col_r2_1, col_r2_2, col_r2_3, col_r2_4]
    for i in range(4):
        sname = scenarios_list[i+4]
        rec = recommendations[sname]
        with row2_cols[i]:
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="recommendation-title">{sname}</div>
                <div class="recommendation-algo">{rec['algorithm']}</div>
                <div class="recommendation-reason">{rec['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


    # ==============================================================================
    # --- Section 7 — AI Explanation Panel ---
    # ==============================================================================
    st.header("🤖 Educational AI Explanation Panel")
    st.write(
        "Generate a detailed academic-level performance analysis using the Google Gemini model. "
        "The generated report uses the simulation metrics to explain trade-offs and choices."
    )

    # Input for API key or load from secrets
    api_key_env = st.secrets.get("GEMINI_API_KEY", "")
    
    api_key_input = st.text_input(
        "Enter your Google Gemini API Key:",
        type="password",
        value="" if not api_key_env else "⚡ PRE-CONFIGURED IN SECRETS ⚡",
        help="Provide a Gemini API key. If already set up in Streamlit secrets (.streamlit/secrets.toml), this can be left as-is."
    )

    # Determine which key to use
    final_api_key = api_key_env if api_key_input == "⚡ PRE-CONFIGURED IN SECRETS ⚡" else api_key_input

    # Generation trigger button
    if st.button("✨ Generate AI Performance Report", type="primary"):
        if not final_api_key or final_api_key.strip() == "":
            st.warning("⚠️ Please input a valid Gemini API Key first.")
        else:
            with st.spinner("AI Professor is compiling the report and analysis..."):
                explanation_text = get_explanation(all_runs, recommendations, final_api_key)
                st.markdown(f"""
                <div class="ai-box">
                    {explanation_text}
                </div>
                """, unsafe_allow_html=True)
