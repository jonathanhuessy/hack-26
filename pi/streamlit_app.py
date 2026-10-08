"""Modern browser UI for the interactive Python plant and Pi stream."""

from __future__ import annotations

import numpy as np

try:
    from .dashboard_session import (
        CLASS_NAMES,
        CLASS_LABELS,
        DashboardSession as SharedDashboardSession,
        build_stream as shared_build_stream,
        LIVE_UPDATE_SECONDS,
        MAX_DISPLAY_SAMPLES,
        MAX_PLOT_POINTS,
        PATH_COLORS,
        trajectory_options as shared_trajectory_options,
    )
except ImportError:  # pragma: no cover
    from dashboard_session import (
        CLASS_NAMES,
        CLASS_LABELS,
        DashboardSession as SharedDashboardSession,
        build_stream as shared_build_stream,
        LIVE_UPDATE_SECONDS,
        MAX_DISPLAY_SAMPLES,
        MAX_PLOT_POINTS,
        PATH_COLORS,
        trajectory_options as shared_trajectory_options,
    )

build_stream = shared_build_stream
trajectory_options = shared_trajectory_options

def _session(st):
    # Retain this entry point as a compatibility fallback for existing
    # Streamlit operators.
    options = shared_trajectory_options()
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = SharedDashboardSession("Python default", options)
    elif st.session_state.dashboard.options != options:
        st.session_state.dashboard.options = options
        if st.session_state.dashboard.selection not in options:
            st.session_state.dashboard.switch("Python default")
    return st.session_state.dashboard


def _plotly_figures(session: DashboardSession):
    import plotly.graph_objects as go

    if not session.samples:
        return []
    samples = session.samples[-MAX_DISPLAY_SAMPLES:]
    states = session.path_states[-MAX_DISPLAY_SAMPLES:]
    stride = max(1, int(np.ceil(len(samples) / MAX_PLOT_POINTS)))
    samples = samples[::stride]
    states = states[::stride]
    time_s = np.asarray([sample.timestamp_s for sample in samples])
    x = np.asarray([sample.diagnostics.get("X", 0.0) for sample in samples])
    y = np.asarray([sample.diagnostics.get("Y", 0.0) for sample in samples])
    delta = np.asarray([sample.delta for sample in samples])
    vx = np.asarray([sample.vx for sample in samples])
    yaw = np.asarray([sample.yaw_rate for sample in samples])
    ay = np.asarray([sample.diagnostics.get("ay", 0.0) for sample in samples])

    path = go.Figure()
    for state, color in PATH_COLORS.items():
        px = [value if current == state else None for value, current in zip(x, states)]
        py = [value if current == state else None for value, current in zip(y, states)]
        path.add_trace(go.Scatter(x=px, y=py, mode="lines", name=state))
    path.update_layout(title="Vehicle path", xaxis_title="X [m]", yaxis_title="Y [m]")

    edge = go.Figure()
    edge.add_trace(go.Scatter(x=time_s, y=delta, name="delta"))
    edge.add_trace(go.Scatter(x=time_s, y=vx, name="Vx"))
    edge.update_layout(title="Edge signals", xaxis_title="time [s]")

    plant = go.Figure()
    plant.add_trace(go.Scatter(x=time_s, y=yaw, name="r"))
    plant.add_trace(go.Scatter(x=time_s, y=ay, name="ay"))
    plant.update_layout(title="Plant signals", xaxis_title="time [s]")

    confidence = go.Figure()
    if session.verdicts:
        verdict_times = [item[0] for item in session.verdicts]
        probs = np.asarray([item[1].class_probabilities for item in session.verdicts])
        for index, name in enumerate(CLASS_NAMES):
            confidence.add_trace(go.Scatter(
                x=verdict_times,
                y=probs[:, index],
                mode="lines+markers",
                name=CLASS_LABELS.get(name, name),
            ))
    confidence.update_layout(title="Classifier probabilities", yaxis_range=[0, 1])

    mass = go.Figure()
    stiffness = go.Figure()
    sample_count = (
        len(session.stream.measured)
        if hasattr(session.stream, "measured")
        else len(session.stream.delta)
    )
    end_time = (sample_count - 1) * session.stream.period_s
    metadata = getattr(session.stream, "metadata", {})
    class_name = str(metadata.get("className", "nominal")).upper()
    change_time = metadata.get("changeTimeS")
    if class_name in {"A", "B", "AB"} and change_time is not None:
        change_time = float(change_time)
        added_mass = float(metadata.get("addedMassKg", 0.0))
        kf = float(metadata.get("kf", 1.0))
        reference_time = [0.0, change_time, change_time, end_time]
        mass_reference = [0.0, 0.0, added_mass, added_mass]
        stiffness_reference = [1.0, 1.0, kf, kf]
        mass.add_trace(go.Scatter(
            x=reference_time,
            y=mass_reference,
            mode="lines",
            name="True reference",
            line={"dash": "dash", "color": "#d62728"},
        ))
        stiffness.add_trace(go.Scatter(
            x=reference_time,
            y=stiffness_reference,
            mode="lines",
            name="True reference",
            line={"dash": "dash", "color": "#d62728"},
        ))
    if session.verdicts:
        mass_points = [(t, v.delta_m_kg) for t, v in session.verdicts if v.delta_m_kg is not None]
        stiffness_points = [(t, v.k_f) for t, v in session.verdicts if v.k_f is not None]
        mass.add_trace(go.Scatter(
            x=[item[0] for item in mass_points],
            y=[item[1] for item in mass_points],
            mode="lines+markers",
            name="delta_m",
        ))
        stiffness.add_trace(go.Scatter(
            x=[item[0] for item in stiffness_points],
            y=[item[1] for item in stiffness_points],
            mode="lines+markers",
            name="k_f",
        ))
    mass.update_layout(
        title="delta_m estimate",
        xaxis_title="time [s]",
        yaxis_title="delta_m [kg]",
        xaxis_range=[0, end_time],
    )
    stiffness.update_layout(
        title="k_f estimate",
        xaxis_title="time [s]",
        yaxis_title="k_f",
        xaxis_range=[0, end_time],
    )
    return path, edge, plant, confidence, mass, stiffness


def _render_dashboard(st, session: DashboardSession) -> None:
    figures = _plotly_figures(session)
    if not figures:
        st.info("Press Start to stream the selected trajectory.")
        return
    st.markdown("#### Telemetry")
    columns = st.columns(2)
    for figure, column in zip(figures, columns * 3):
        with column:
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_live(st, session: DashboardSession) -> None:
    """Render one bounded telemetry update without rerunning the controls."""
    session.drain()
    session.commands = list(getattr(session.stream.schedule, "history", ()))
    configuration = session.configuration_status()

    overview = st.columns([1, 1.35, 1])
    with overview[0].container(border=True):
        st.markdown("#### Run status")
        run_cols = st.columns(2)
        run_cols[0].metric("Source", "MATLAB" if not session.supports_events else "Python")
        run_cols[1].metric("Playback", f"{session.speed:.1f}x")
        run_cols[0].metric("Time", f"{session.stream.time_s:.1f}s")
    classifier_status = (
        "Detected Change"
        if session.classifier_class not in (None, "nominal")
        else "No Change"
    )
    run_cols[1].metric("Detected Change", classifier_status)

    with overview[1].container(border=True):
        st.markdown("#### Tractor Configuration")
        implement_state = "Implement Attached" if configuration["implement"] else "No Implement"
        tire_state = "FLAT" if configuration["tire"] else "NORMAL"
        implement_class = "changed" if configuration["implement"] else "nominal"
        tire_class = "changed" if configuration["tire"] else "nominal"
        st.markdown(
            f'<div class="configuration-status {implement_class}">'
            f'<span>Implement</span><strong>{implement_state}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="configuration-status {tire_class}">'
            f'<span>Tire</span><strong>{tire_state}</strong></div>',
            unsafe_allow_html=True,
        )

    with overview[2].container(border=True):
        st.markdown("#### Detected Change")
        class_cols = st.columns(2)
        for column, name in zip(class_cols * 2, CLASS_NAMES):
            if name == session.classifier_class:
                column.success(CLASS_LABELS[name])
            else:
                column.info(CLASS_LABELS[name])

    _render_dashboard(st, session)
    if session.runner.errors:
        st.error(str(session.runner.errors[-1]))


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required for this UI. Install PC UI dependencies first."
        ) from exc

    st.set_page_config(page_title="Tractor Edge Demo", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
            line-height: 1.2;
        }
        [data-testid="stMetric"] {
            padding: 0;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.65rem 0.8rem 0.45rem;
        }
        .configuration-status {
            align-items: center;
            border-radius: 0.35rem;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            padding: 0.5rem 0.65rem;
        }
        .configuration-status.nominal {
            background: rgba(46, 160, 67, 0.18);
            color: #5bd276;
        }
        .configuration-status.changed {
            background: rgba(214, 39, 40, 0.2);
            color: #ff7777;
        }
        .configuration-status span {
            color: #f0f0f0;
            font-weight: 600;
        }
        .configuration-status strong {
            letter-spacing: 0.03em;
        }
        .plant-status {
            border-radius: 0.35rem;
            padding: 0.35rem 0.55rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-align: center;
        }
        .plant-status.nominal {
            background: rgba(46, 160, 67, 0.18);
            color: #5bd276;
        }
        .plant-status.perturbed {
            background: rgba(214, 39, 40, 0.2);
            color: #ff7777;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    session = _session(st)
    st.title("Tractor Edge Detection")

    with st.sidebar:
        st.header("Controls")
        selection = st.selectbox(
            "Trajectory",
            list(session.options),
            index=list(session.options).index(session.selection),
        )
        if selection != session.selection:
            session.switch(selection)
            st.rerun()
        speed = st.slider("Playback speed", 0.1, 20.0, session.speed, 0.1)
        session.set_speed(speed)
        host = st.text_input("Pi host (optional)", value=session.tcp_host)
        port = st.number_input("Pi port", 1, 65535, session.tcp_port)
        capture = st.text_input("Capture path (optional)")
        session.configure_transport(host, port, capture)
        st.subheader("Playback")
        if st.button("Start", use_container_width=True):
            session.start()
        if st.button("Pause", use_container_width=True):
            session.pause()
        if st.button("Resume", use_container_width=True):
            session.resume()
        if st.button("Reset", use_container_width=True):
            session.reset()
        if st.button("Stop", use_container_width=True):
            session.stop()
        st.caption(
            f"{'fixed MATLAB playback' if not session.supports_events else 'Python synthesized'}; "
            f"{session.stream.time_s:.1f}s / "
            f"{((len(session.stream.measured) - 1) * session.stream.period_s if hasattr(session.stream, 'measured') else (len(session.stream.delta) - 1) * session.stream.period_s):.1f}s"
        )
    if hasattr(st, "fragment"):
        @st.fragment(run_every=LIVE_UPDATE_SECONDS)
        def render_live_fragment():
            _render_live(st, session)

        render_live_fragment()
    else:  # pragma: no cover - compatibility with older Streamlit versions
        _render_live(st, session)


if __name__ == "__main__":
    main()
