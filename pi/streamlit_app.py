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
        PATH_LABELS,
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
        PATH_LABELS,
        trajectory_options as shared_trajectory_options,
    )

build_stream = shared_build_stream
trajectory_options = shared_trajectory_options


def run_status_text(session) -> str:
    """Return the operator-facing state for the current run."""
    if not session.running or not session.samples:
        return "Waiting for run"
    return "Running"


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
        path.add_trace(go.Scatter(
            x=px,
            y=py,
            mode="lines",
            name=PATH_LABELS.get(state, state),
        ))
    path.update_layout(title="Vehicle path", xaxis_title="X [m]", yaxis_title="Y [m]")

    edge = go.Figure()
    edge.add_trace(go.Scatter(x=time_s, y=delta, name="Steering Angle"))
    edge.add_trace(go.Scatter(x=time_s, y=vx, name="Forward Speed"))
    edge.update_layout(title="Edge signals", xaxis_title="time [s]")

    plant = go.Figure()
    plant.add_trace(go.Scatter(x=time_s, y=yaw, name="Yaw Rate"))
    plant.add_trace(go.Scatter(x=time_s, y=ay, name="Lateral Acceleration"))
    plant.update_layout(title="Tractor Signals", xaxis_title="time [s]")

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
    added_mass = (
        float(metadata.get("addedMassKg", 0.0))
        if class_name in {"A", "AB"}
        else 0.0
    )
    kf = (
        float(metadata.get("kf", 1.0))
        if class_name in {"B", "AB"}
        else 1.0
    )
    if class_name in {"A", "B", "AB"} and change_time is not None:
        change_time = float(change_time)
        reference_time = [0.0, change_time, change_time, end_time]
        mass_reference = [0.0, 0.0, added_mass, added_mass]
        stiffness_reference = [1.0, 1.0, kf, kf]
    else:
        reference_time = [0.0, end_time]
        mass_reference = [0.0, 0.0]
        stiffness_reference = [1.0, 1.0]
    mass.add_trace(go.Scatter(
        x=reference_time,
        y=mass_reference,
        mode="lines",
        name="Reference",
        line={"dash": "dash", "color": "#d62728"},
    ))
    stiffness.add_trace(go.Scatter(
        x=reference_time,
        y=stiffness_reference,
        mode="lines",
        name="Reference",
        line={"dash": "dash", "color": "#d62728"},
    ))

    def estimate_series(attribute: str, baseline: float) -> tuple[list[float], list[float]]:
        times = [0.0]
        values = [baseline]
        current = baseline
        for timestamp, verdict in session.verdicts:
            estimate = getattr(verdict, attribute)
            if estimate is None:
                continue
            timestamp = float(timestamp)
            times.extend((timestamp, timestamp))
            values.extend((current, float(estimate)))
            current = float(estimate)
        times.append(end_time)
        values.append(current)
        return times, values

    mass_time, mass_estimate = estimate_series("delta_m_kg", 0.0)
    stiffness_time, stiffness_estimate = estimate_series("k_f", 1.0)
    mass.add_trace(go.Scatter(
        x=mass_time,
        y=mass_estimate,
        mode="lines",
        name="Current Estimate",
    ))
    stiffness.add_trace(go.Scatter(
        x=stiffness_time,
        y=stiffness_estimate,
        mode="lines",
        name="Current Estimate",
    ))
    mass.update_layout(
        title="Δ Mass",
        xaxis_title="time [s]",
        yaxis_title="Δ Mass [kg]",
        xaxis_range=[0, end_time],
    )
    stiffness.update_layout(
        title="Front Cornering Stiffness Estimate",
        xaxis_title="time [s]",
        yaxis_title="Front Cornering Stiffness Factor",
        xaxis_range=[0, end_time],
    )
    return path, edge, plant, confidence, mass, stiffness


def _render_dashboard(st, session: DashboardSession) -> None:
    if run_status_text(session) == "Waiting for run":
        st.info("Waiting for run")
        return
    figures = _plotly_figures(session)
    if not figures:
        st.info("Waiting for run")
        return
    st.markdown("#### Telemetry")
    columns = st.columns(2)
    for figure, column in zip(figures, columns * 3):
        with column:
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _render_live(st, session: DashboardSession) -> None:
    """Render one bounded telemetry update without rerunning the controls."""
    session._sync_transport_status()
    session.drain()
    session.commands = list(getattr(session.stream.schedule, "history", ()))
    configuration = session.configuration_status()
    transport_stats = session.transport_stats()

    overview = st.columns([1, 1.35, 1])
    with overview[0].container(border=True):
        st.markdown("#### Run status")
        run_cols = st.columns(2)
        run_cols[0].metric("Source", "MATLAB" if not session.supports_events else "Python")
        run_cols[1].metric("Playback", f"{session.speed:.1f}x")
        run_cols[0].metric("Time", f"{session.stream.time_s:.1f}s")
        st.caption(
            f"Samples produced: {session.processed_samples} | "
            f"sent to Pi: {transport_stats['sent']} | "
            f"queued: {transport_stats['queued']}"
        )
        if session.tcp_host:
            st.caption(f"Transport: {session.transport_state} — {session.transport_message}")
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
        st.subheader("Direct Pi connection")
        host = st.text_input("Pi host or IP (optional)", value=session.tcp_host)
        port = st.number_input("Pi port", 1, 65535, session.tcp_port)
        capture = st.text_input("Capture path (optional)")
        session.configure_transport(host, port, capture)
        if session.tcp_host:
            st.caption(f"Status: {session.transport_state}")
            st.caption(session.transport_message)
        else:
            st.caption("Status: local playback only")
        st.subheader("Playback")
        if st.button("Start streaming", disabled=session.running, width="stretch"):
            session.start()
        if st.button("Pause", width="stretch"):
            session.pause()
        if st.button("Resume", width="stretch"):
            session.resume()
        if st.button("Reset", width="stretch"):
            session.reset()
        if st.button("Stop streaming", disabled=not session.running, width="stretch"):
            session.stop()
        if session.runner.errors and st.button(
            "Retry from beginning",
            disabled=session.running,
            width="stretch",
        ):
            session.retry()
            st.rerun()
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
