"""Modern browser UI for the interactive Python plant and Pi stream."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

try:
    from .dashboard_session import (
        CLASS_NAMES,
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
    # The shared session is also used by Dash; retain this entry point as a
    # compatibility fallback for existing Streamlit operators.
    if "dashboard" not in st.session_state:
        options = shared_trajectory_options()
        st.session_state.dashboard = SharedDashboardSession("Python default", options)
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
                x=verdict_times, y=probs[:, index], mode="lines+markers", name=name.upper()
            ))
    confidence.update_layout(title="Classifier probabilities", yaxis_range=[0, 1])

    mass = go.Figure()
    stiffness = go.Figure()
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
    mass.update_layout(title="Added mass estimate", yaxis_title="kg")
    stiffness.update_layout(title="Front stiffness estimate", yaxis_title="k_f")
    return path, edge, plant, confidence, mass, stiffness


def _render_dashboard(st, session: DashboardSession) -> None:
    figures = _plotly_figures(session)
    if not figures:
        st.info("Press Start to stream the selected trajectory.")
        return
    columns = st.columns(2)
    for figure, column in zip(figures, columns * 3):
        column.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _render_live(st, session: DashboardSession) -> None:
    """Render one bounded telemetry update without rerunning the controls."""
    session.drain()
    session.commands = list(getattr(session.stream.schedule, "history", ()))
    active = set()
    if session.samples:
        active = set(filter(None, session.samples[-1].diagnostics.get("active_events", "").split(",")))
    cols = st.columns(6)
    cols[0].metric("Source", "MATLAB" if not session.supports_events else "Python")
    cols[1].metric("Playback", f"{session.speed:.1f}x")
    cols[2].metric("Time", f"{session.stream.time_s:.1f}s")
    cols[3].success("Implement ON" if active & {"implement_attached", "A", "AB"} else "Implement OFF")
    cols[4].error("Tire FLAT" if active & {"tire_flat", "B", "AB"} else "Tire NORMAL")
    cols[5].metric("Classifier", session.classifier_class or "waiting")

    class_cols = st.columns(4)
    for column, name in zip(class_cols, CLASS_NAMES):
        if name == session.classifier_class:
            column.success(f"CLASS {name.upper()}")
        else:
            column.info(f"CLASS {name.upper()}")

    _render_dashboard(st, session)
    if session.commands:
        st.subheader("Event command history")
        st.dataframe([asdict(command) for command in session.commands], use_container_width=True)
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
        st.subheader("Events")
        event_disabled = not session.supports_events
        for name, label in (("implement_attached", "Implement"), ("tire_flat", "Tire flat")):
            add, remove = st.columns(2)
            if add.button(f"Add {label}", disabled=event_disabled, use_container_width=True):
                session.event("add", name)
            if remove.button(f"Remove {label}", disabled=event_disabled, use_container_width=True):
                session.event("remove", name)

    if hasattr(st, "fragment"):
        @st.fragment(run_every=LIVE_UPDATE_SECONDS)
        def render_live_fragment():
            _render_live(st, session)

        render_live_fragment()
    else:  # pragma: no cover - compatibility with older Streamlit versions
        _render_live(st, session)


if __name__ == "__main__":
    main()
