# Phase 4 transport contract

Phase 4 keeps the edge classifier independent of how samples arrive:

```text
PC plant -> JSON serializer -> TCP -> Pi receiver -> MeasuredSample -> EdgePipeline
```

The local plant and replay modes remain available and use the same
`EdgePipeline`.

## Wire protocol

The stream is newline-delimited UTF-8 JSON. The first record is a `hello`
handshake. The Pi replies with `hello_ack`; only then may the PC send `sample`
records. The PC terminates a finite stream with an `end` record.

The handshake carries:

- `protocol_version`
- `schema_version`, `units_version`, `feature_version`, and `model_version`
- canonical `channel_order`: `["delta", "Vx", "r"]`
- `sample_period_s`

All compatibility fields must match. A mismatch closes the connection before
any sample reaches the detector.

Each sample contains `schema_version`, `timestamp_s`, `sequence`, `delta`,
`Vx`, and `r`. An optional `diagnostics` object is preserved for replay and
observability but is not part of `edge_payload()`. JSON non-finite values are
rejected.

## Quality and failure semantics

- TCP provides ordered delivery and back-pressure. The sender blocks when the
  receiver or operating-system buffer cannot accept more data.
- The receiver validates each sample before forwarding it. Duplicate and late
  samples are rejected by `EdgePipeline` and do not change detector state.
- A valid sample after a sequence or timestamp discontinuity is forwarded with
  a `gap` status; the edge does not interpolate missing samples.
- Invalid JSON, unsupported versions, malformed records, or a wrong handshake
  close the stream and leave detector state unchanged for that record.
- A socket timeout is reported as `stale` once it exceeds the configured
  `stale_timeout_s`. A disconnect is reported as `disconnected`. Reconnect mode
  accepts a new compatible handshake without silently replaying old samples.
- Transport status is separate from classifier verdict state. A connection
  failure does not fabricate a verdict or reset the detector automatically.

The in-process adapter uses a bounded queue and supports deterministic
drop/duplicate/reorder/delay faults. `FileSampleSink` writes the same framed
records for repeatable offline debugging.

## Operations

Pi receiver:

```bash
python -m pi.app --tcp-listen 0.0.0.0:8765 --realtime
```

Transport capture replay:

```bash
python -m pi.app --transport-file capture.ndjson
```

PC-side Python sender for a source is available through
`pi.transport.tcp.TcpSampleSender`; a Simulink sender can implement the same
handshake and record format.

## Verification

`pi.test_transport` verifies JSON round trips, compatibility rejection,
bounded loopback behavior, file replay, and a TCP sender/receiver round trip.
The TCP and file paths are expected to produce the same `MeasuredSample`
sequence as local/replay sources for the same deterministic input.
