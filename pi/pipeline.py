"""Minimal source-independent Phase 0 edge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .contracts import (
        ExecutionConfig,
        MeasuredSample,
        SampleResult,
        SampleStatus,
        SampleValidationError,
        compare_sample,
    )
except ImportError:  # Allows direct execution from the pi directory.
    from contracts import (
        ExecutionConfig,
        MeasuredSample,
        SampleResult,
        SampleStatus,
        SampleValidationError,
        compare_sample,
    )


@dataclass(frozen=True)
class PipelineEvent:
    result: SampleResult
    consumer_result: Any = None


class NullConsumer:
    """Placeholder consumer used until the Phase 1 detector exists."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float]] = []

    def start(self) -> None:
        pass

    def push(self, sample: MeasuredSample) -> None:
        self.samples.append(sample.edge_payload())

    def flush(self) -> None:
        pass

    def reset(self) -> None:
        self.samples.clear()

    def close(self) -> None:
        pass


class EdgePipeline:
    """Validate samples and forward only the three required edge channels."""

    def __init__(self, consumer: Any | None = None, config: ExecutionConfig | None = None):
        self.consumer = consumer or NullConsumer()
        self.config = config or ExecutionConfig()
        self.previous: MeasuredSample | None = None
        self.started = False

    def start(self) -> None:
        self.consumer.start()
        self.started = True

    def push(self, sample: MeasuredSample) -> PipelineEvent:
        if not self.started:
            self.start()
        try:
            status = compare_sample(self.previous, sample, self.config.validation)
        except SampleValidationError as exc:
            return PipelineEvent(SampleResult(exc.status, None, str(exc)))
        if status in (SampleStatus.LATE, SampleStatus.DUPLICATE):
            return PipelineEvent(
                SampleResult(status, None, f"sample rejected: {status.value}")
            )
        self.previous = sample
        result = SampleResult(
            status,
            sample,
            "timestamp/sequence gap" if status is SampleStatus.GAP else "",
        )
        return PipelineEvent(result, self.consumer.push(sample))

    def flush(self) -> Any:
        return self.consumer.flush()

    def reset(self) -> None:
        self.previous = None
        self.consumer.reset()

    def close(self) -> None:
        self.consumer.close()
        self.started = False

    def run(self, source: Any) -> list[PipelineEvent]:
        events = [self.push(sample) for sample in source]
        flushed = self.flush()
        if flushed is not None:
            events.append(
                PipelineEvent(
                    SampleResult(SampleStatus.ACCEPTED, None, "end of stream"),
                    flushed,
                )
            )
        return events
