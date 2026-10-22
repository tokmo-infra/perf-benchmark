from perf_benchmark.core.profiles import Continuation
from perf_benchmark.core.models import Session, Turn
from perf_benchmark.pipeline.contracts.executor import AttemptInput, ExecutorTrial
from perf_benchmark.pipeline.preparation.payloads import BlobStore, PayloadRef


def test_attempt_input_is_an_immutable_snapshot_of_domain_state():
    first = Turn(
        id="first",
        ordinal=0,
        request_payload_ref=PayloadRef(0, 1),
        status="succeeded",
        metadata={"kind": "prefix"},
    )
    current = Turn(
        id="current",
        ordinal=1,
        request_payload_ref=PayloadRef(1, 1),
        context_mode="live",
        metadata={"dataset_output_policy": "fixed"},
    )
    session = Session(id="session", turns=[first, current], metadata={"cycle": 2})
    session.link()
    payloads = BlobStore()
    attempt = AttemptInput.capture(
        session,
        current,
        {"first": Continuation(b'{"role":"assistant","content":"answer"}')},
        attempt=2,
        payloads=payloads,
    )

    session.metadata["cycle"] = 9
    current.metadata["dataset_output_policy"] = "natural"

    assert attempt.session.metadata == {"cycle": 2}
    assert attempt.turn.metadata == {"dataset_output_policy": "fixed"}
    assert attempt.previous_turns[0].status == "succeeded"
    assert attempt.live_history["first"].payload.endswith(b'"answer"}')
    assert attempt.is_final_turn
    payloads.close()


def test_executor_trial_freezes_request_parameters():
    values = {"temperature": 0.2, "thinking": {"enabled": False}}
    trial = ExecutorTrial("trial", values, {"source": "test"})
    values["thinking"]["enabled"] = True

    assert trial.request_parameters["thinking"] == {"enabled": False}
