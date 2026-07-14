from __future__ import annotations

import pytest

from crypto_event_trader.task_queue import (
    InMemoryTaskQueue,
    RedisStreamQueue,
    TaskEnvelope,
    process_tasks_once,
    task_payload,
)


def _task(task_type: str = "cycle") -> TaskEnvelope:
    return TaskEnvelope(
        task_type=task_type,
        correlation_id="trace-1",
        payload={"symbol": "BTCUSDT"},
    )


def test_worker_acknowledges_success() -> None:
    queue = InMemoryTaskQueue()
    seen: list[str] = []
    message_id = queue.publish(_task())

    assert process_tasks_once(queue, {"cycle": lambda task: seen.append(task.task_id)}) == 1
    assert seen and queue.acked == [message_id]


def test_worker_retries_then_dead_letters() -> None:
    queue = InMemoryTaskQueue(max_attempts=2)
    queue.publish(_task())

    def fail(_: TaskEnvelope) -> None:
        raise RuntimeError("boom")

    process_tasks_once(queue, {"cycle": fail})
    assert queue.pending[0][1].attempt == 1
    process_tasks_once(queue, {"cycle": fail})
    assert queue.dead_letters[0][1] == "boom"
    assert not queue.pending


def test_unknown_task_is_not_silently_dropped() -> None:
    queue = InMemoryTaskQueue(max_attempts=1)
    queue.publish(_task("unknown"))
    process_tasks_once(queue, {})
    assert "No handler" in queue.dead_letters[0][1]


def test_task_payload_rejects_non_json_secrets_or_objects() -> None:
    assert task_payload({"symbols": ["BTCUSDT"]}) == {"symbols": ["BTCUSDT"]}
    with pytest.raises(TypeError):
        task_payload({"credential_object": object()})


class _FakeRedis:
    def __init__(self, *, claimed: list[tuple[object, dict[object, object]]] | None = None) -> None:
        self.claimed = claimed or []
        self.claim_calls: list[tuple[object, ...]] = []
        self.read_calls = 0

    def xgroup_create(self, *_: object, **__: object) -> None:
        return None

    def xautoclaim(self, *args: object, **kwargs: object) -> list[object]:
        self.claim_calls.append((*args, kwargs))
        messages, self.claimed = self.claimed, []
        return [b"0-0", messages, []]

    def xreadgroup(self, *_: object, **__: object) -> list[object]:
        self.read_calls += 1
        return []


def test_redis_queue_reclaims_idle_pending_work_before_reading_new_messages() -> None:
    task = _task()
    redis = _FakeRedis(claimed=[(b"123-0", {b"task": task.model_dump_json().encode()})])
    queue = RedisStreamQueue(
        "redis://unused",
        stream="work",
        group="workers",
        consumer="replacement",
        reclaim_idle_ms=12_345,
        client=redis,
    )

    assert queue.read(count=7) == (("123-0", task),)
    assert redis.read_calls == 0
    assert redis.claim_calls[0][:5] == (
        "work",
        "workers",
        "replacement",
        12_345,
        "0-0",
    )


def test_redis_queue_reads_new_messages_after_pending_scan_is_empty() -> None:
    redis = _FakeRedis()
    queue = RedisStreamQueue(
        "redis://unused",
        stream="work",
        group="workers",
        consumer="consumer",
        client=redis,
    )

    assert queue.read() == ()
    assert redis.read_calls == 1


def test_redis_queue_rejects_disabled_pending_recovery() -> None:
    with pytest.raises(ValueError, match="reclaim_idle_ms"):
        RedisStreamQueue(
            "redis://unused",
            stream="work",
            group="workers",
            consumer="consumer",
            reclaim_idle_ms=0,
            client=_FakeRedis(),
        )
