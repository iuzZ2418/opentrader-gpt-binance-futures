from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class TaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex}")
    task_type: str = Field(min_length=1, max_length=100)
    correlation_id: str = Field(min_length=1, max_length=100)
    payload: dict[str, JsonValue]
    attempt: int = Field(default=0, ge=0, le=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RedisStreamQueue:
    """Small Redis Streams adapter with consumer groups and a dead-letter stream."""

    def __init__(
        self,
        redis_url: str,
        *,
        stream: str,
        group: str,
        consumer: str,
        max_attempts: int = 3,
        reclaim_idle_ms: int = 60_000,
        client: Any | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if reclaim_idle_ms < 1:
            raise ValueError("reclaim_idle_ms must be positive")
        if client is None:
            try:
                import redis
            except ImportError as error:  # pragma: no cover - optional production dependency
                raise RuntimeError("Redis queue requires the optional redis package") from error
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dead_letter_stream = f"{stream}:dead"
        self.max_attempts = max_attempts
        self.reclaim_idle_ms = reclaim_idle_ms
        self._claim_cursor = "0-0"
        self.ensure_group()

    def ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except Exception as error:
            if "BUSYGROUP" not in str(error):
                raise

    def publish(self, task: TaskEnvelope) -> str:
        return str(
            self.client.xadd(
                self.stream,
                {"task": task.model_dump_json()},
                maxlen=100_000,
                approximate=True,
            )
        )

    def read(
        self, *, count: int = 10, block_ms: int = 5_000
    ) -> tuple[tuple[str, TaskEnvelope], ...]:
        if count < 1:
            raise ValueError("count must be positive")
        # A consumer can die after delivery but before ACK.  Reading only with `>` would leave
        # that message in the group's pending-entry list forever.  Reclaim sufficiently idle
        # work first; handlers remain responsible for idempotency because the original consumer
        # may have completed its side effect immediately before dying.
        claimed = self._reclaim_stale(count=count)
        if claimed:
            return claimed
        response = self.client.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        messages = [message for _, stream_messages in response or [] for message in stream_messages]
        return self._decode_messages(messages)

    def _reclaim_stale(self, *, count: int) -> tuple[tuple[str, TaskEnvelope], ...]:
        response = self.client.xautoclaim(
            self.stream,
            self.group,
            self.consumer,
            self.reclaim_idle_ms,
            self._claim_cursor,
            count=count,
        )
        if not response:
            self._claim_cursor = "0-0"
            return ()
        cursor = response[0]
        if isinstance(cursor, bytes):
            cursor = cursor.decode("utf-8")
        self._claim_cursor = str(cursor or "0-0")
        messages = response[1] if len(response) > 1 else ()
        return self._decode_messages(messages)

    @staticmethod
    def _decode_messages(
        messages: Sequence[tuple[Any, Mapping[Any, Any]]],
    ) -> tuple[tuple[str, TaskEnvelope], ...]:
        tasks: list[tuple[str, TaskEnvelope]] = []
        for message_id, fields in messages:
            raw = fields.get("task") or fields.get(b"task")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(message_id, bytes):
                message_id = message_id.decode("utf-8")
            tasks.append((str(message_id), TaskEnvelope.model_validate_json(raw)))
        return tuple(tasks)

    def ack(self, message_id: str) -> None:
        self.client.xack(self.stream, self.group, message_id)

    def fail(self, message_id: str, task: TaskEnvelope, error: Exception) -> None:
        next_attempt = task.attempt + 1
        if next_attempt >= self.max_attempts:
            self.client.xadd(
                self.dead_letter_stream,
                {
                    "task": task.model_dump_json(),
                    "error": str(error)[:2_000],
                    "failed_at": datetime.now(UTC).isoformat(),
                },
            )
        else:
            retry = task.model_copy(update={"attempt": next_attempt})
            self.publish(retry)
        self.ack(message_id)


class InMemoryTaskQueue:
    """Deterministic test/development queue with the same worker-facing methods."""

    def __init__(self, *, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts
        self.pending: list[tuple[str, TaskEnvelope]] = []
        self.dead_letters: list[tuple[TaskEnvelope, str]] = []
        self.acked: list[str] = []

    def publish(self, task: TaskEnvelope) -> str:
        message_id = f"memory-{uuid4().hex}"
        self.pending.append((message_id, task))
        return message_id

    def read(self, *, count: int = 10, block_ms: int = 0) -> tuple[tuple[str, TaskEnvelope], ...]:
        del block_ms
        result = tuple(self.pending[:count])
        del self.pending[:count]
        return result

    def ack(self, message_id: str) -> None:
        self.acked.append(message_id)

    def fail(self, message_id: str, task: TaskEnvelope, error: Exception) -> None:
        if task.attempt + 1 >= self.max_attempts:
            self.dead_letters.append((task, str(error)))
        else:
            self.publish(task.model_copy(update={"attempt": task.attempt + 1}))
        self.ack(message_id)


def process_tasks_once(
    queue: Any,
    handlers: Mapping[str, Callable[[TaskEnvelope], None]],
    *,
    count: int = 10,
    block_ms: int = 5_000,
) -> int:
    processed = 0
    for message_id, task in queue.read(count=count, block_ms=block_ms):
        try:
            handler = handlers.get(task.task_type)
            if handler is None:
                raise LookupError(f"No handler registered for {task.task_type}")
            handler(task)
        except Exception as error:
            queue.fail(message_id, task, error)
        else:
            queue.ack(message_id)
        processed += 1
    return processed


def run_task_worker(
    queue: Any,
    handlers: Mapping[str, Callable[[TaskEnvelope], None]],
    *,
    stop: Callable[[], bool],
    idle_sleep_seconds: float = 0.1,
) -> None:
    while not stop():
        processed = process_tasks_once(queue, handlers)
        if not processed:
            time.sleep(idle_sleep_seconds)


def publish_many(queue: Any, tasks: Sequence[TaskEnvelope]) -> tuple[str, ...]:
    return tuple(queue.publish(task) for task in tasks)


def task_payload(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Round-trip through JSON so secrets/non-serializable objects cannot enter Redis."""

    encoded = json.dumps(dict(value), ensure_ascii=False, separators=(",", ":"))
    return json.loads(encoded)
