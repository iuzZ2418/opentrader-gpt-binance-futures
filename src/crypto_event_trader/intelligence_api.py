from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status

from .config import Settings
from .ingestion.github import parse_github_webhook, verify_webhook_signature
from .intelligence_worker import IntelligenceWorker, build_intelligence_runtime

GitHubSignature = Annotated[str | None, Header(alias="X-Hub-Signature-256")]
GitHubEvent = Annotated[str | None, Header(alias="X-GitHub-Event")]
GitHubDelivery = Annotated[str | None, Header(alias="X-GitHub-Delivery")]

MAX_WEBHOOK_BYTES = 1_000_000
DELIVERY_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def create_intelligence_webhook_app(
    settings: Settings | None = None,
    worker: IntelligenceWorker | Any | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    owned_runtime = None
    if worker is None:
        owned_runtime = build_intelligence_runtime(
            runtime_settings,
            webhook_only=True,
        )
        worker = owned_runtime.worker

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(worker.startup_check)
            yield
        finally:
            if owned_runtime is not None:
                await asyncio.to_thread(worker.close)

    app = FastAPI(
        title="Crypto Intelligence Webhooks",
        description="Authenticated, allow-listed inert evidence ingress",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "github_webhook_ingress"}

    @app.post("/webhooks/github", status_code=status.HTTP_202_ACCEPTED)
    async def github_webhook(
        request: Request,
        x_hub_signature_256: GitHubSignature = None,
        x_github_event: GitHubEvent = None,
        x_github_delivery: GitHubDelivery = None,
    ) -> dict[str, Any]:
        secret = runtime_settings.github_webhook_secret
        if not secret or not runtime_settings.github_allowed_repositories:
            raise HTTPException(status_code=503, detail="GitHub webhook ingress is unavailable")
        if not x_github_delivery or not DELIVERY_ID.fullmatch(x_github_delivery):
            raise HTTPException(status_code=400, detail="invalid GitHub delivery identity")
        if not x_github_event:
            raise HTTPException(status_code=400, detail="missing GitHub event type")

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_WEBHOOK_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="GitHub webhook body is too large",
                    )
            except ValueError as error:
                raise HTTPException(status_code=400, detail="invalid content length") from error
        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_WEBHOOK_BYTES:
                raise HTTPException(status_code=413, detail="GitHub webhook body is too large")
            chunks.append(chunk)
        body = b"".join(chunks)
        if not verify_webhook_signature(secret, body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="invalid GitHub webhook signature")
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            batch = parse_github_webhook(
                x_github_event,
                payload,
                allowed_repositories=runtime_settings.github_allowed_repositories,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail="repository is not allow-listed") from error
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="unsupported or invalid webhook") from error

        outcomes = []
        for deletion in batch.deletions:
            outcomes.append(
                await asyncio.to_thread(
                    worker.ingest_deletion,
                    source=deletion.source,
                    source_id=deletion.source_id,
                )
            )
        for document in batch.documents:
            outcomes.append(await asyncio.to_thread(worker.ingest_document, document))
        return {
            "delivery_id": x_github_delivery,
            "repository": batch.repository,
            "accepted": len(outcomes),
            "evidence_record_ids": [item.evidence_record_id for item in outcomes],
        }

    return app
