CREATE TABLE "LiteLLM_SpendAccountingEvent" (
    "event_key" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "call_type" TEXT NOT NULL,
    "spend" DOUBLE PRECISION NOT NULL,
    "version" INTEGER NOT NULL,
    "payload_json" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "processed_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_SpendAccountingEvent_pkey" PRIMARY KEY ("event_key")
);

CREATE INDEX "LiteLLM_SpendAccountingEvent_request_status_idx"
    ON "LiteLLM_SpendAccountingEvent"("request_id", "status");

CREATE INDEX "LiteLLM_SpendAccountingEvent_pending_idx"
    ON "LiteLLM_SpendAccountingEvent"("processed_at", "created_at", "event_key");
