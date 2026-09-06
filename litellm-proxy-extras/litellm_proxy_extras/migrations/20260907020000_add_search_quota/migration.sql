CREATE TABLE IF NOT EXISTS "LiteLLM_SearchQuota" (
    "day" DATE NOT NULL,
    "identity" TEXT NOT NULL,
    "calls" INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT "LiteLLM_SearchQuota_pkey" PRIMARY KEY ("day", "identity")
);
