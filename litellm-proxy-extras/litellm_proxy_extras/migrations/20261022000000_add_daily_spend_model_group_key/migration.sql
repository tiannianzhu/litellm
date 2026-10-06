BEGIN;

CREATE UNIQUE INDEX "LiteLLM_DailyUserSpend_model_group_key" ON "LiteLLM_DailyUserSpend"("user_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX "LiteLLM_DailyTeamSpend_model_group_key" ON "LiteLLM_DailyTeamSpend"("team_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX "LiteLLM_DailyOrganizationSpend_model_group_key" ON "LiteLLM_DailyOrganizationSpend"("organization_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX "LiteLLM_DailyEndUserSpend_model_group_key" ON "LiteLLM_DailyEndUserSpend"("end_user_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX "LiteLLM_DailyAgentSpend_model_group_key" ON "LiteLLM_DailyAgentSpend"("agent_id", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

CREATE UNIQUE INDEX "LiteLLM_DailyTagSpend_model_group_key" ON "LiteLLM_DailyTagSpend"("tag", "date", "api_key", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint");

DROP INDEX "LiteLLM_DailyUserSpend_user_id_date_api_key_model_custom_ll_key";
DROP INDEX "LiteLLM_DailyTeamSpend_team_id_date_api_key_model_custom_ll_key";
DROP INDEX "LiteLLM_DailyOrganizationSpend_organization_id_date_api_key_key";
DROP INDEX "LiteLLM_DailyEndUserSpend_end_user_id_date_api_key_model_cu_key";
DROP INDEX "LiteLLM_DailyAgentSpend_agent_id_date_api_key_model_custom__key";
DROP INDEX "LiteLLM_DailyTagSpend_tag_date_api_key_model_custom_llm_pro_key";

COMMIT;
