"use client";

import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  useActivityDateRange,
  type ActivityDateRange,
} from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";
import { getTopModels } from "@/app/(dashboard)/usage/_components/components/EntityUsage/entityUsageAggregations";
import { ActivityMetrics, ModelSection, processActivityData } from "@/components/activity_metrics";
import { dailyActivityAggregatedCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { ChartLoader } from "@/components/shared/chart_loader";
import { ErrorState } from "@/components/shared/ErrorState";
import { hasProxyWideSpendView, spendScopeUserId } from "@/utils/roles";
import { EMPTY_DAILY_ACTIVITY_METADATA, toDailyData, type DailyActivityRequest } from "./dailyActivityApi";
import { keyDetailFromResponse, overallUsageMetrics, reportingKeyId } from "./keyActivityData";
import { matchesWildcard } from "@/lib/wildcard_routes";

interface ResourceActivityProps {
  scope: "key" | "team" | "user" | "model";
  id: string;
  activity?: ActivityDateRange;
}

const getActivityScope = (
  scope: ResourceActivityProps["scope"],
  id: string,
  userId: string | null,
  userRole: string,
) => {
  const isTeam = scope === "team";
  const isUser = scope === "user";
  const proxyWideView = hasProxyWideSpendView(userRole);
  return {
    entity: isTeam ? "team" : "user",
    entityId: isTeam || isUser ? id : spendScopeUserId(userRole, userId),
    apiKey: scope === "key" ? reportingKeyId(id) : undefined,
    canView: !isUser || proxyWideView || id === userId,
    scopedToViewer: !isUser && !proxyWideView,
  } as const;
};

export default function ResourceActivity({ scope, id, activity }: ResourceActivityProps) {
  const { t } = useTranslation();
  const { accessToken, userId, userRole } = useAuthorized();
  const localActivity = useActivityDateRange();
  const { dateValue, onDateChange } = activity ?? localActivity;
  const wildcard = scope === "model" && id.includes("*");
  const { entity, entityId, apiKey, canView, scopedToViewer } = getActivityScope(scope, id, userId, userRole);
  const hasAccess = accessToken && id && canView;
  const request: DailyActivityRequest | null =
    hasAccess && dateValue.from && dateValue.to
      ? {
          accessToken,
          startTime: dateValue.from,
          endTime: dateValue.to,
          entityIds: entityId ? [entityId] : null,
          apiKey,
          apiKeyLimit: 1,
          includeEntityBreakdown: scope === "model" ? false : undefined,
          model: scope === "model" && !wildcard ? id : undefined,
        }
      : null;
  const { data, loading, failed, errorMessage, reload } = useAggregatedDailyActivity({
    fetch: () => dailyActivityAggregatedCall(entity, request as DailyActivityRequest),
    enabled: request !== null,
    deps: [accessToken, scope, id, entityId, dateValue.from, dateValue.to],
  });
  const results = useMemo(() => toDailyData(data), [data]);
  const metrics = useMemo(
    () =>
      (apiKey ? keyDetailFromResponse(data, apiKey, []) : undefined) ?? {
        ...overallUsageMetrics(results, data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA),
        top_models:
          scope === "team" || scope === "user"
            ? getTopModels(results, "models", Infinity).map(({ key, ...model }) => ({ model: key, ...model }))
            : [],
      },
    [data, results, apiKey, scope],
  );
  const modelMetrics = useMemo(
    () =>
      wildcard
        ? Object.fromEntries(
            Object.entries(processActivityData({ results }, "models", [])).filter(([model]) =>
              matchesWildcard(id, model),
            ),
          )
        : {},
    [results, wildcard, id],
  );

  if (!request) return null;

  const content = (() => {
    if (loading) return <ChartLoader />;
    if (failed) return <ErrorState message={errorMessage} onRetry={reload} />;
    if (data.results.length === 0 || (wildcard && Object.keys(modelMetrics).length === 0)) {
      return <p className="py-8 text-center text-muted-foreground">{t("usage.keyActivity.noDailyDetails")}</p>;
    }
    return wildcard ? (
      <ActivityMetrics modelMetrics={modelMetrics} />
    ) : (
      <ModelSection modelName={id} metrics={metrics} />
    );
  })();

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-end gap-4">
        <AdvancedDatePicker
          value={dateValue}
          onValueChange={onDateChange}
          timeZoneLabel={t("shared.advancedDatePicker.beijingTimeZone")}
        />
      </div>
      {scopedToViewer && (
        <p className="text-sm text-muted-foreground">
          {t(scope === "team" ? "shared.resourceActivity.teamScope" : "shared.resourceActivity.ownUsage")}
        </p>
      )}
      {scope === "model" && (
        <p className="text-sm text-muted-foreground">{t("shared.resourceActivity.modelScope", { model: id })}</p>
      )}
      {content}
    </div>
  );
}
