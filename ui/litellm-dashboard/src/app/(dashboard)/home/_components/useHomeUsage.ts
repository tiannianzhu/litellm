import { useMemo, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCurrentUser } from "@/app/(dashboard)/hooks/users/useCurrentUser";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData } from "@/components/UsagePage/types";
import { all_admin_roles } from "@/utils/roles";
import { useAggregatedDailyActivity } from "@/app/(dashboard)/usage/_components/hooks/useAggregatedDailyActivity";
import { ENTITY_API } from "@/app/(dashboard)/usage/_components/components/EntityUsage/entityFetchFns";
import { EMPTY_DAILY_ACTIVITY_METADATA, type DailyActivityRequest } from "@/components/UsagePage/dailyActivityApi";
import {
  overviewTotals,
  type OverviewTotals,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";

export const HOME_USAGE_DAYS = 7;

export interface HomeUsage {
  results: readonly DailyData[];
  totals: OverviewTotals;
  loading: boolean;
  requestCountsPending: boolean;
  failed: boolean;
  budget: number | null;
  retry: () => void;
}

/** Last HOME_USAGE_DAYS of the signed-in user's view of usage: deployment-wide for admins, their own otherwise. */
export function useHomeUsage(): HomeUsage {
  const { accessToken, userRole, userId } = useAuthorized();
  const { data: currentUser } = useCurrentUser();
  const isAdmin = all_admin_roles.includes(userRole || "");
  const [attempt, setAttempt] = useState(0);
  const range = useMemo(() => {
    const endTime = new Date();
    const startTime = new Date(endTime.getTime() - HOME_USAGE_DAYS * 24 * 60 * 60 * 1000);
    return { startTime, endTime };
  }, []);

  const request = useMemo<DailyActivityRequest | null>(
    () =>
      accessToken && (isAdmin || userId)
        ? { accessToken, ...range, entityIds: isAdmin ? null : [userId as string] }
        : null,
    [accessToken, isAdmin, range, userId],
  );
  const { data, loading, failed } = useAggregatedDailyActivity({
    fetch: () => ENTITY_API.user.aggregated(request as DailyActivityRequest),
    enabled: request !== null,
    deps: [accessToken, isAdmin, userId, range.startTime.toISOString(), attempt],
  });

  const results = useMemo(() => toDailyData(data), [data]);
  const totals = useMemo(() => overviewTotals(data.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA), [data.metadata]);

  return {
    results,
    totals,
    loading,
    requestCountsPending: loading,
    failed,
    budget: currentUser?.max_budget ?? null,
    retry: () => setAttempt((n) => n + 1),
  };
}
