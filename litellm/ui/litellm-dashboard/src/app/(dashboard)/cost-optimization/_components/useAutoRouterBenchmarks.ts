import { useState } from "react";

import { formatDate } from "@/components/networking";
import { presetRange, relativeTimeOptions } from "@/components/shared/date_range_presets";
import { $api } from "@/lib/http/api";

import type { ActivityDateRange, DateRange } from "./useDailyActivityRange";

const utcToday = () => new Date().toISOString().slice(0, 10);

export const autoRouterDateOptions = relativeTimeOptions.map((option) => ({
  ...option,
  getValue: () => option.getValue(utcToday()),
}));

export const useAutoRouterDateRange = (): ActivityDateRange => {
  const [dateValue, setDateValue] = useState<DateRange>(() => presetRange("last_30_days", utcToday()));
  return { dateValue, onDateChange: setDateValue };
};

export const useAutoRouterBenchmarks = (
  accessToken: string | null,
  range: DateRange,
  apiKey?: string,
  userId?: string,
) =>
  $api.useQuery(
    "get",
    "/auto_router/benchmarks",
    {
      params: {
        query: {
          start_date: range.from ? formatDate(range.from) : undefined,
          end_date: range.to ? formatDate(range.to) : undefined,
          api_key: apiKey,
          user_id: userId,
        },
      },
    },
    { enabled: Boolean(accessToken && range.from && range.to), retry: false },
  );
