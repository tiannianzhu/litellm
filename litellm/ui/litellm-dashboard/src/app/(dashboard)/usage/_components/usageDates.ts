import moment from "moment";
import { useEffect, useState } from "react";

import { beijingDay, beijingDayBounds, utcDay } from "@/components/shared/date_range_presets";

export function useUsageDay(timeZone: "Asia/Shanghai" | "UTC" = "Asia/Shanghai"): string {
  const [day, setDay] = useState(() => (timeZone === "UTC" ? utcDay(new Date()) : beijingDay(new Date())));
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const refresh = () => {
      clearTimeout(timer);
      const now = new Date();
      const today = timeZone === "UTC" ? utcDay(now) : beijingDay(now);
      setDay(today);
      const tomorrow = moment(today, "YYYY-MM-DD", true).add(1, "day").toDate();
      const nextMidnight =
        timeZone === "UTC"
          ? Date.parse(`${today}T00:00:00Z`) + 86_400_000
          : beijingDayBounds(tomorrow, tomorrow).from.getTime();
      timer = setTimeout(refresh, nextMidnight - now.getTime());
    };
    refresh();
    window.addEventListener("focus", refresh);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", refresh);
    };
  }, [timeZone]);
  return day;
}
