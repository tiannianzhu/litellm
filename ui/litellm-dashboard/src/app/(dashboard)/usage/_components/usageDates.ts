import moment from "moment";
import { useEffect, useState } from "react";

import { beijingDay, beijingDayBounds } from "@/components/shared/date_range_presets";

export function useUsageDay(): string {
  const [day, setDay] = useState(() => beijingDay(new Date()));
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const refresh = () => {
      clearTimeout(timer);
      const now = new Date();
      const today = beijingDay(now);
      setDay(today);
      const tomorrow = moment(today, "YYYY-MM-DD", true).add(1, "day").toDate();
      timer = setTimeout(refresh, beijingDayBounds(tomorrow, tomorrow).from.getTime() - now.getTime());
    };
    refresh();
    window.addEventListener("focus", refresh);
    return () => {
      clearTimeout(timer);
      window.removeEventListener("focus", refresh);
    };
  }, []);
  return day;
}
