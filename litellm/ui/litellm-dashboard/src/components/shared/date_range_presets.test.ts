import { afterEach, describe, expect, it, vi } from "vitest";

import { beijingDayBounds, presetRange, relativeTimeOptions } from "./date_range_presets";

describe("report date presets", () => {
  afterEach(() => vi.useRealTimers());

  it.each(["America/Los_Angeles", "Pacific/Auckland"])(
    "uses Beijing today for inclusive calendar ranges in %s",
    (timezone) => {
      const originalTimezone = process.env.TZ;
      process.env.TZ = timezone;
      try {
        vi.useFakeTimers({ toFake: ["Date"] });
        vi.setSystemTime(new Date("2026-12-31T16:00:00Z"));
        const end = new Date(2027, 0, 1, 23, 59, 59, 999);
        expect(relativeTimeOptions.find((option) => option.shortLabel === "today")?.getValue()).toEqual({
          from: new Date(2027, 0, 1),
          to: end,
        });
        expect(relativeTimeOptions.find((option) => option.shortLabel === "7d")?.getValue()).toEqual({
          from: new Date(2026, 11, 26),
          to: end,
        });
        expect(relativeTimeOptions.find((option) => option.shortLabel === "30d")?.getValue()).toEqual({
          from: new Date(2026, 11, 3),
          to: end,
        });
        expect(relativeTimeOptions.find((option) => option.shortLabel === "7d")?.getValue("2026-12-31")).toEqual({
          from: new Date(2026, 11, 25),
          to: new Date(2026, 11, 31, 23, 59, 59, 999),
        });
        expect(presetRange("last_30_days", "2026-12-31")).toEqual({
          from: new Date(2026, 11, 2),
          to: new Date(2026, 11, 31, 23, 59, 59, 999),
        });
        expect(beijingDayBounds(new Date(2026, 9, 22), new Date(2026, 9, 22))).toEqual({
          from: new Date("2026-10-21T16:00:00.000Z"),
          to: new Date("2026-10-22T15:59:59.999Z"),
        });
      } finally {
        if (originalTimezone === undefined) delete process.env.TZ;
        else process.env.TZ = originalTimezone;
      }
    },
  );
});
