import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/http/api", () => ({ $api: { useQuery: vi.fn() } }));

import { $api } from "@/lib/http/api";

import { autoRouterDateOptions, useAutoRouterBenchmarks } from "./useAutoRouterBenchmarks";

describe("useAutoRouterBenchmarks", () => {
  const range = { from: new Date(2026, 7, 15), to: new Date(2026, 7, 21, 23, 59, 59, 999) };

  afterEach(() => vi.useRealTimers());

  it("keeps custom UTC dates without extending the end to the current UTC day", () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-08-22T03:00:00Z"));

    useAutoRouterBenchmarks("sk-test", range, "key-hash-1");

    expect($api.useQuery).toHaveBeenLastCalledWith(
      "get",
      "/auto_router/benchmarks",
      { params: { query: { start_date: "2026-08-15", end_date: "2026-08-21", api_key: "key-hash-1" } } },
      { enabled: true, retry: false },
    );
  });

  it.each([
    ["last_7_days", "2026-08-16"],
    ["last_30_days", "2026-07-24"],
  ])("uses exactly %s through the current UTC date", (settingValue, startDate) => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-08-22T16:15:00Z"));
    const option = autoRouterDateOptions.find((item) => item.settingValue === settingValue);

    useAutoRouterBenchmarks("sk-test", option?.getValue() ?? {});

    expect($api.useQuery).toHaveBeenLastCalledWith(
      "get",
      "/auto_router/benchmarks",
      { params: { query: { start_date: startDate, end_date: "2026-08-22" } } },
      { enabled: true, retry: false },
    );
  });
});
