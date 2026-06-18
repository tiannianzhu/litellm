/* @vitest-environment jsdom */
import type { PaginationState } from "@tanstack/react-table";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import HealthCheckComponent from "./HealthCheckComponent";

const mockIndividualModelHealthCheckCall = vi.fn();
const mockLatestHealthChecksCall = vi.fn();
const mockVllmQueueMetricsCall = vi.fn();

vi.mock("../networking", () => ({
  individualModelHealthCheckCall: (...args: unknown[]) => mockIndividualModelHealthCheckCall(...args),
  latestHealthChecksCall: (...args: unknown[]) => mockLatestHealthChecksCall(...args),
  vllmQueueMetricsCall: (...args: unknown[]) => mockVllmQueueMetricsCall(...args),
}));

const getDisplayModelName = (model: { model_name?: string }) => model.model_name ?? "";

const makeModel = (id: string, name = "gpt-4") => ({
  model_name: name,
  model_info: { id },
  litellm_model_name: name,
});

interface HarnessProps {
  modelData: { data: ReturnType<typeof makeModel>[] };
  allModelsOnProxy: string[];
  accessToken?: string | null;
  rowCount?: number;
  onPageIndexChange?: (pageIndex: number) => void;
}

/** Holds pagination state so page changes exercise the real controlled wiring. */
function Harness({
  modelData,
  allModelsOnProxy,
  accessToken = "token",
  rowCount = 1,
  onPageIndexChange,
}: HarnessProps) {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });

  return (
    <>
      <span data-testid="page-index">{pagination.pageIndex}</span>
      <HealthCheckComponent
        accessToken={accessToken}
        modelData={modelData}
        all_models_on_proxy={allModelsOnProxy}
        getDisplayModelName={getDisplayModelName}
        pagination={pagination}
        onPaginationChange={(updater) => {
          setPagination((previous) => {
            const next = typeof updater === "function" ? updater(previous) : updater;
            onPageIndexChange?.(next.pageIndex);
            return next;
          });
        }}
        rowCount={rowCount}
      />
    </>
  );
}

const renderHealthCheck = async (props: HarnessProps) => {
  await act(async () => {
    render(<Harness {...props} />);
  });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
};

const numericDateTime: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "numeric",
  second: "numeric",
};
const germanDateTime = new Intl.DateTimeFormat("de-DE", numericDateTime);

const germanLocaleString = function (this: Date): string {
  return germanDateTime.format(this);
};

const rowIds = (): string[] =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => row.querySelector('[data-testid^="datatable-select-row-"]'))
    .filter((node): node is Element => node !== null)
    .map((node) => (node.getAttribute("data-testid") ?? "").replace("datatable-select-row-", ""));

describe("HealthCheckComponent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLatestHealthChecksCall.mockResolvedValue({ latest_health_checks: {} });
    mockVllmQueueMetricsCall.mockResolvedValue({ queues: {} });
    mockIndividualModelHealthCheckCall.mockResolvedValue({
      healthy_count: 1,
      unhealthy_count: 0,
      healthy_endpoints: [],
      unhealthy_endpoints: [],
    });
  });

  it("should render the health check section", async () => {
    await renderHealthCheck({ modelData: { data: [makeModel("deployment-1")] }, allModelsOnProxy: ["deployment-1"] });

    expect(screen.getByText("Model Health Status")).toBeInTheDocument();
    expect(
      screen.getByText("Run health checks on individual models to verify they are working correctly"),
    ).toBeInTheDocument();
  });

  it("should display vLLM running and waiting counts in one queue column", async () => {
    mockVllmQueueMetricsCall.mockResolvedValue({
      queues: {
        "deployment-1": {
          running: 12,
          waiting: 3,
        },
      },
    });
    const modelData = {
      data: [
        {
          model_name: "test-model",
          model_info: { id: "deployment-1" },
          litellm_model_name: "hosted_vllm/test-model",
        },
      ],
    };

    await renderHealthCheck({ modelData, allModelsOnProxy: ["deployment-1"] });

    await waitFor(() => {
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("12 / 3");
    });
    expect(mockVllmQueueMetricsCall).toHaveBeenCalledWith("token", expect.any(AbortSignal));
  });

  it("clears queue counts after a failed refresh and shows new counts after recovery", async () => {
    vi.useFakeTimers();
    try {
      mockVllmQueueMetricsCall
        .mockResolvedValueOnce({ queues: { "deployment-1": { running: 12, waiting: 3 } } })
        .mockRejectedValueOnce(new Error("Queue metrics unavailable"))
        .mockResolvedValueOnce({ queues: { "deployment-1": { running: 6, waiting: 1 } } });

      await act(async () => {
        render(
          <Harness
            modelData={{ data: [makeModel("deployment-1", "hosted_vllm/test-model")] }}
            allModelsOnProxy={["deployment-1"]}
          />,
        );
      });
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent(/^12 \/ 3$/);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent(/^- \/ -$/);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent(/^6 \/ 1$/);
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears timed-out queue counts before starting the next poll", async () => {
    vi.useFakeTimers();
    try {
      mockVllmQueueMetricsCall
        .mockResolvedValueOnce({ queues: { "deployment-1": { running: 12, waiting: 3 } } })
        .mockImplementationOnce(
          (_token: string, signal: AbortSignal) =>
            new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true })),
        )
        .mockResolvedValueOnce({ queues: { "deployment-1": { running: 0, waiting: 0 } } });

      await act(async () => {
        render(<Harness modelData={{ data: [makeModel("deployment-1")] }} allModelsOnProxy={["deployment-1"]} />);
      });
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("12 / 3");

      await act(async () => vi.advanceTimersByTimeAsync(10_000));
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("- / -");
      expect(mockVllmQueueMetricsCall).toHaveBeenCalledTimes(2);

      await act(async () => vi.advanceTimersByTimeAsync(5_000));
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("0 / 0");
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears queue counts on token change and ignores the previous token's late response", async () => {
    vi.useFakeTimers();
    try {
      const first = { queues: { "deployment-1": { running: 12, waiting: 3 } } };
      const previous = Promise.withResolvers<typeof first>();
      const replacement = Promise.withResolvers<typeof first>();
      mockVllmQueueMetricsCall
        .mockResolvedValueOnce(first)
        .mockReturnValueOnce(previous.promise)
        .mockReturnValueOnce(replacement.promise);
      const props = { modelData: { data: [makeModel("deployment-1")] }, allModelsOnProxy: ["deployment-1"] };
      const { rerender } = render(<Harness {...props} />);
      await act(async () => vi.advanceTimersByTimeAsync(5_000));
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("12 / 3");

      rerender(<Harness {...props} accessToken="replacement" />);
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("- / -");
      await act(async () => previous.resolve(first));
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("- / -");

      await act(async () => replacement.resolve({ queues: { "deployment-1": { running: 6, waiting: 1 } } }));
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("6 / 1");
      rerender(<Harness {...props} accessToken={null} />);
      expect(screen.getByTestId("queue-deployment-1")).toHaveTextContent("- / -");
    } finally {
      vi.useRealTimers();
    }
  });

  it("should call individualModelHealthCheckCall with model id when run health check is triggered", async () => {
    render(
      <Harness modelData={{ data: [makeModel("deployment-abc-123")] }} allModelsOnProxy={["deployment-abc-123"]} />,
    );

    const runButtons = screen.getAllByTestId("run-health-check-btn");
    expect(runButtons.length).toBeGreaterThanOrEqual(1);

    await act(async () => {
      runButtons[0].click();
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "deployment-abc-123");
    expect(mockIndividualModelHealthCheckCall).not.toHaveBeenCalledWith("token", "gpt-4");
  });

  it("should page through results with the shared pagination footer", async () => {
    const onPageIndexChange = vi.fn();
    await renderHealthCheck({
      modelData: { data: [makeModel("deployment-1")] },
      allModelsOnProxy: ["deployment-1"],
      rowCount: 75,
      onPageIndexChange,
    });

    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 75");

    const user = userEvent.setup();
    await user.click(screen.getByTestId("pagination-next"));

    expect(onPageIndexChange).toHaveBeenCalledWith(1);
    expect(screen.getByTestId("page-index")).toHaveTextContent("1");
  });

  describe("row selection drives the bulk run", () => {
    const twoModels = { data: [makeModel("id-alpha", "alpha"), makeModel("id-beta", "beta")] };
    const bothIds = ["id-alpha", "id-beta"];

    it("runs only the selected models and labels the button accordingly", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");

      await user.click(screen.getByTestId("datatable-select-row-id-beta"));
      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run Selected Checks");

      await act(async () => {
        screen.getByTestId("run-health-checks").click();
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-beta");
      expect(mockIndividualModelHealthCheckCall).not.toHaveBeenCalledWith("token", "id-alpha");
    });

    it("falls back to every model on the page when nothing is selected", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });

      await act(async () => {
        screen.getByTestId("run-health-checks").click();
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-alpha");
      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-beta");
    });

    it("treats a full page selection as running everything", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      await user.click(screen.getByTestId("datatable-select-all"));

      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");
    });

    it("clears the selection from the Clear Selection button", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      expect(screen.queryByTestId("clear-health-selection")).not.toBeInTheDocument();

      await user.click(screen.getByTestId("datatable-select-row-id-alpha"));
      await user.click(screen.getByTestId("clear-health-selection"));

      expect(screen.getByTestId("datatable-select-row-id-alpha")).not.toBeChecked();
      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");
    });

    // The pager swaps the underlying rows, so a carried-over selection would target
    // models that are no longer on screen.
    it("wipes the selection when the page changes", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 120 });
      const user = userEvent.setup();

      await user.click(screen.getByTestId("datatable-select-row-id-alpha"));
      expect(screen.getByTestId("clear-health-selection")).toBeInTheDocument();

      await user.click(screen.getByTestId("pagination-next"));

      expect(screen.queryByTestId("clear-health-selection")).not.toBeInTheDocument();
      expect(screen.getByTestId("datatable-select-row-id-alpha")).not.toBeChecked();
    });
  });

  describe("latest_health_checks keyed by model id", () => {
    it("should show status from latest_health_checks when keys match model ids", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "id-alpha": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
          "id-beta": { status: "unhealthy", checked_at: "2024-01-15T10:05:00Z", error_message: "Connection failed" },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("id-alpha"), makeModel("id-beta")] },
        allModelsOnProxy: ["id-alpha", "id-beta"],
        rowCount: 2,
      });

      expect(mockLatestHealthChecksCall).toHaveBeenCalledWith("token");
      expect(screen.getAllByText("healthy").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("unhealthy").length).toBeGreaterThanOrEqual(1);
    });

    it("should skip latest_health_checks entries whose key is not a known model id", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "current-model-id": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
          "deleted-or-unknown-id": { status: "unhealthy", checked_at: "2024-01-15T10:05:00Z", error_message: "Stale" },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("current-model-id")] },
        allModelsOnProxy: ["current-model-id"],
      });

      expect(screen.getByText("healthy")).toBeInTheDocument();
      expect(screen.queryByText("unhealthy")).not.toBeInTheDocument();
    });

    it("should not apply status when latest_health_checks key is model name not model id", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "gpt-4": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("model-id-123")] },
        allModelsOnProxy: ["model-id-123"],
      });

      expect(screen.queryByText("healthy")).not.toBeInTheDocument();
      expect(screen.getByText("none")).toBeInTheDocument();
    });

    it("does not count a failed stored check as the last success", async () => {
      const checkedAt = "2024-01-15T10:05:00Z";
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "id-alpha": { status: "unhealthy", checked_at: checkedAt, error_message: "Connection failed" },
        },
      });

      await renderHealthCheck({ modelData: { data: [makeModel("id-alpha")] }, allModelsOnProxy: ["id-alpha"] });

      expect(screen.getAllByText(new Date(checkedAt).toLocaleString())).toHaveLength(1);
    });
  });

  describe("check times on a German-locale browser", () => {
    const twoModels = { data: [makeModel("id-alpha", "alpha"), makeModel("id-beta", "beta")] };
    const bothIds = ["id-alpha", "id-beta"];
    const isoLikeLocaleString = vi.mocked(Date.prototype.toLocaleString).getMockImplementation();

    beforeEach(() => {
      vi.mocked(Date.prototype.toLocaleString).mockImplementation(germanLocaleString);
    });

    afterEach(() => {
      vi.useRealTimers();
      if (isoLikeLocaleString) {
        vi.mocked(Date.prototype.toLocaleString).mockImplementation(isoLikeLocaleString);
      }
    });

    it("sorts stored checks most-recent-first and shows them in the browser's format", async () => {
      const march = "2024-03-02T10:00:00Z";
      const june = "2024-06-01T10:00:00Z";
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "id-alpha": { status: "healthy", checked_at: march, error_message: null },
          "id-beta": { status: "healthy", checked_at: june, error_message: null },
        },
      });
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      expect(await screen.findAllByText(germanLocaleString.call(new Date(june)))).toHaveLength(2);

      await user.click(screen.getByTestId("sort-header-last_check"));

      expect(rowIds()).toEqual(["id-beta", "id-alpha"]);
    });

    const runBetaAlone = {
      "a single-model check": async (user: ReturnType<typeof userEvent.setup>) => {
        await user.click(screen.getAllByTestId("run-health-check-btn")[1]);
      },
      "a bulk run of the selected model": async (user: ReturnType<typeof userEvent.setup>) => {
        await user.click(screen.getByTestId("datatable-select-row-id-beta"));
        await user.click(screen.getByTestId("run-health-checks"));
      },
    };

    describe.each(Object.entries(runBetaAlone))("after %s", (_trigger, runCheck) => {
      it.each([
        ["succeeds", () => mockIndividualModelHealthCheckCall.mockResolvedValue({ unhealthy_count: 0 })],
        ["fails", () => mockIndividualModelHealthCheckCall.mockRejectedValue(new Error("upstream down"))],
      ])("that %s, sorts the fresh check above an older stored one", async (_outcome, arrangeCheck) => {
        const now = new Date("2026-01-20T12:00:00Z");
        vi.useFakeTimers({ toFake: ["Date"] });
        vi.setSystemTime(now);
        arrangeCheck();
        mockLatestHealthChecksCall.mockResolvedValue({
          latest_health_checks: {
            "id-alpha": { status: "healthy", checked_at: "2024-06-01T10:00:00Z", error_message: null },
          },
        });
        await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
        const user = userEvent.setup();

        await runCheck(user);
        expect((await screen.findAllByText(germanLocaleString.call(now))).length).toBeGreaterThan(0);

        await user.click(screen.getByTestId("sort-header-last_check"));

        expect(rowIds()).toEqual(["id-beta", "id-alpha"]);
      });
    });
  });
});
