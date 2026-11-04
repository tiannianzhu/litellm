import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import TopAgents from "./TopAgents";
import type { TagSummaryRow } from "./agentCatalog";

const desktopTag = "User-Agent: claude-cli/2.1.263 (external, claude-desktop-3p)";
const row = (tag: string, tokens: number, spend = tokens / 100, requests = 1): TagSummaryRow => ({
  tag,
  total_spend: spend,
  total_tokens: tokens,
  total_requests: requests,
  successful_requests: requests,
  failed_requests: 0,
  unique_users: 1,
});
const rows = [row(desktopTag, 400), row("User-Agent: claude-code/2.1.263", 200)];

it("uses the selected metric for both charts and ranks and includes clients outside the top ten in shares", async () => {
  const user = userEvent.setup();
  const clients = [
    row(desktopTag, 400, 80, 4),
    row("User-Agent: Python", 800, 20, 8),
    row("User-Agent: curl", 200, 40, 20),
    ...Array.from({ length: 8 }, (_, index) => row(`User-Agent: tool-${index}`, 100, 10)),
  ];
  const metrics = (spend: number, tokens: number, requests: number): SpendMetrics => ({
    spend,
    prompt_tokens: tokens,
    completion_tokens: 0,
    total_tokens: tokens,
    api_requests: requests,
    successful_requests: requests,
    failed_requests: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  });
  const day: DailyData = {
    date: "2026-10-28",
    metrics: metrics(220, 2200, 40),
    breakdown: {
      models: {},
      model_groups: {},
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: Object.fromEntries(
        clients.map((client) => [
          client.tag,
          {
            metrics: metrics(client.total_spend, client.total_tokens, client.total_requests),
            metadata: {},
            api_key_breakdown: {},
          },
        ]),
      ),
    },
  };
  render(<TopAgents rows={clients} daily={[day]} loading={false} totalTokens={2200} onOpenAgent={vi.fn()} />);
  const ranks = () => screen.getAllByRole("button", { name: /^View .+ usage$/ });
  const chart = within(screen.getByRole("application"));
  const expectLeader = (name: string, value: string, share: string) => {
    const [first] = ranks();
    expect(first).toHaveAccessibleName(`View ${name} usage`);
    expect(first).toHaveTextContent(`${value} · ${share}`);
  };

  expect(ranks()).toHaveLength(10);
  expectLeader("Python", "800 tokens", "36.4%");
  expect(chart.getAllByText(/^[0-9.]+K$/).length).toBeGreaterThan(0);

  await user.click(screen.getByRole("radio", { name: "Spend" }));
  expectLeader("Claude Desktop", "$80.00", "36.4%");
  expect(chart.getAllByText(/^\$[1-9]\d{2}$/).length).toBeGreaterThan(0);
  expect(chart.queryAllByText(/^\$[0-9.]+K$/)).toHaveLength(0);

  await user.click(screen.getByRole("radio", { name: "Requests" }));
  expectLeader("curl", "20 requests", "50.0%");
  expect(chart.getByText("40", { exact: true })).toBeInTheDocument();
});

describe("TopAgents row navigation", () => {
  it("should render recognizable client logos and open usage when the row totals are clicked", async () => {
    const user = userEvent.setup();
    const onOpenAgent = vi.fn();
    render(<TopAgents rows={rows} daily={[]} loading={false} totalTokens={600} onOpenAgent={onOpenAgent} />);

    expect(screen.getByRole("heading", { name: "Top agents" })).toBeInTheDocument();
    const desktop = screen.getByRole("button", { name: "View Claude Desktop usage" });
    expect(within(desktop).getByRole("img", { name: "Claude Desktop logo" })).toHaveAttribute(
      "src",
      expect.any(String),
    );
    expect(screen.getByRole("img", { name: "Claude Code logo" })).toBeInTheDocument();

    await user.click(within(desktop).getByText("400 tokens"));
    expect(onOpenAgent).toHaveBeenCalledOnce();
    expect(onOpenAgent).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-desktop", tags: [desktopTag] }));
  });

  it("should open a focused agent row with the keyboard", async () => {
    const user = userEvent.setup();
    const onOpenAgent = vi.fn();
    render(<TopAgents rows={rows} daily={[]} loading={false} totalTokens={600} onOpenAgent={onOpenAgent} />);
    act(() => screen.getByRole("button", { name: "View Claude Desktop usage" }).focus());

    await user.keyboard("{Enter}");
    expect(onOpenAgent).toHaveBeenCalledWith(expect.objectContaining({ id: "claude-desktop" }));
  });

  it("should leave an unfilterable residual descriptive instead of making it an inaccurate drill", () => {
    const rollup = { ...row("User-Agent: claude-cli", 1000), total_requests: 2 };
    render(<TopAgents rows={[rollup, rows[0]]} daily={[]} loading={false} totalTokens={1000} onOpenAgent={vi.fn()} />);

    expect(screen.getByText("Claude (unspecified)")).toBeInTheDocument();
    expect(screen.getByText("Client not identified")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View Claude (unspecified) usage" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View Claude Desktop usage" })).toBeInTheDocument();
  });
});
