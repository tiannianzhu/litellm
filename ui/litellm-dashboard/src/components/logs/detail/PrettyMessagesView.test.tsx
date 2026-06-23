import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { PrettyMessagesView } from "./PrettyMessagesView";

describe("PrettyMessagesView", () => {
  it("should render the component for standard chat completions", () => {
    const request = {
      messages: [{ role: "user", content: "Hello" }],
    };
    const response = {
      choices: [{ message: { role: "assistant", content: "Hi there!" } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("Hi there!")).toBeInTheDocument();
  });

  it("renders input when request is a bare messages array (cold storage payload)", () => {
    const request = [{ role: "user", content: "Write me a poem" }];
    const response = {
      choices: [{ message: { role: "assistant", content: "A quiet moment." } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Write me a poem")).toBeInTheDocument();
    expect(screen.getByText("A quiet moment.")).toBeInTheDocument();
  });

  it("renders Responses API input and output messages", () => {
    const request = {
      input: [
        {
          type: "message",
          role: "developer",
          content: [{ type: "input_text", text: "Follow the project instructions." }],
        },
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Inspect the repository." }],
        },
      ],
    };
    const response = {
      object: "response",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "The repository is ready." }],
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);

    expect(screen.getByText("Follow the project instructions.")).toBeInTheDocument();
    expect(screen.getByText("Inspect the repository.")).toBeInTheDocument();
    expect(screen.getByText("The repository is ready.")).toBeInTheDocument();
  });

  it("should render the realtime pretty view for realtime API responses", () => {
    const request = {};
    const response = {
      results: [
        {
          type: "session.created",
          session: {
            id: "sess_123",
            model: "gpt-4o-mini-realtime-preview",
            voice: "alloy",
            modalities: ["audio", "text"],
          },
        },
        {
          type: "response.done",
          response: {
            id: "resp_1",
            status: "completed",
            output: [
              {
                id: "item_1",
                role: "assistant",
                type: "message",
                content: [{ type: "audio", transcript: "Hello from realtime!" }],
              },
            ],
          },
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Session")).toBeInTheDocument();
    expect(screen.getByText("Hello from realtime!")).toBeInTheDocument();
    const modelElements = screen.getAllByText("gpt-4o-mini-realtime-preview");
    expect(modelElements.length).toBeGreaterThanOrEqual(1);
  });

  it("renders a Responses API log, whose body uses input/output instead of messages/choices", () => {
    const request = {
      model: "gpt-5.6",
      input: [{ role: "user", content: "Reply with exactly: hello from responses api" }],
    };
    const response = {
      output: [
        {
          id: "msg_070989277645d4ae",
          role: "assistant",
          type: "message",
          status: "completed",
          content: [{ text: "hello from responses api", type: "output_text", annotations: [] }],
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Reply with exactly: hello from responses api")).toBeInTheDocument();
    expect(screen.getByText("hello from responses api")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders a Responses API tool call, whose output item is a function_call", () => {
    const request = {
      model: "gpt-5.6",
      input: [{ role: "user", content: "What is the weather in San Francisco? Use the tool." }],
    };
    const response = {
      output: [
        {
          id: "fc_08edf6c2312f1485",
          name: "get_weather",
          type: "function_call",
          status: "completed",
          call_id: "call_AtO0J9eNy5jgECXzBicMJM8W",
          arguments: '{"city":"San Francisco"}',
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("What is the weather in San Francisco? Use the tool.")).toBeInTheDocument();
    expect(screen.getByText("get_weather")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders instructions as the system turn and a bare string input", () => {
    const request = { model: "gpt-5.6", instructions: "You are terse.", input: "Say A" };
    const response = {
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "A" }] }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("You are terse.")).toBeInTheDocument();
    expect(screen.getByText("Say A")).toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
  });

  it("skips reasoning output items rather than rendering them as empty turns", () => {
    const request = { input: [{ role: "user", content: "Think then answer" }] };
    const response = {
      output: [
        { type: "reasoning", id: "rs_1", summary: [] },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "answered" }] },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("answered")).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders a Responses API follow-up turn carrying a prior function_call and its output", () => {
    const request = {
      input: [
        { role: "user", content: "What is the weather in San Francisco? Use the tool." },
        {
          type: "function_call",
          name: "get_weather",
          call_id: "call_AtO0J9eNy5jgECXzBicMJM8W",
          arguments: '{"city":"San Francisco"}',
        },
        { type: "function_call_output", call_id: "call_AtO0J9eNy5jgECXzBicMJM8W", output: '{"temp":18}' },
      ],
    };
    const response = {
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "It is 18 degrees." }] }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("It is 18 degrees.")).toBeInTheDocument();
    expect(screen.getByText('{"temp":18}')).toBeInTheDocument();
    expect(screen.getByText("TOOL")).toBeInTheDocument();
  });

  it("maps the developer and legacy function roles onto the roles the drawer renders", () => {
    const request = {
      messages: [
        { role: "developer", content: "Stay terse." },
        { role: "user", content: "Weather?" },
        { role: "function", name: "get_weather", content: '{"temp":18}' },
      ],
    };
    const response = { choices: [{ message: { role: "assistant", content: "18 degrees." } }] };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Stay terse.")).toBeInTheDocument();
    expect(screen.getByText("TOOL")).toBeInTheDocument();
    expect(screen.queryByText("FUNCTION")).not.toBeInTheDocument();
  });

  it("still reports missing output when a Responses API log has an empty output array", () => {
    const request = { input: [{ role: "user", content: "Hello" }] };

    render(<PrettyMessagesView request={request} response={{ output: [] }} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("No response data available")).toBeInTheDocument();
  });

  it("should render standard view when response has results but no realtime events", () => {
    const request = {
      messages: [{ role: "user", content: "Test" }],
    };
    const response = {
      results: [{ type: "some.other.type" }],
      choices: [{ message: { role: "assistant", content: "Reply" } }],
    };

    render(<PrettyMessagesView request={request} response={response} />);
    expect(screen.getByText("Test")).toBeInTheDocument();
    expect(screen.getByText("Reply")).toBeInTheDocument();
  });

  it("renders search API results", () => {
    const request = [{ role: "user", content: "Search for LiteLLM" }];
    const response = {
      object: "search",
      results: [
        {
          title: "LiteLLM documentation",
          url: "https://docs.litellm.ai/",
          snippet: "Use LiteLLM to call multiple model providers.",
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);

    expect(screen.getByText("LiteLLM documentation", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("https://docs.litellm.ai/", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("Use LiteLLM to call multiple model providers.", { exact: false })).toBeInTheDocument();
    expect(screen.queryByText("No response data available")).not.toBeInTheDocument();
  });

  it("renders Anthropic content blocks by type", async () => {
    const user = userEvent.setup();
    const request = {
      system: [{ type: "text", text: "Follow the system policy." }],
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "toolu_123",
              name: "search_code",
              input: { query: "build_launch_command" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_123",
              content: "Found the implementation.",
            },
            { type: "text", text: "Continue the review." },
          ],
        },
      ],
    };
    const response = {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "Check the implementation details." },
        { type: "text", text: "The implementation is correct." },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);

    expect(screen.getByText("SYSTEM")).toBeInTheDocument();
    expect(screen.getByText("Follow the system policy.")).toBeInTheDocument();
    expect(screen.getByText("TOOL RESULT")).toBeInTheDocument();
    expect(screen.getByText("Found the implementation.")).toBeInTheDocument();
    expect(screen.getByText("Continue the review.")).toBeInTheDocument();
    expect(screen.getByText("USER")).toBeInTheDocument();
    expect(screen.getByText("THINKING")).toBeInTheDocument();
    expect(screen.getByText("The implementation is correct.")).toBeInTheDocument();
    expect(screen.queryByText(/"tool_use_id"/)).not.toBeInTheDocument();

    await user.click(screen.getByText("HISTORY (1 message)"));
    expect(screen.getByText("search_code")).toBeInTheDocument();
    expect(screen.getAllByText("toolu_123").length).toBeGreaterThanOrEqual(2);
  });

  it("renders OpenAI tool calls and tool results without text content", async () => {
    const user = userEvent.setup();
    const request = {
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_123",
              function: { name: "get_weather", arguments: '{"city":"Paris"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_123", content: "Sunny" },
        { role: "user", content: "Summarize the result." },
      ],
    };
    const response = {
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_456",
                function: { name: "save_weather", arguments: '{"city":"Paris"}' },
              },
            ],
          },
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={response} />);

    expect(screen.getByText("Summarize the result.")).toBeInTheDocument();
    expect(screen.getByText("save_weather")).toBeInTheDocument();
    await user.click(screen.getByText("HISTORY (2 messages)"));
    expect(screen.getByText("get_weather")).toBeInTheDocument();
    expect(screen.getByText("TOOL RESULT")).toBeInTheDocument();
    expect(screen.getByText("Sunny")).toBeInTheDocument();
  });

  it("collapses long tool results by default", async () => {
    const user = userEvent.setup();
    const longResult = "x".repeat(1201);
    const request = {
      messages: [
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "toolu_large", content: longResult }],
        },
      ],
    };

    render(<PrettyMessagesView request={request} response={{}} />);

    expect(screen.getByText("TOOL")).toBeInTheDocument();
    expect(screen.queryByText("USER")).not.toBeInTheDocument();
    expect(screen.getByText("TOOL RESULT")).toBeInTheDocument();
    expect(screen.getByText("toolu_large")).toBeInTheDocument();
    expect(screen.queryByText(longResult)).not.toBeInTheDocument();

    await user.click(screen.getByText("TOOL RESULT"));
    expect(screen.getByText(longResult)).toBeInTheDocument();
  });
});
