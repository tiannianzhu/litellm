/**
 * Utility functions for parsing and formatting messages for pretty view
 */

import {
  MessageContentBlock,
  MessageRole,
  ParsedMessage,
  ParsedMessages,
  RequestPayload,
  ResponsePayload,
  RoleStyle,
  ToolCall,
} from "./prettyMessagesTypes";

/**
 * Role color styles for message cards - minimal, professional design
 * Color only used for labels and left border accent
 */
export const ROLE_STYLES: Record<string, RoleStyle> = {
  system: {
    background: "transparent",
    borderColor: "var(--color-muted-foreground)",
    label: "SYSTEM",
    labelColor: "var(--color-muted-foreground)",
  },
  user: {
    background: "transparent",
    borderColor: "var(--color-info)",
    label: "USER",
    labelColor: "var(--color-info)",
  },
  assistant: {
    background: "transparent",
    borderColor: "var(--color-success)",
    label: "ASSISTANT",
    labelColor: "var(--color-success)",
  },
  tool: {
    background: "transparent",
    borderColor: "var(--color-warning)",
    label: "TOOL RESULT",
    labelColor: "var(--color-warning)",
  },
};

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asRecord = (value: unknown): UnknownRecord | null => (isRecord(value) ? value : null);

const asString = (value: unknown, fallback = ""): string => (typeof value === "string" ? value : fallback);

const classifyRequest = (request: unknown): RequestPayload => {
  if (Array.isArray(request)) return { kind: "chat", messages: request };
  if (!isRecord(request)) return { kind: "unknown" };
  if (Array.isArray(request.messages)) return { kind: "chat", messages: request.messages };
  const { input } = request;
  if (typeof input === "string" || Array.isArray(input)) {
    return { kind: "responses", instructions: asString(request.instructions), input };
  }
  return { kind: "unknown" };
};

const classifyResponse = (response: unknown): ResponsePayload => {
  if (!isRecord(response)) return { kind: "unknown" };
  if (Array.isArray(response.choices)) return { kind: "chat", choices: response.choices };
  if (Array.isArray(response.output)) return { kind: "responses", output: response.output };
  return { kind: "unknown" };
};

/**
 * Parse request messages and response message from log data
 */
export const parseMessages = (request: unknown, response: unknown): ParsedMessages => ({
  requestMessages: parseRequestMessages(classifyRequest(request), request),
  responseMessage:
    parseResponseMessage(classifyResponse(response)) ??
    parseSearchResponse(response) ??
    parseMessage(response, "assistant"),
});

const parseRequestMessages = (payload: RequestPayload, request: unknown): ParsedMessage[] => {
  const topLevelSystem = isRecord(request) ? parseTopLevelSystem(request.system) : null;
  switch (payload.kind) {
    case "chat":
      return [
        ...(topLevelSystem ? [topLevelSystem] : []),
        ...payload.messages
          .map((message) => parseChatMessage(message))
          .filter((message): message is ParsedMessage => message !== null),
      ];
    case "responses": {
      const instructions: ParsedMessage[] = payload.instructions
        ? [{ role: "system", content: payload.instructions }]
        : [];
      const input: ParsedMessage[] =
        typeof payload.input === "string"
          ? [{ role: "user", content: payload.input }]
          : payload.input.flatMap(parseResponsesInputItem);
      return [...(topLevelSystem ? [topLevelSystem] : []), ...instructions, ...input];
    }
    case "unknown":
      return [];
  }
};

const parseResponseMessage = (payload: ResponsePayload): ParsedMessage | null => {
  switch (payload.kind) {
    case "chat": {
      const choice = payload.choices[0];
      const message = isRecord(choice) ? choice.message : undefined;
      return parseChatMessage(message, "assistant");
    }
    case "responses": {
      const messages = payload.output
        .filter((item): item is UnknownRecord => isRecord(item) && item.type === "message")
        .map((item) => parseMessage(item, "assistant"))
        .filter((message): message is ParsedMessage => message !== null);
      const toolCalls = payload.output.filter(isResponsesFunctionCall).map(parseResponsesFunctionCall);
      const content = messages
        .map((message) => message.content)
        .filter(Boolean)
        .join("\n");
      const contentBlocks = messages.flatMap((message) => message.contentBlocks ?? []);
      if (content.length === 0 && contentBlocks.length === 0 && toolCalls.length === 0) return null;
      return {
        role: "assistant",
        content,
        contentBlocks: contentBlocks.length > 0 ? contentBlocks : undefined,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      };
    }
    case "unknown":
      return null;
  }
};

const parseSearchResponse = (response: unknown): ParsedMessage | null => {
  const responseRecord = asRecord(response);
  if (responseRecord?.object !== "search" || !Array.isArray(responseRecord.results)) return null;
  return { role: "assistant", content: formatSearchResults(responseRecord.results) };
};

const parseChatMessage = (message: unknown, fallbackRole: MessageRole = "user"): ParsedMessage | null => {
  if (typeof message === "string") return { role: fallbackRole, content: message };
  return parseMessage(message, fallbackRole);
};

const parseResponsesInputItem = (item: unknown): ParsedMessage[] => {
  if (typeof item === "string") return [{ role: "user", content: item }];
  if (!isRecord(item)) return [];
  if (item.type === "function_call") {
    return [{ role: "assistant", content: "", toolCalls: [parseResponsesFunctionCall(item)] }];
  }
  if (item.type === "function_call_output") {
    const parsed = parseMessageContent(item.output);
    return [
      {
        role: "tool",
        content: parsed.content,
        contentBlocks: parsed.blocks,
        toolCallId: asString(item.call_id),
      },
    ];
  }
  if (item.type === "reasoning") return [];
  if ("role" in item || "content" in item) {
    const message = parseMessage(item);
    return message === null ? [] : [message];
  }
  return [];
};

const isResponsesFunctionCall = (item: unknown): item is UnknownRecord =>
  isRecord(item) && item.type === "function_call";

const parseResponsesFunctionCall = (item: UnknownRecord): ToolCall => ({
  id: asString(item.call_id) || asString(item.id),
  name: asString(item.name) || "unknown",
  arguments: parseToolArguments(item.arguments),
});

const stringify = (value: unknown): string => JSON.stringify(value, null, 2) ?? String(value ?? "");

const formatSearchResults = (results: unknown[]): string => {
  if (results.length === 0) return "No search results.";

  return results
    .map((value, index) => {
      const result = asRecord(value);
      if (!result) return `${index + 1}. ${stringify(value)}`;

      return [
        `${index + 1}. ${asString(result.title, `Result ${index + 1}`)}`,
        asString(result.url),
        asString(result.snippet, asString(result.content)),
      ]
        .filter(Boolean)
        .join("\n");
    })
    .join("\n\n");
};

const parseRole = (value: unknown, fallback: ParsedMessage["role"]): ParsedMessage["role"] => {
  switch (value) {
    case "developer":
      return "system";
    case "function":
      return "tool";
    case "system":
    case "user":
    case "assistant":
    case "tool":
      return value;
    default:
      return fallback;
  }
};

const parseTopLevelSystem = (system: unknown): ParsedMessage | null => {
  if (system === undefined || system === null) return null;

  const parsedContent = parseMessageContent(system);
  if (!parsedContent.content) return null;

  return {
    role: "system",
    content: parsedContent.content,
    contentBlocks: parsedContent.blocks,
  };
};

const parseMessage = (value: unknown, fallbackRole: ParsedMessage["role"] = "user"): ParsedMessage | null => {
  const message = asRecord(value);
  if (!message) return null;

  const sourceRole = parseRole(message.role, fallbackRole);
  const toolCalls = parseToolCalls(message.tool_calls);
  const hasContent = message.content !== undefined && message.content !== null;
  if (!hasContent && !toolCalls?.length) return null;

  const parsedContent = hasContent ? parseMessageContent(message.content) : { content: "" };
  const toolCallId = asString(message.tool_call_id);
  const contentBlocks =
    sourceRole === "tool" && toolCallId
      ? [
          {
            type: "tool_result" as const,
            toolUseId: toolCallId,
            content: parsedContent.content,
            isError: false,
          },
        ]
      : parsedContent.blocks;
  const role =
    sourceRole === "user" && contentBlocks?.length && contentBlocks.every((block) => block.type === "tool_result")
      ? "tool"
      : sourceRole;

  return {
    role,
    content: parsedContent.content,
    contentBlocks,
    toolCalls,
    toolCallId: toolCallId || undefined,
  };
};

const parseMessageContent = (content: unknown): { content: string; blocks?: MessageContentBlock[] } => {
  if (typeof content === "string") return { content };

  if (!Array.isArray(content)) return { content: stringify(content) };

  const blocks = content.map(parseContentBlock);
  return {
    content: blocks.map(contentBlockText).filter(Boolean).join("\n"),
    blocks,
  };
};

const parseContentBlock = (value: unknown): MessageContentBlock => {
  if (typeof value === "string") return { type: "text", text: value };

  const block = asRecord(value);
  if (!block) return { type: "unknown", label: "CONTENT", content: stringify(value) };

  const type = asString(block.type, "content");
  if (type === "text" || type === "input_text" || type === "output_text") {
    return { type: "text", text: asString(block.text) };
  }
  if (type === "tool_use") {
    return {
      type: "tool_use",
      tool: {
        id: asString(block.id),
        name: asString(block.name, "unknown"),
        arguments: asRecord(block.input) ?? (block.input === undefined ? {} : { input: block.input }),
      },
    };
  }
  if (type === "tool_result") {
    return {
      type: "tool_result",
      toolUseId: asString(block.tool_use_id),
      content: contentValueText(block.content),
      isError: block.is_error === true,
    };
  }
  if (type === "thinking" || type === "redacted_thinking") {
    return {
      type: "thinking",
      text:
        type === "redacted_thinking"
          ? "Thinking content was redacted."
          : asString(block.thinking, asString(block.text)),
      redacted: type === "redacted_thinking",
    };
  }
  if (type === "image" || type === "image_url" || type === "input_image") {
    return { type: "media", label: "Image" };
  }
  if (type === "document" || type === "input_file") {
    return { type: "media", label: "Document" };
  }

  return { type: "unknown", label: type.toUpperCase(), content: stringify(block) };
};

const contentValueText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return stringify(value);

  return value
    .map((item) => {
      const block = asRecord(item);
      return block && block.type === "text" ? asString(block.text) : stringify(item);
    })
    .join("\n");
};

const contentBlockText = (block: MessageContentBlock): string => {
  switch (block.type) {
    case "text":
      return block.text;
    case "tool_use":
      return `${block.tool.name}(${stringify(block.tool.arguments)})`;
    case "tool_result":
      return block.content;
    case "thinking":
      return block.text;
    case "media":
      return `[${block.label}]`;
    case "unknown":
      return block.content;
  }
};

/**
 * Parse tool calls from response message
 */
const parseToolCalls = (value: unknown): ToolCall[] | undefined => {
  if (!Array.isArray(value)) return undefined;

  return value.map((item) => {
    const toolCall = asRecord(item);
    const fn = asRecord(toolCall?.function);
    return {
      id: asString(toolCall?.id),
      name: asString(fn?.name, "unknown"),
      arguments: parseToolArguments(fn?.arguments),
    };
  });
};

/**
 * Parse tool arguments - handle both string and object formats
 */
const parseToolArguments = (args: unknown): Record<string, unknown> => {
  if (!args) return {};
  if (typeof args === "string") {
    try {
      const parsed = JSON.parse(args) as unknown;
      return asRecord(parsed) ?? { value: parsed };
    } catch {
      return { raw: args };
    }
  }
  return asRecord(args) ?? { value: args };
};
