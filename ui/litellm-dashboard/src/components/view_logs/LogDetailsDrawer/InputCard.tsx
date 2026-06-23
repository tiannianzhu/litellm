/**
 * InputCard - Displays all input messages with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

import { useState } from "react";
import { toast } from "@/lib/toast";
import { ParsedMessage } from "./prettyMessagesTypes";
import { SectionHeader } from "./SectionHeader";
import { CollapsibleMessage } from "./CollapsibleMessage";
import { HistoryTree } from "./HistoryTree";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

interface InputCardProps {
  messages: ParsedMessage[];
  promptTokens?: number;
  inputCost?: number;
}

export function InputCard({ messages, promptTokens, inputCost }: InputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  const systemContent = messages
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .filter(Boolean)
    .join("\n\n");
  const nonSystemMessages = messages.filter((m) => m.role !== "system");
  const lastMessage = nonSystemMessages.length > 0 ? nonSystemMessages[nonSystemMessages.length - 1] : null;
  const historyMessages = nonSystemMessages.slice(0, -1);

  const handleCopy = () => {
    const content = lastMessage?.content || "";
    navigator.clipboard.writeText(content);
    toast.success("Input copied");
  };

  return (
    <div
      style={{
        border: "1px solid var(--color-border)",
        borderRadius: 6,
        marginBottom: 8,
        overflow: "hidden",
      }}
    >
      {/* Datadog-style Header */}
      <SectionHeader
        type="input"
        tokens={promptTokens}
        cost={inputCost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      {/* Content */}
      <div
        style={{
          maxHeight: isCollapsed ? "0px" : "10000px",
          overflow: "hidden",
          transition: "max-height 0.3s ease-out, opacity 0.3s ease-out",
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div style={{ padding: "12px 16px" }}>
          {systemContent && (
            <CollapsibleMessage label="SYSTEM" content={systemContent} defaultExpanded={systemContent.length < 200} />
          )}

          {/* History - Tree style, collapsed by default */}
          {historyMessages.length > 0 && <HistoryTree messages={historyMessages} />}

          {/* Last User Message - Always visible */}
          {lastMessage && (
            <SimpleMessageBlock
              label={lastMessage.role.toUpperCase()}
              content={lastMessage.content}
              contentBlocks={lastMessage.contentBlocks}
              toolCalls={lastMessage.toolCalls}
            />
          )}
        </div>
      </div>
    </div>
  );
}
