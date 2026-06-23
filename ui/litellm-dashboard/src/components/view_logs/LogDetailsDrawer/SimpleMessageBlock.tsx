/**
 * SimpleMessageBlock - Simple message display without collapsing
 * Used for messages in tree view and last user message
 */

import { cn } from "@/lib/cva.config";
import { MessageContentBlock, ToolCall } from "./prettyMessagesTypes";
import { SimpleToolCallBlock } from "./SimpleToolCallBlock";
import { MessageContentBlocks } from "./MessageContentBlocks";

interface SimpleMessageBlockProps {
  label: string;
  content?: string;
  contentBlocks?: MessageContentBlock[];
  toolCalls?: ToolCall[];
  isCompact?: boolean;
}

export function SimpleMessageBlock({
  label,
  content,
  contentBlocks,
  toolCalls,
  isCompact = false,
}: SimpleMessageBlockProps) {
  const hasContentBlocks = !!contentBlocks?.length;
  const displayContent = !hasContentBlocks && content && content !== "null" && content.length > 0 ? content : null;
  const hasToolCalls = !!toolCalls?.length;

  if (!displayContent && !hasContentBlocks && !hasToolCalls) {
    return null;
  }

  return (
    <div className={cn(isCompact && "mb-2")}>
      <span className="mb-[3px] block text-[10px] uppercase tracking-[0.5px] text-muted-foreground">{label}</span>

      {displayContent && (
        <div
          className={cn(
            "whitespace-pre-wrap break-words text-[13px] leading-[1.7] text-foreground",
            hasToolCalls && "mb-1.5",
          )}
        >
          {displayContent}
        </div>
      )}

      {hasContentBlocks && <MessageContentBlocks blocks={contentBlocks} compact={isCompact} />}

      {hasToolCalls && (
        <div>
          {toolCalls.map((tc, index) => (
            <SimpleToolCallBlock key={tc.id || index} tool={tc} compact={isCompact} />
          ))}
        </div>
      )}
    </div>
  );
}
