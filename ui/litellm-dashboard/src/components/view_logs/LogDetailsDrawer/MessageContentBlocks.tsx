import { useState } from "react";
import { ChevronDown, ChevronRight, FileImage, FileText } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { MessageContentBlock } from "./prettyMessagesTypes";
import { SimpleToolCallBlock } from "./SimpleToolCallBlock";

interface MessageContentBlocksProps {
  blocks: MessageContentBlock[];
  compact?: boolean;
}

interface CollapsibleBlockProps {
  label: string;
  metadata?: string;
  content: string;
  defaultExpanded: boolean;
  error?: boolean;
}

function CollapsibleBlock({ label, metadata, content, defaultExpanded, error = false }: CollapsibleBlockProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <Collapsible
      open={isExpanded}
      onOpenChange={setIsExpanded}
      className={`mt-2 border-l-2 pl-2.5 ${error ? "border-destructive" : "border-border"}`}
    >
      <CollapsibleTrigger
        className={`flex w-full items-center gap-1.5 py-0.5 text-left ${
          error ? "text-destructive" : "text-muted-foreground"
        }`}
      >
        {isExpanded ? <ChevronDown className="size-2.5" /> : <ChevronRight className="size-2.5" />}
        <span className="text-[10px] font-medium">{label}</span>
        {metadata ? <code className="text-[10px] text-muted-foreground">{metadata}</code> : null}
        <span className="text-[10px] text-muted-foreground">({content.length.toLocaleString()} chars)</span>
      </CollapsibleTrigger>
      <CollapsibleContent className="max-h-80 overflow-auto whitespace-pre-wrap break-words py-1.5 pb-0.5 font-mono text-xs leading-relaxed text-foreground">
        {content}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function MessageContentBlocks({ blocks, compact = false }: MessageContentBlocksProps) {
  return (
    <div>
      {blocks.map((block, index) => {
        switch (block.type) {
          case "text":
            return block.text ? (
              <div
                key={index}
                className={`${index === 0 ? "" : "mt-1.5"} whitespace-pre-wrap break-words text-[13px] leading-[1.7] text-foreground`}
              >
                {block.text}
              </div>
            ) : null;
          case "tool_use":
            return <SimpleToolCallBlock key={block.tool.id || index} tool={block.tool} compact={compact} />;
          case "tool_result":
            return (
              <CollapsibleBlock
                key={`${block.toolUseId}-${index}`}
                label={block.isError ? "TOOL ERROR" : "TOOL RESULT"}
                metadata={block.toolUseId}
                content={block.content}
                defaultExpanded={block.content.length <= 1200}
                error={block.isError}
              />
            );
          case "thinking":
            return (
              <CollapsibleBlock
                key={index}
                label={block.redacted ? "REDACTED THINKING" : "THINKING"}
                content={block.text}
                defaultExpanded={false}
              />
            );
          case "media":
            return (
              <div key={index} className="mt-2 flex items-center gap-1.5 text-muted-foreground">
                {block.label === "Image" ? <FileImage className="size-3.5" /> : <FileText className="size-3.5" />}
                <span className="text-xs">{block.label}</span>
              </div>
            );
          case "unknown":
            return <CollapsibleBlock key={index} label={block.label} content={block.content} defaultExpanded={false} />;
        }
      })}
    </div>
  );
}
