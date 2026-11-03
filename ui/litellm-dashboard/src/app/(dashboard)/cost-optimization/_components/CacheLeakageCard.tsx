"use client";

import React, { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Info } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  CacheLeakageDimension,
  CacheLeakageSort,
  computeCacheLeakage,
  leakageRowsFromKeyRows,
  netSavingsPerCachedToken,
  pct,
  sortCacheLeakageRows,
  usd,
} from "./costOptimizationUtils";
import { DailyActivityRange } from "./useDailyActivityRange";
import { useCacheLeakageKeys } from "./useCacheLeakageKeys";

interface CacheLeakageCardProps {
  activity: DailyActivityRange;
}

type SortColumn = CacheLeakageSort["column"];
type SortState = CacheLeakageSort;

const NATURAL_DIR: Record<SortColumn, "asc" | "desc"> = {
  uncachedPromptTokens: "desc",
  cacheHitRatio: "asc",
  potentialSavings: "desc",
};

const InfoTooltip = ({ info }: { info: string }) => (
  <Tooltip>
    <TooltipTrigger render={<span className="inline-flex" aria-label={info} />}>
      <Info className="h-3 w-3 text-muted-foreground" />
    </TooltipTrigger>
    <TooltipContent className="max-w-xs">{info}</TooltipContent>
  </Tooltip>
);

const SortableHead = ({
  column,
  label,
  info,
  sort,
  onSort,
}: {
  column: SortColumn;
  label: string;
  info: string;
  sort: SortState;
  onSort: (column: SortColumn) => void;
}) => {
  const active = sort.column === column;
  const ActiveArrow = sort.dir === "asc" ? ArrowUp : ArrowDown;
  const Arrow = active ? ActiveArrow : ArrowUpDown;
  return (
    <TableHead className="text-right">
      <span className="inline-flex items-center justify-end gap-1">
        <button
          type="button"
          onClick={() => onSort(column)}
          aria-label={`Sort by ${label}`}
          className="inline-flex items-center gap-1 font-medium hover:text-foreground"
        >
          {label}
          <Arrow className={`h-3 w-3 ${active ? "text-foreground" : "text-muted-foreground"}`} />
        </button>
        <InfoTooltip info={info} />
      </span>
    </TableHead>
  );
};

const CacheLeakageCard: React.FC<CacheLeakageCardProps> = ({ activity }) => {
  const { results, loading } = activity;
  const [dimension, setDimension] = useState<CacheLeakageDimension>("key");
  const [sort, setSort] = useState<SortState>({ column: "potentialSavings", dir: "desc" });
  const leakageRate = useMemo(() => netSavingsPerCachedToken(results), [results]);
  const keyLeakage = useCacheLeakageKeys(activity, dimension === "key");
  const unsortedRows = useMemo(
    () =>
      dimension === "key"
        ? leakageRowsFromKeyRows(keyLeakage.rows, leakageRate)
        : computeCacheLeakage(results, "model").rows,
    [dimension, keyLeakage.rows, leakageRate, results],
  );
  const rows = useMemo(() => sortCacheLeakageRows(unsortedRows, sort), [unsortedRows, sort]);
  const rowsLoading = dimension === "key" ? keyLeakage.loading : loading;

  const onSort = (column: SortColumn) =>
    setSort((prev) =>
      prev.column === column
        ? { column, dir: prev.dir === "asc" ? "desc" : "asc" }
        : { column, dir: NATURAL_DIR[column] },
    );

  const subject = dimension === "model" ? "Models" : "Keys";
  const firstColumn = dimension === "model" ? "Model" : "Key";
  const emptyNoun = dimension === "model" ? "model" : "key";
  const hasUsage = computeCacheLeakage(results, dimension).hasUsage;
  const emptyMessage = (() => {
    if (dimension === "key" && keyLeakage.failed) {
      return "Could not load key usage for this range.";
    }
    if (rowsLoading) {
      return "Loading...";
    }
    return hasUsage ? "No uncached input tokens in this range." : `No ${emptyNoun} usage in this range.`;
  })();
  const savingsInfo =
    dimension === "model"
      ? "Estimated from this model's own cached traffic: net cache savings after write premiums, divided by this model's cache read and write tokens. Shown as “Cannot estimate” when this model has no cache sample or caching is not currently net positive."
      : "Estimated from cached traffic across all keys: net cache savings after write premiums, divided by cache read and write tokens. Shown as “Cannot estimate” when caching is not currently net positive or there is no cache sample.";

  return (
    <TooltipProvider delay={300}>
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <CardTitle>Cache leakage by {dimension === "model" ? "model" : "virtual key"}</CardTitle>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                {subject} sending large volumes of uncached input with a low cache hit rate may benefit from prompt
                caching. Potential savings are approximate and use observed net cache savings after cache-write
                premiums.
              </p>
            </div>
          </div>
          <Tabs value={dimension} onValueChange={(value) => setDimension(value === "model" ? "model" : "key")}>
            <TabsList>
              <TabsTrigger value="key">By virtual key</TabsTrigger>
              <TabsTrigger value="model">By model</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">{emptyMessage}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{firstColumn}</TableHead>
                  <SortableHead
                    column="uncachedPromptTokens"
                    label="Uncached input tokens"
                    info="Input tokens you sent in this range that weren't served from or written to the cache"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="cacheHitRatio"
                    label="Cache hit rate"
                    info="Share of your input tokens that were served from the cache"
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="potentialSavings"
                    label="Potential savings"
                    info={savingsInfo}
                    sort={sort}
                    onSort={onSort}
                  />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-medium">
                      {row.label}
                      {row.sublabel && <span className="ml-1 text-xs text-muted-foreground">({row.sublabel})</span>}
                    </TableCell>
                    <TableCell className="text-right">{formatNumberWithCommas(row.uncachedPromptTokens)}</TableCell>
                    <TableCell className="text-right">{pct(row.cacheHitRatio)}</TableCell>
                    <TableCell className="text-right">
                      {row.potentialSavings == null ? "Cannot estimate" : usd(row.potentialSavings)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

export default CacheLeakageCard;
