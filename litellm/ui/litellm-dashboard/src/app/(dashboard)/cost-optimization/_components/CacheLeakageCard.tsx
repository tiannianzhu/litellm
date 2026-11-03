"use client";

import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  const active = sort.column === column;
  const ActiveArrow = sort.dir === "asc" ? ArrowUp : ArrowDown;
  const Arrow = active ? ActiveArrow : ArrowUpDown;
  return (
    <TableHead className="text-right">
      <span className="inline-flex items-center justify-end gap-1">
        <button
          type="button"
          onClick={() => onSort(column)}
          aria-label={t("costOptimization.cacheLeakage.sortBy", { label })}
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
  const { t } = useTranslation();
  const { results, loading } = activity;
  const [dimension, setDimension] = useState<CacheLeakageDimension>("key");
  const [sort, setSort] = useState<SortState>({ column: "potentialSavings", dir: "desc" });
  const leakageRate = useMemo(() => netSavingsPerCachedToken(results), [results]);
  const keyLeakage = useCacheLeakageKeys(activity, dimension === "key");
  const { rows: unsortedRows, hasUsage } = useMemo(
    () =>
      dimension === "key"
        ? {
            rows: leakageRowsFromKeyRows(keyLeakage.rows, leakageRate),
            hasUsage: results.some((day) => Object.keys(day.breakdown?.api_keys ?? {}).length > 0),
          }
        : computeCacheLeakage(results, "model"),
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

  const subject =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.subjects.models")
      : t("costOptimization.cacheLeakage.subjects.keys");
  const firstColumn =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.columns.model")
      : t("costOptimization.cacheLeakage.columns.key");
  const emptyNoun =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.nouns.model")
      : t("costOptimization.cacheLeakage.nouns.key");
  const emptyUsageMessage = hasUsage
    ? t("costOptimization.cacheLeakage.noUncachedInput")
    : t("costOptimization.cacheLeakage.empty", { noun: emptyNoun });
  const emptyMessage =
    dimension === "key" && keyLeakage.failed ? t("costOptimization.cacheLeakage.keyLoadFailed") : emptyUsageMessage;
  const savingsInfo =
    dimension === "model"
      ? t("costOptimization.cacheLeakage.columns.potentialSavings.infoModel")
      : t("costOptimization.cacheLeakage.columns.potentialSavings.info");

  return (
    <TooltipProvider delay={300}>
      <Card>
        <CardHeader>
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <CardTitle>
                {t("costOptimization.cacheLeakage.title", {
                  dimension:
                    dimension === "model"
                      ? t("costOptimization.cacheLeakage.nouns.model")
                      : t("costOptimization.cacheLeakage.nouns.virtualKey"),
                })}
              </CardTitle>
              <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                {t("costOptimization.cacheLeakage.description", { subject })}
              </p>
            </div>
          </div>
          <Tabs value={dimension} onValueChange={(value) => setDimension(value === "model" ? "model" : "key")}>
            <TabsList>
              <TabsTrigger value="key">{t("costOptimization.cacheLeakage.tabs.byKey")}</TabsTrigger>
              <TabsTrigger value="model">{t("costOptimization.cacheLeakage.tabs.byModel")}</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardHeader>
        <CardContent>
          {rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              {rowsLoading ? t("costOptimization.cacheLeakage.loading") : emptyMessage}
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{firstColumn}</TableHead>
                  <SortableHead
                    column="uncachedPromptTokens"
                    label={t("costOptimization.cacheLeakage.columns.uncachedTokens.label")}
                    info={t("costOptimization.cacheLeakage.columns.uncachedTokens.info")}
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="cacheHitRatio"
                    label={t("costOptimization.cacheLeakage.columns.cacheHitRate.label")}
                    info={t("costOptimization.cacheLeakage.columns.cacheHitRate.info")}
                    sort={sort}
                    onSort={onSort}
                  />
                  <SortableHead
                    column="potentialSavings"
                    label={t("costOptimization.cacheLeakage.columns.potentialSavings.label")}
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
                      {row.potentialSavings == null
                        ? t("costOptimization.cacheLeakage.cannotEstimate")
                        : usd(row.potentialSavings)}
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
