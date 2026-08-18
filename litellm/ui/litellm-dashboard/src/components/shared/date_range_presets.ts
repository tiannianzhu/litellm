import type { ParseKeys } from "i18next";
import moment, { type Moment } from "moment";

export interface RelativeTimeOption {
  label: ParseKeys;
  shortLabel: string;
  settingValue: string;
  getValue: (day?: string) => { from: Date; to: Date };
}

export const BEIJING_TIME_ZONE = "Asia/Shanghai";

const beijingFormatOptions: Intl.DateTimeFormatOptions = {
  timeZone: BEIJING_TIME_ZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
};
const beijingFormatter = new Intl.DateTimeFormat("en-US", beijingFormatOptions);

const beijingParts = (date: Date): Record<string, string> =>
  Object.fromEntries(beijingFormatter.formatToParts(date).map(({ type, value }) => [type, value]));

export const beijingDay = (date: Date): string => {
  const parts = beijingParts(date);
  return `${parts.year}-${parts.month}-${parts.day}`;
};

export const utcDay = (date: Date): string => date.toISOString().slice(0, 10);

const beijingOffset = (time: number): number => {
  const parts = beijingParts(new Date(time));
  return (
    Date.UTC(
      Number(parts.year),
      Number(parts.month) - 1,
      Number(parts.day),
      Number(parts.hour),
      Number(parts.minute),
      Number(parts.second),
    ) - time
  );
};

const startOfBeijingDay = (date: Date): Date => {
  const localMidnight = Date.UTC(date.getFullYear(), date.getMonth(), date.getDate());
  const approximate = localMidnight - beijingOffset(localMidnight);
  return new Date(localMidnight - beijingOffset(approximate));
};

export const beijingDayBounds = (from: Date, to: Date): { from: Date; to: Date } => ({
  from: startOfBeijingDay(from),
  to: new Date(startOfBeijingDay(new Date(to.getFullYear(), to.getMonth(), to.getDate() + 1)).getTime() - 1),
});

const presetStarts = {
  today: (today: Moment) => today.clone(),
  last_7_days: (today: Moment) => today.clone().subtract(6, "days"),
  last_30_days: (today: Moment) => today.clone().subtract(29, "days"),
  month_to_date: (today: Moment) => today.clone().startOf("month"),
  year_to_date: (today: Moment) => today.clone().startOf("year"),
};

export type RelativeTimePreset = keyof typeof presetStarts;

export const presetRange = (preset: RelativeTimePreset, day = beijingDay(new Date())): { from: Date; to: Date } => {
  const today = moment(day, "YYYY-MM-DD", true);
  return { from: presetStarts[preset](today).startOf("day").toDate(), to: today.endOf("day").toDate() };
};

export const relativeTimeOptions = [
  {
    label: "shared.dateRangePresets.today",
    shortLabel: "today",
    settingValue: "today",
    getValue: (day?: string) => presetRange("today", day),
  },
  {
    label: "shared.dateRangePresets.last7Days",
    shortLabel: "7d",
    settingValue: "last_7_days",
    getValue: (day?: string) => presetRange("last_7_days", day),
  },
  {
    label: "shared.dateRangePresets.last30Days",
    shortLabel: "30d",
    settingValue: "last_30_days",
    getValue: (day?: string) => presetRange("last_30_days", day),
  },
  {
    label: "shared.dateRangePresets.monthToDate",
    shortLabel: "MTD",
    settingValue: "month_to_date",
    getValue: (day?: string) => presetRange("month_to_date", day),
  },
  {
    label: "shared.dateRangePresets.yearToDate",
    shortLabel: "YTD",
    settingValue: "year_to_date",
    getValue: (day?: string) => presetRange("year_to_date", day),
  },
] as const satisfies readonly RelativeTimeOption[];

export const utcRelativeTimeOptions: readonly RelativeTimeOption[] = relativeTimeOptions.map((option) => ({
  ...option,
  getValue: () => option.getValue(utcDay(new Date())),
}));
