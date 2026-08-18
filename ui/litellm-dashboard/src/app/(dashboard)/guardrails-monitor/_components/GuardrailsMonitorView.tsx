import { Page } from "@/components/shared/Page";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { parseAsString, useQueryState } from "nuqs";
import React, { useCallback, useState } from "react";
import { formatDate } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { presetRange } from "@/components/shared/date_range_presets";
import { GuardrailDetail } from "./GuardrailDetail";
import { GuardrailsOverview } from "./GuardrailsOverview";

interface GuardrailsMonitorViewProps {
  accessToken?: string | null;
}

export default function GuardrailsMonitorView({ accessToken = null }: GuardrailsMonitorViewProps) {
  const [selectedGuardrailId, setSelectedGuardrailId] = useQueryState(
    "guardrail",
    parseAsString.withOptions({ history: "push" }),
  );

  const [dateValue, setDateValue] = useState<DateRangePickerValue>(() => presetRange("last_7_days"));

  const startDate = dateValue.from ? formatDate(dateValue.from) : "";
  const endDate = dateValue.to ? formatDate(dateValue.to) : "";

  const handleDateChange = useCallback((newValue: DateRangePickerValue) => {
    setDateValue(newValue);
  }, []);

  const handleSelectGuardrail = (id: string) => {
    void setSelectedGuardrailId(id);
  };

  const handleBack = () => {
    void setSelectedGuardrailId(null, { history: "replace" });
  };

  const dateRangeControl = (
    <AdvancedDatePicker
      value={dateValue}
      onValueChange={handleDateChange}
      label=""
      showTimeRange={false}
      timeZoneLabel="Beijing time (Asia/Shanghai)"
    />
  );

  return (
    <Page>
      {!selectedGuardrailId ? (
        <GuardrailsOverview
          accessToken={accessToken}
          startDate={startDate}
          endDate={endDate}
          onSelectGuardrail={handleSelectGuardrail}
          dateRangeControl={dateRangeControl}
        />
      ) : (
        <>
          <div className="mb-4 flex items-center justify-end">{dateRangeControl}</div>
          <GuardrailDetail
            guardrailId={selectedGuardrailId}
            onBack={handleBack}
            accessToken={accessToken}
            startDate={startDate}
            endDate={endDate}
          />
        </>
      )}
    </Page>
  );
}
