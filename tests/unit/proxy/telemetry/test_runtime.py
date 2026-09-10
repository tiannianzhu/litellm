import asyncio
from typing import Final, Literal

import httpx
import pytest
from pydantic import HttpUrl

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.runtime import TelemetryRuntime, deployment_hasher
from litellm.proxy.telemetry.settings import TelemetryDestination, TelemetrySettings
from litellm.telemetry.consent import ConsentGatedSink, TelemetryConsent
from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, TelemetryGroup, UIEvent
from tests.unit.proxy.telemetry.fake_database import SettingsDatabase

_ENDPOINT: Final = HttpUrl("http://telemetry.invalid/v1/reports")


def _offline() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(204)))


_HEARTBEAT: Final = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT}))
_SUCCESS: Final = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.REQUEST_SUCCESS}))


async def _started(settings: TelemetrySettings, db: SettingsDatabase | None) -> TelemetryRuntime:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=settings, db=lambda: db, register=lambda _logger: None, http_client=_offline
    )
    return runtime


@pytest.mark.parametrize(
    "settings",
    [
        TelemetrySettings(),
        TelemetrySettings(groups="", endpoint=_ENDPOINT),
        TelemetrySettings(groups="heartbeat"),
        TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, disabled=True),
    ],
)
@pytest.mark.asyncio
async def test_without_a_database_telemetry_needs_valid_pinned_groups_an_endpoint_and_no_veto(
    settings: TelemetrySettings,
) -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=settings, db=lambda: None, register=registered.append, http_client=_offline
    )
    assert runtime.sink is None
    assert registered == []


@pytest.mark.asyncio
async def test_a_veto_never_opens_the_database() -> None:
    opened: Final[list[bool]] = []  # mutable-ok: records whether the db factory ran

    def _db() -> SettingsDatabase:
        opened.append(True)
        return SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')

    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=TelemetrySettings(disabled=True), db=_db, register=lambda _logger: None
    )
    assert (runtime.sink, runtime.store, opened) == (None, None, [])


@pytest.mark.asyncio
async def test_pinned_groups_register_the_attempt_logger_and_stop_cleanly() -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="HEARTBEAT", endpoint=_ENDPOINT, destination=TelemetryDestination.HTTP),
        db=lambda: None,
        register=registered.append,
        http_client=_offline,
    )
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    assert len(registered) == 1
    await runtime.stop()
    assert runtime.sink is None


@pytest.mark.asyncio
async def test_pinned_groups_win_over_the_stored_groups() -> None:
    db: Final = SettingsDatabase(
        stored_groups='{"groups": ["heartbeat", "request_success"], "destination": "local_table"}'
    )
    runtime: Final = await _started(TelemetrySettings(groups="heartbeat"), db)
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    await runtime.stop()


@pytest.mark.asyncio
async def test_stored_groups_are_picked_up_at_the_next_window_without_a_restart() -> None:
    db: Final = SettingsDatabase()
    runtime: Final = await _started(TelemetrySettings(), db)
    assert runtime.sink is None
    db.stored_groups = '{"groups": ["heartbeat", "request_success"], "destination": "local_table"}'
    await runtime.refresh()
    assert runtime.sink is not None and runtime.sink.consent == _SUCCESS
    db.stored_groups = '{"groups": [], "destination": "local_table"}'
    await runtime.refresh()
    assert runtime.sink is None
    await runtime.stop()


@pytest.mark.asyncio
async def test_a_proxy_that_never_opts_in_registers_no_logger_and_writes_nothing() -> None:
    db: Final = SettingsDatabase()
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(),
        db=lambda: db,
        register=registered.append,
        http_client=_offline,
    )
    await runtime.refresh()
    assert (runtime.sink, registered, db.inserted) == (None, [], ())

    for stored in ('{"groups": ["heartbeat"], "destination": "local_table"}', '{"groups": []}'):
        db.stored_groups = stored
        await runtime.refresh()
    db.stored_groups = '{"groups": ["heartbeat"], "destination": "local_table"}'
    await runtime.refresh()
    assert runtime.sink is not None
    assert len(registered) == 1
    assert db.inserted.count("telemetry_deployment_hash_secret") == 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_invalid_stored_groups_are_ignored_and_telemetry_stays_off() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["token_info"]}')
    runtime: Final = await _started(TelemetrySettings(), db)
    assert runtime.sink is None
    await runtime.stop()


@pytest.mark.asyncio
async def test_a_failing_settings_read_keeps_the_current_groups() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"], "destination": "local_table"}')
    runtime: Final = await _started(TelemetrySettings(), db)
    db.fail_reads = True
    await runtime.refresh()
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    await runtime.stop()


@pytest.mark.asyncio
async def test_groups_without_a_chosen_destination_record_nothing_until_one_is_saved() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')
    runtime: Final = await _started(TelemetrySettings(groups="heartbeat"), db)
    assert (runtime.sink, runtime.destination) == (None, None)
    db.stored_groups = '{"groups": [], "destination": "local_table"}'
    await runtime.refresh()
    assert runtime.sink is not None and runtime.destination is TelemetryDestination.LOCAL_TABLE
    await runtime.stop()


@pytest.mark.asyncio
async def test_the_env_destination_wins_over_the_stored_one() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"], "destination": "local_table"}')
    runtime: Final = await _started(TelemetrySettings(endpoint=_ENDPOINT, destination=TelemetryDestination.HTTP), db)
    assert runtime.sink is not None and runtime.destination is TelemetryDestination.HTTP
    await runtime.stop()


@pytest.mark.asyncio
async def test_a_stored_http_destination_without_an_endpoint_records_nothing() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"], "destination": "http"}')
    runtime: Final = await _started(TelemetrySettings(), db)
    assert runtime.sink is None
    await runtime.stop()


def test_deployment_hashes_are_stable_per_secret_and_differ_across_secrets() -> None:
    assert deployment_hasher(b"secret-a")("model-1") == deployment_hasher(b"secret-a")("model-1")
    assert deployment_hasher(b"secret-a")("model-1") != deployment_hasher(b"secret-b")("model-1")
    assert "model-1" not in deployment_hasher(b"secret-a")("model-1")


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_request_finalizers_before_the_last_flush() -> None:
    runtime: Final = await _started(
        TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, destination=TelemetryDestination.HTTP), None
    )
    finished: Final[list[bool]] = []  # mutable-ok: records that the finalizer ran to completion

    async def _finalizer() -> None:
        await asyncio.sleep(0.01)
        finished.append(True)

    runtime.spawn(_finalizer())
    await runtime.stop()
    assert finished == [True]


@pytest.mark.asyncio
async def test_stopping_a_runtime_that_never_started_is_a_no_op() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.stop()
    assert runtime.sink is None


class _HeldFlushSink:
    def __init__(self) -> None:
        self.started: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.finished: Final = asyncio.Event()
        self.flushes: Final[list[bool]] = []  # mutable-ok: records each flush that ran to completion

    def set_instance(self, info: InstanceInfo) -> None: ...

    def record_request(self, record: RequestRecord) -> None: ...

    def record_attempt(self, record: AttemptRecord) -> None: ...

    def record_ui_event(self, event: UIEvent) -> None: ...

    async def flush(self) -> None:
        self.started.set()
        try:
            await self.release.wait()
            self.flushes.append(True)
        finally:
            self.finished.set()


@pytest.mark.asyncio
async def test_stop_during_an_export_lets_that_export_finish() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, destination=TelemetryDestination.HTTP),
        flush_interval_s=0.001,
        db=lambda: None,
        register=lambda _logger: None,
        http_client=_offline,
    )
    held: Final = _HeldFlushSink()
    runtime.sink = ConsentGatedSink(held, _HEARTBEAT)
    await held.started.wait()
    stopping: Final = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    held.release.set()
    await stopping
    assert len(held.flushes) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ("cancel", "deadline", "expired"))
async def test_interrupted_stop_cancels_exports_and_request_finalizers(
    interruption: Literal["cancel", "deadline", "expired"],
) -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, destination=TelemetryDestination.HTTP),
        flush_interval_s=0.001,
        db=lambda: None,
        register=lambda _logger: None,
        http_client=_offline,
    )
    held: Final = _HeldFlushSink()
    runtime.sink = ConsentGatedSink(held, _HEARTBEAT)
    finalizer_started: Final = asyncio.Event()
    finalizer_release: Final = asyncio.Event()
    finalizer_finished: Final = asyncio.Event()

    async def finalize() -> None:
        finalizer_started.set()
        try:
            await finalizer_release.wait()
        finally:
            finalizer_finished.set()

    runtime.spawn(finalize())
    try:
        await asyncio.wait_for(held.started.wait(), timeout=5)
        await asyncio.wait_for(finalizer_started.wait(), timeout=5)
        if interruption == "cancel":
            stopping: Final = asyncio.create_task(runtime.stop())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            stopping.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(stopping, timeout=5)
        else:
            deadline: Final = asyncio.get_running_loop().time() + (0.01 if interruption == "deadline" else -1)
            with pytest.raises(asyncio.TimeoutError):
                await runtime.stop(shutdown_deadline=deadline)
        assert (runtime.sink, runtime.store, held.finished.is_set(), finalizer_finished.is_set()) == (
            None, None, True, True
        )
    finally:
        held.release.set()
        finalizer_release.set()
        await asyncio.wait_for(runtime.stop(), timeout=5)


async def test_settle_timeout_reads_the_configured_value_after_start_and_the_default_before() -> None:
    runtime: Final = TelemetryRuntime()
    assert runtime.settings.settle_timeout_seconds == 2.0
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, settle_timeout_seconds=7.5),
        db=lambda: None,
        register=lambda _logger: None,
        http_client=_offline,
    )
    assert runtime.settings.settle_timeout_seconds == 7.5
    await runtime.stop()
