from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from tools import tier1_corrected_current7_slurm_harvest as harvest


class FakeSshError(Exception):
    pass


class FakeFleet:
    def __init__(
        self,
        files: dict[tuple[str, str], bytes],
        *,
        failures: dict[int, BaseException] | None = None,
        read_values: list[bytes] | None = None,
    ):
        self.files = {key: bytes(value) for key, value in files.items()}
        self.failures = dict(failures or {})
        self.read_values = list(read_values or [])
        self.events: list[tuple[str, str, str | int]] = []
        self.operation_attempt = 0
        self.sessions: list[FakeSession] = []
        self.sftps: list[FakeSftp] = []

    def fail_if_requested(self) -> None:
        attempt = self.operation_attempt
        self.operation_attempt += 1
        failure = self.failures.get(attempt)
        if failure is not None:
            raise failure


class FakeStream:
    def __init__(self, sftp: "FakeSftp", path: str):
        self.sftp = sftp
        self.path = path
        self.closed = False

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.closed = True

    def read(self, _maximum_bytes: int) -> bytes:
        fleet = self.sftp.fleet
        fleet.events.append(("read", self.sftp.account_name, self.path))
        fleet.fail_if_requested()
        if fleet.read_values:
            return fleet.read_values.pop(0)
        return fleet.files[(self.sftp.account_name, self.path)]


class FakeSftp:
    def __init__(self, fleet: FakeFleet, account_name: str):
        self.fleet = fleet
        self.account_name = account_name
        self.closed = False

    def stat(self, path: str) -> SimpleNamespace:
        self.fleet.events.append(("stat", self.account_name, path))
        self.fleet.fail_if_requested()
        value = self.fleet.files[(self.account_name, path)]
        return SimpleNamespace(st_size=len(value), st_mtime=1234, st_mode=0o444)

    def file(self, path: str, mode: str) -> FakeStream:
        assert mode == "rb"
        return FakeStream(self, path)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.fleet.events.append(("sftp-close", self.account_name, 0))


class FakeClient:
    def __init__(self, session: "FakeSession"):
        self.session = session

    def open_sftp(self) -> FakeSftp:
        fleet = self.session.fleet
        sftp = FakeSftp(fleet, self.session.account_name)
        fleet.sftps.append(sftp)
        fleet.events.append(("sftp-open", self.session.account_name, 0))
        return sftp


class FakeSession:
    def __init__(self, fleet: FakeFleet, account: SimpleNamespace):
        self.fleet = fleet
        self.account_name = account.name
        self.client = FakeClient(self)
        self.connected = False
        self.closed = False
        fleet.sessions.append(self)

    def ensure_connected(self) -> None:
        self.connected = True
        self.fleet.events.append(("ssh-connect", self.account_name, 0))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.fleet.events.append(("ssh-close", self.account_name, 0))


def _reader(fleet: FakeFleet, *account_names: str) -> harvest.AccountSftpReader:
    reader = harvest.AccountSftpReader.__new__(harvest.AccountSftpReader)
    reader._session_type = lambda account: FakeSession(fleet, account)
    reader._transport_exception_types = (
        ConnectionError,
        EOFError,
        TimeoutError,
    )
    reader._ssh_exception_type = FakeSshError
    reader._accounts = {name: SimpleNamespace(name=name) for name in account_names}
    reader._sessions = {}
    reader._sftp_clients = {}
    reader.read_count = 0
    reader.stat_count = 0
    reader.sftp_open_count = 0
    reader.sftp_reconnect_count = 0
    return reader


def _file_operations(fleet: FakeFleet) -> list[tuple[str, str, str | int]]:
    return [event for event in fleet.events if event[0] in {"read", "stat"}]


def test_stable_reads_reuse_one_channel_per_account_and_keep_exact_sequence():
    files = {
        ("account-a", "/one.json"): b'{"value": 1}\n',
        ("account-a", "/two.json"): b'{"value": 2}\n',
        ("account-b", "/other.json"): b'{"value": 3}\n',
    }
    fleet = FakeFleet(files)
    reader = _reader(fleet, "account-a", "account-b")

    one = harvest.read_stable_remote_file(
        reader,
        account_name="account-a",
        path="/one.json",
        maximum_bytes=100,
    )
    two = harvest.read_stable_remote_file(
        reader,
        account_name="account-a",
        path="/two.json",
        maximum_bytes=100,
    )
    other = harvest.read_stable_remote_file(
        reader,
        account_name="account-b",
        path="/other.json",
        maximum_bytes=100,
    )

    assert (one.payload, two.payload, other.payload) == (
        files[("account-a", "/one.json")],
        files[("account-a", "/two.json")],
        files[("account-b", "/other.json")],
    )
    assert _file_operations(fleet) == [
        (operation, account, path)
        for account, path in (
            ("account-a", "/one.json"),
            ("account-a", "/two.json"),
            ("account-b", "/other.json"),
        )
        for operation in ("stat", "read", "stat", "read", "stat")
    ]
    assert reader.stat_count == 9
    assert reader.read_count == 6
    assert reader.sftp_open_count == 2
    assert reader.sftp_reconnect_count == 0

    reader.close()
    assert all(sftp.closed for sftp in fleet.sftps)
    assert all(session.closed for session in fleet.sessions)
    assert reader._sftp_clients == {}
    assert reader._sessions == {}


def test_transport_failure_discards_account_and_retries_exactly_once():
    files = {("account-a", "/result.json"): b'{"sealed": true}\n'}
    fleet = FakeFleet(
        files,
        failures={0: ConnectionResetError(errno.ECONNRESET, "reset")},
    )
    reader = _reader(fleet, "account-a")

    stable = harvest.read_stable_remote_file(
        reader,
        account_name="account-a",
        path="/result.json",
        maximum_bytes=100,
    )

    assert stable.payload == files[("account-a", "/result.json")]
    assert _file_operations(fleet) == [
        ("stat", "account-a", "/result.json"),
        ("stat", "account-a", "/result.json"),
        ("read", "account-a", "/result.json"),
        ("stat", "account-a", "/result.json"),
        ("read", "account-a", "/result.json"),
        ("stat", "account-a", "/result.json"),
    ]
    assert reader.sftp_open_count == 2
    assert reader.sftp_reconnect_count == 1
    assert len(fleet.sessions) == 2
    assert fleet.sessions[0].closed is True
    assert fleet.sftps[0].closed is True
    reader.close()


def test_second_transport_failure_is_not_retried_again_and_fails_closed():
    files = {("account-a", "/result.json"): b'{"sealed": true}\n'}
    fleet = FakeFleet(
        files,
        failures={
            0: ConnectionResetError(errno.ECONNRESET, "first reset"),
            1: ConnectionResetError(errno.ECONNRESET, "second reset"),
        },
    )
    reader = _reader(fleet, "account-a")

    with pytest.raises(ConnectionResetError, match="second reset"):
        reader.stat("account-a", "/result.json")

    assert reader.sftp_open_count == 2
    assert reader.sftp_reconnect_count == 1
    assert len(_file_operations(fleet)) == 2
    assert all(sftp.closed for sftp in fleet.sftps)
    reader.close()
    assert all(session.closed for session in fleet.sessions)


def test_path_error_is_not_misclassified_as_transport_or_retried():
    files = {("account-a", "/missing.json"): b"unused"}
    fleet = FakeFleet(
        files,
        failures={0: FileNotFoundError(errno.ENOENT, "missing")},
    )
    reader = _reader(fleet, "account-a")

    with pytest.raises(FileNotFoundError):
        reader.stat("account-a", "/missing.json")

    assert reader.sftp_open_count == 1
    assert reader.sftp_reconnect_count == 0
    assert len(_file_operations(fleet)) == 1
    assert fleet.sftps[0].closed is True
    reader.close()


def test_byte_tamper_still_fails_stable_read_without_reconnect():
    original = b'{"sealed": true}\n'
    tampered = b'{"sealed": null}\n'
    assert len(original) == len(tampered)
    files = {("account-a", "/result.json"): original}
    fleet = FakeFleet(files, read_values=[original, tampered])
    reader = _reader(fleet, "account-a")

    with pytest.raises(RuntimeError, match="changed during stable read"):
        harvest.read_stable_remote_file(
            reader,
            account_name="account-a",
            path="/result.json",
            maximum_bytes=100,
        )

    assert _file_operations(fleet) == [
        (operation, "account-a", "/result.json")
        for operation in ("stat", "read", "stat", "read", "stat")
    ]
    assert reader.sftp_open_count == 1
    assert reader.sftp_reconnect_count == 0
    reader.close()
