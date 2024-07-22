# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from io import StringIO
from typing import Any, Dict, List, Optional

import paramiko


class SshClient:
    def __init__(
        self,
        hostname: str,
        user: str,
        passwd: Optional[str] = None,
        private_key: Optional[str] = None,
        port: int = 22,
        command_timeout: Optional[int] = 120,
        # pyre-ignore
        loop: Optional[Any] = None,
        disabled_algorithms: Optional[Dict[str, List[str]]] = None,
        max_stdout_read_bytes: int = 4096,
        max_stderr_read_bytes: int = 512,
    ) -> None:
        self.hostname = hostname
        self.user = user
        self.port = port
        self.command_timeout = command_timeout
        self.loop = loop
        self.disabled_algorithms = disabled_algorithms
        self._client: Optional[paramiko.SSHClient] = None
        # pyre-ignore
        self.sock = None
        if private_key:
            # pyre-ignore
            self.pkey = paramiko.Ed25519Key.from_private_key(StringIO(private_key))
        elif passwd:
            self.passwd: str = passwd
        self.max_stdout_read_bytes = max_stdout_read_bytes
        self.max_stderr_read_bytes = max_stderr_read_bytes
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self) -> "SshClient":
        if self._client is not None:
            raise RuntimeError("SSH Client is already in use")
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            assert self._client is not None

            if self.pkey:
                self._client.connect(
                    hostname=self.hostname,
                    port=self.port,
                    username=self.user,
                    sock=self.sock,
                    pkey=self.pkey,
                    disabled_algorithms=self.disabled_algorithms,
                )
                self.logger.debug("SSH Client established with Private Key")
            elif self.passwd:
                self._client.connect(
                    hostname=self.hostname,
                    port=self.port,
                    username=self.user,
                    password=self.passwd,
                    sock=self.sock,
                    disabled_algorithms=self.disabled_algorithms,
                )
                self.logger.debug("SSH Client established with Password")
            else:
                raise RuntimeError(
                    "SSH Client requires either a password or private key"
                )
        except Exception as e:
            self.logger.error(f"SSH Connection failed with error: {e}")
            raise e
        return self

    def __exit__(
        self,
        # pyre-ignore
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        # pyre-ignore
        exc_tb: Optional[Any],
    ) -> bool:
        if self._client:
            try:
                self._client.close()
            except OSError as e:
                self.logger.error(f"Failed to close SSH connection: {e}")
            finally:
                self._client = None
        return False

    async def __aenter__(self) -> "SshClient":
        return self.__enter__()

    async def __aexit__(
        self,
        # pyre-ignore
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        # pyre-ignore
        exc_tb: Optional[Any],
    ) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    def execute_command(self, command: str) -> str:
        if not self._client:
            raise RuntimeError("SSH Client is not connected")

        try:
            _stdin, _stdout, _stderr = self._client.exec_command(
                command=command, timeout=self.command_timeout
            )

            try:
                stdout = _stdout.read(self.max_stdout_read_bytes).decode()
            except UnicodeDecodeError:
                stdout = "[!] The command produced non-unicode output on stdout."

            try:
                stderr = _stderr.read(self.max_stderr_read_bytes).decode()
            except UnicodeDecodeError:
                stderr = "[!] The command produced non-unicode output on stderr."
        except TimeoutError:
            stdout = ""
            stderr = "[!] The command timed out."

        self.logger.debug(f"SSH stdout: {stdout}")
        self.logger.debug(f"SSH stderr: {stderr}")

        return stderr + stdout
