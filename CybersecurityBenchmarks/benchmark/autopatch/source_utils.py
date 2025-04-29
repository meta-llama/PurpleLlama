# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import time
from abc import ABC, abstractmethod
from logging import Logger, LoggerAdapter
from pathlib import Path
from subprocess import CalledProcessError
from types import TracebackType
from typing import Callable, Dict, List, Optional, Type

# pyre-fixme[21]: Could not find name `Node`, `Parser` in `tree_sitter`.
from tree_sitter import Language, Node as TreeSitterNode, Parser

from .tools import subprocess_run

from .types import PathAndLine

LOGGER: Logger = logging.getLogger(__name__)


class SourceUtilsInterface(ABC):
    """
    This class defines the interface for retrieving source code information about the repo.
    It enables better testability as we can use a fake source utils implementation in the tests.
    """

    @abstractmethod
    async def retrieve_src_of_func(self, path_and_line: PathAndLine) -> Optional[str]:
        """
        Retrieves the source code of the function encompassing `path_and_line` from a stacktrace.

        Args:
            path_and_line: A path and line from a stacktrace.
            repo_hash: Repo hash to use. If None, the latest repo hash will be used.

        Returns:
            Source code of the function. None if the function is not found.
        """

    @staticmethod
    @abstractmethod
    def _get_repo_root() -> str:
        """
        Returns the root of the repo.
        """

    @classmethod
    def get_absolute_path(cls, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            # If the filename is not absolute, we assume it is relative to the repo root.
            return cls._get_repo_root() / path
        return path

    @staticmethod
    @abstractmethod
    async def save_current_changes_to_file(
        output_patch: Path, patched_src_path: Optional[str] = None
    ) -> None:
        """
        Saves the current changes to a file.
        """

    @staticmethod
    @abstractmethod
    async def revert_current_changes(hard: bool = False) -> None:
        """
        Reverts the current changes.

        Args:
            hard: If True, revert all changes including untracked files. If False, revert only tracked changes.
        """

    @classmethod
    def add_dummy_line(cls, filename: str) -> None:
        """
        Adds a dummy line to the end of the file.
        """
        ext = Path(filename).suffix
        if ext in [".h", ".hpp", ".hh", ".c", ".cpp", ".cc"]:
            comment_header = "// "
        elif ext in [".py"]:
            comment_header = "# "
        else:
            raise RuntimeError(f"Unsupported file extension: {ext}")

        with open(cls.get_absolute_path(filename), "a") as f:
            f.write(f"\n{comment_header} dummy line\n")


class RepoRevisionHelper:
    """
    This context manager helps checkout the repo to a specific revision and then revert to the previous revision when the context manager exits.
    """

    def __init__(
        self, target_revision: Optional[str], logger: Optional[Logger] = None
    ) -> None:
        """
        Args:
            target_revision: The revision to checkout to. If None, the context manager will not perform any actions, including checking out revisions and shelving.
        """
        self._target_revision: Optional[str] = target_revision
        self._prev_revision: Optional[str] = None
        self._logger: Logger = logger or LOGGER

    async def __aenter__(self) -> None:
        if self._target_revision is None:
            # we only maintain the context when a target revision is provided
            return
        self.shelve()  # shelve the uncommited changes
        self._prev_revision = self.get_repo_revision()
        self._target_revision: str
        self.checkout_repo_revision(self._target_revision)

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if self._target_revision is None:
            return False  # Do not suppress exceptions raised in the context

        if self._prev_revision:
            self.checkout_repo_revision(self._prev_revision)
        self.unshelve()
        return False  # Do not suppress exceptions raised in the context

    def checkout_repo_revision(self, repo_hash: str) -> None:
        """
        Checks out the current repo to the specified revision hash.
        """
        self._logger.debug(f"checking out repo to revision: {repo_hash}")
        try:
            subprocess_run(["hg", "update", repo_hash, "-C"], check=True)
        except CalledProcessError as e:
            raise RuntimeError(
                f"Failed to checkout repo to revision: {repo_hash}. Error: {e}"
            )

    def get_repo_revision(self) -> str:
        """
        Returns the current revision (full hash) of the repo.
        """
        try:
            result = subprocess_run(["hg", "--debug", "id", "-i"])
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to get repo revision. Error: {e}")
        return result.stdout.splitlines()[-1].strip()

    def shelve(self) -> None:
        """
        Shelves the current repo.
        """
        # We do not check for success, just in case there are no changes.
        _ = subprocess_run(["hg", "shelve"])

    def unshelve(self) -> None:
        """
        Unshelves the shelved changes.
        """
        # We do not check for success, just in case there is no shelved changes.
        _ = subprocess_run(["hg", "unshelve"])


class FakeRepoRevisionHelper:
    """
    This class helps to fake the repo revision.
    """

    def __init__(self, target_revision: Optional[str]) -> None:
        self._target_revision: Optional[str] = target_revision

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        return False


class FakeSourceUtils(SourceUtilsInterface):
    def __init__(self, fake_source_config: Optional[Dict[str, str]] = None) -> None:
        self._fake_source_config: Optional[Dict[str, str]] = fake_source_config

    @staticmethod
    def _get_repo_root() -> str:
        return ""

    @staticmethod
    async def save_current_changes_to_file(
        output_patch: Path, patched_src_path: Optional[str] = None
    ) -> None:
        pass

    @staticmethod
    async def revert_current_changes(hard: bool = False) -> None:
        pass

    async def retrieve_src_of_func(self, path_and_line: PathAndLine) -> Optional[str]:
        if self._fake_source_config is None:
            return None
        if path_and_line.file_path not in self._fake_source_config:
            return None
        return self._fake_source_config[path_and_line.file_path]


def ts_traverse(
    # pyre-fixme[11]: Annotation `TreeSitterNode` is not defined as a type.
    node: TreeSitterNode,
    collected_nodes: List[TreeSitterNode],
    fn: Callable[[TreeSitterNode, List[TreeSitterNode]], bool],
) -> None:
    """
    Traverse the AST and collect nodes using the provided function. The function should return True
    if traversal should continue on the node's children and False otherwise.

    Args:
        node: The node to traverse.
        collected_nodes: Collected nodes will be stored in this list.
        fn: The function to use to collect nodes.
    """
    if fn(node, collected_nodes):
        for child in node.children:
            ts_traverse(child, collected_nodes, fn=fn)


class TreeSitterSourceUtils(SourceUtilsInterface):
    """
    Tree-sitter-based implementation of SourceUtils.
    """

    # pyre-ignore[11]: Annotation `Parser` is not defined as a type.
    _ts_parser: Optional[Parser] = None

    def __init__(self, cells: Optional[List[str]] = None) -> None:
        """
        Args:
            cells: List of folders in the root to search for source files. If
            None, only the root will be searched.
        """
        self.cells: List[str] = cells or [""]

    @classmethod
    def get_ts_parser(cls) -> Parser:
        if cls._ts_parser is not None:
            return cls._ts_parser
        try:
            # pyre-fixme[16]: Module `tree_sitter` has no attribute `Parser`.
            ts_parser: Parser = Parser()
            # The language library is a remote resource managed by BUCK
            ts_parser.set_language(
                Language(str(Path(__file__).parent / "tree-sitter-languages.so"), "cpp")
            )
            cls._ts_parser = ts_parser
            return cls._ts_parser
        except Exception:
            # pyre-ignore[21]: Could not find a module corresponding to import `tree_sitter_cpp`.
            import tree_sitter_cpp as tscpp  # @manual

            # pyre-fixme[16]: Module `tree_sitter` has no attribute `Parser`.
            # pyre-ignore[20]: Call `Language.__init__` expects argument `name`.
            cls._ts_parser = Parser(Language(tscpp.language()))
            return cls._ts_parser

    async def retrieve_src_of_func(self, path_and_line: PathAndLine) -> Optional[str]:
        """Retrieves the source code of the function encompassing
        `path_and_line` from a stacktrace.

        Args:
            path_and_line: A path and line from a stacktrace.

        Returns:
            Source code of the function. None if the function is not found.

        Raises:
            RuntimeError: If the function is not found.
        """
        repo_root = Path(self._get_repo_root())
        path = Path("")
        for cell in self.cells:
            path = repo_root / cell / path_and_line.file_path
            if await self.does_path_exist(path):
                break
        if not (await self.does_path_exist(path)):
            return None

        return await self._retrieve_src_of_func(path, path_and_line.line_number)

    @classmethod
    async def _retrieve_src_of_func(cls, path: Path, line_nbr: int) -> Optional[str]:
        file_content = await cls.get_file_content(path)
        function_nodes = cls.get_function_nodes_cpp(file_content)
        for node in function_nodes:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            if start_line <= line_nbr <= end_line:
                # pyre-fixme[16]: `int` has no attribute `decode`.
                return file_content[node.start_byte : node.end_byte].decode()
        return None

    @staticmethod
    async def get_file_content(path: Path) -> bytes:
        with open(path, "rb") as fp:
            return fp.read()

    async def does_path_exist(self, path: Path) -> bool:
        return path.exists()

    @classmethod
    def get_function_nodes_cpp(cls, file_content: bytes) -> List[TreeSitterNode]:
        """
        Returns a list of (templated and untemplated) function TreeSitterNodes in
        the given C++ source code.
        """
        root_node = cls.get_ts_parser().parse(file_content).root_node
        collected_nodes: List[TreeSitterNode] = []

        ts_traverse(
            root_node, collected_nodes, fn=TreeSitterSourceUtils.collect_functions
        )
        return collected_nodes

    @staticmethod
    def collect_functions(
        node: TreeSitterNode, collected_nodes: List[TreeSitterNode]
    ) -> bool:
        # Standard function.
        if node.type == "function_definition":
            collected_nodes.append(node)
            return False

        # Templated function.
        if node.type == "template_declaration":
            for child in node.children:
                if child.type == "function_definition":
                    collected_nodes.append(node)
                    return False

        return True


def parse_function_name(
    function_source: str,
    # pyre-ignore[24]: Generic type `LoggerAdapter` expects 1 type parameter
    logger: Optional[LoggerAdapter] = None,
) -> Optional[str]:
    """Parsers the source code of a function and returns the function name.
    Returns None upon parsing failure.
    """
    buf = function_source.encode()
    root_node = TreeSitterSourceUtils.get_ts_parser().parse(buf).root_node
    collected_nodes: List[TreeSitterNode] = []

    def collect_function_names(
        node: TreeSitterNode, collected_nodes: List[TreeSitterNode]
    ) -> bool:
        if node.type == "function_declarator":
            collected_nodes.append(node.children[0])
            return False

        return True

    ts_traverse(root_node, collected_nodes, fn=collect_function_names)

    if len(collected_nodes) == 0:
        return None

    if len(collected_nodes) > 1 and logger:
        # warn and heusistically choose the first function name
        logger.warn(
            f"Multiple function names found in {function_source}: {collected_nodes}"
        )

    node = collected_nodes[0]
    # pyre-fixme[16]: `int` has no attribute `decode`.
    return buf[node.start_byte : node.end_byte].decode()


class Stopwatch:
    """A stop watch that measures time elapsed within a context manager"""

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._elapsed_time: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        """Start a new timer"""
        self.start()
        return self

    def start(self) -> "Stopwatch":
        """Start a new timer"""
        if self._elapsed_time:
            raise RuntimeError("Cannot re-start a finished timer")
        self._start_time = time.monotonic()
        return self

    def stop(self) -> int:
        """Stop the timer"""
        if self._elapsed_time:
            raise RuntimeError("The timer has already stopped")
        end_time = time.monotonic()
        self._elapsed_time = (end_time - self._start_time) if self._start_time else 0
        return self.elapsed_seconds()

    def elapsed_seconds(self) -> int:
        """Return the elapsed time in seconds. Return 0 if the timer did not run."""
        return round(self._elapsed_time or 0.0)

    def __exit__(
        self, exc_type: Type[BaseException], exc_value: BaseException, tb: TracebackType
    ) -> None:
        """Stop the timer"""
        self.stop()
