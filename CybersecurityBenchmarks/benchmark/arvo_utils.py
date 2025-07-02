# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import json
import logging
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor

from enum import auto, Enum
from functools import lru_cache
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple

import boto3
import pkg_resources

from async_lru import alru_cache
from botocore import UNSIGNED
from botocore.config import Config


class BuildStatus(Enum):
    """Tracks the compilation status of the container."""

    UNBUILT = auto()  # Initial state, not yet compiled
    SUCCESSFUL = auto()  # Compilation completed successfully
    FAILED = auto()  # Compilation failed


def dumb_log(path: Optional[Path], msg: str) -> None:
    if path:
        with open(path, "a") as f:
            f.write(msg + "\n")


def get_least_common_ancestor(
    trace: List[str], other: List[List[str]]
) -> Tuple[Optional[str], int]:
    for i in range(len(trace)):
        filtered_others = [
            other_trace
            for other_trace in other
            if len(other_trace) > i and other_trace[i] == trace[i]
        ]
        if len(filtered_others) == 0:
            if i > 0:
                return trace[i - 1], len(trace) - i
            return None, len(trace)
    return trace[-1], 0


def filter_frame(frame: str) -> str:
    # remove all arguments after ( from this frame's function name
    frame = frame.split("(")[0]
    frame = frame.split("::")[-1]
    return frame


def filter_traces(traces: List[List[str]]) -> List[List[str]]:
    res = []
    for trace in traces:
        cur = []
        for frame in trace:
            cur.append(filter_frame(frame))
        res.append(cur)
    return res


def get_all_least_common_ancestors(
    ground_truth_stacktraces: List[List[str]],
    llm_stracktraces: List[List[str]],
    diff_log: Optional[Path] = None,
) -> List[str]:
    res = set()
    sum_losses = 0
    sum_len = 0
    ground_truth_stacktraces = filter_traces(ground_truth_stacktraces)
    llm_stracktraces = filter_traces(llm_stracktraces)
    assert len(ground_truth_stacktraces) != 0
    assert len(llm_stracktraces) != 0
    for trace in ground_truth_stacktraces:
        if len(trace) == 0:
            continue
        lca, lost_frames = get_least_common_ancestor(trace, llm_stracktraces)
        sum_losses += lost_frames
        sum_len += len(trace)
        if lca:
            res.add(lca)

    dumb_log(
        diff_log,
        f"FOUND LCAs with avg {sum_losses/len(ground_truth_stacktraces)} lost frames out of {sum_len / len(ground_truth_stacktraces)}: {repr(res)}",
    )
    return list(res)


class ArvoContainer:
    # ============================
    #       Class Constants
    # ============================
    DEFAULT_CONTAINER_REPOSITORY: str = "localhost/autopatch"

    CONTAINER_ID_NOT_INITIALIZED: str = "not_initialized"

    CONTAINER_TYPE_VUL: str = "vul"
    CONTAINER_TYPE_FIX: str = "fix"

    CONTAINER_CREATION_MAX_RETRIES: int = 12
    CONTAINER_CREATION_INITIAL_RETRY_DELAY: int = 1  # In seconds
    CONTAINER_CREATION_RETRY_BACKOFF_RATE: int = 2

    DOCKERFILE_VUL_TEMPLATE_FILENAME: str = "dockerfile_vul_template"
    DOCKERFILE_FIX_TEMPLATE_FILENAME: str = "dockerfile_fix_template"

    DPKG_U2004_FILENAME: str = "differential-debugging-deps-20.04.deb"
    DPKG_U1604_FILENAME: str = "differential-debugging-deps-16.04.deb"

    RUN_CMD_ARGS = [
        "podman",  # Use podman instead of docker to allow for use on devservers
        "run",
        "--platform=linux/amd64",
        "--net=none",  # running on devserver, this this seems to be necessary to avoid Error: pasta failed with exit code 1: Couldn't set IPv4 route(s) in guest: Invalid argument.
        "-d",
    ]

    OPT_FLAG_PATTERN = r"-(O0|O1|O2|O3|Os|Ofast|Og)(?=\s|$)"
    DEBUG_FLAG_PATTERN = r"-(g|g0|g1|g2|g3)(?=\s|$)"

    OPT_FLAG_FOR_BINARIES = "-O0"
    DEBUG_FLAG_FOR_BINARIES = "-g2"

    FULL_FUZZING_DURATION: int = 600

    CONTAINER_FILEPATH_TO_FUZZING_CORPUS = "/tmp/corpus"

    # ============================
    #       Class Variables
    # ============================
    # Class variable to track all container ids
    _container_ids: MutableMapping[tuple[int, str, str], str] = {}

    # Class variable to track build status of all containers
    # The `_container_build_status` attribute tracks the compilation status of each container.
    # It allows us to avoid unnecessary compilations by skipping re-compilations if the source code has not
    # been modified since last compilation. It should be updated to reflect the current build state.
    _container_build_status: MutableMapping[str, BuildStatus] = {}

    _container_repository: Optional[str] = None

    def __init__(
        self,
        arvo_id: int,
        container_type: str,
        model_under_test: str,
        log_level: int,
        log_filepath: Path,
        artifacts_dir: Path,
    ) -> None:
        self.arvo_id = arvo_id
        self.container_type = container_type
        self.model_under_test = model_under_test
        self.container_id: str = ArvoContainer.CONTAINER_ID_NOT_INITIALIZED
        self._log_level = log_level
        self._log_filepath = log_filepath
        self._artifacts_dir = artifacts_dir
        self.logger: logging.LoggerAdapter[logging.Logger] = self._get_logger(
            self.arvo_id,
            self.container_type,
            self.container_id,
            self._log_level,
            self._log_filepath,
        )

    @classmethod
    def set_container_repository(cls, container_repository: Optional[str]) -> None:
        container_repository = container_repository or cls.DEFAULT_CONTAINER_REPOSITORY
        logging.debug(f"Setting container repository to {container_repository}.")
        cls._container_repository = container_repository

    @classmethod
    def _get_container_repository(cls) -> str:
        container_repository = cls._container_repository
        assert (
            container_repository is not None
        ), "Container repository must be set before use."
        return container_repository

    @classmethod
    def build_container_images(
        cls,
        test_cases: list[int],
        max_concurrency: int,
        log_level: int,
    ) -> list[int]:  # return the list of arvo_ids that have the usable container images
        build_logger = logging.getLogger("container_build")
        build_logger.setLevel(log_level)

        container_repository = cls._get_container_repository()
        if ".thefacebook.com/" in container_repository:
            """
            Skip container image build if Facebook repository is specified.
            These images should already exist in the repository.
            """
            build_logger.debug(
                f"Skipping container image build for repository {container_repository}."
            )
            return test_cases

        # Otherwise, build images that are missing from the container repository
        # to build images, we need to check/download required dpkgs first
        if not cls._get_required_dpkgs():
            RuntimeError("Failed to download required dpkgs. Exiting.")
        build_logger.info("Building container images. This may take a while...")
        existing_arvo_ids = cls._get_arvo_ids_that_exist_in_container_repository(
            container_repository
        )
        arvo_ids_to_evaluate = set(test_cases)
        arvo_ids_to_build = arvo_ids_to_evaluate - existing_arvo_ids
        build_logger.debug(
            f"Images for the following ARVO IDs already exist: {existing_arvo_ids}."
        )
        build_logger.debug(
            f"Images for the following ARVO IDs will be built: {arvo_ids_to_build}."
        )

        # Set max_workers to half of max_concurrency because each thread will fork again to build 2 container images in parallel
        images_available = list(arvo_ids_to_evaluate.intersection(existing_arvo_ids))
        with ThreadPoolExecutor(max_workers=max_concurrency // 2) as executor:
            futures = [
                executor.submit(
                    cls._build_container_images_for_arvo_id,
                    arvo_id,
                    container_repository,
                )
                for arvo_id in arvo_ids_to_build
            ]
            for future in futures:
                result = future.result()
                if result >= 0:
                    images_available.append(result)

        return images_available

    @classmethod
    def _get_s3_object(cls, bucket_name: str, key: str, local_file_path: Path) -> bool:
        build_logger = logging.getLogger("container_build")
        try:
            # Create an S3 client with unsigned requests
            s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            # Specify the bucket and object key
            bucket_name = "purplellama-cyberseceval"
            object_key = f"cyberseceval3/autopatch_bench/{key}"
            # Download the S3 object
            s3_client.download_file(bucket_name, object_key, local_file_path)
            build_logger.debug(
                f"Successfully downloaded {object_key} to {local_file_path}"
            )
            return True
        except Exception as e:
            build_logger.error(
                f"Failed to download {key} from S3 bucket {bucket_name}: {e}"
            )
            return False

    @classmethod
    def _get_required_dpkgs(cls) -> bool:
        build_logger = logging.getLogger("container_build")
        build_dir = cls._get_build_dir()
        u2004_dpkg = build_dir / cls.DPKG_U2004_FILENAME
        u1604_dpkg = build_dir / cls.DPKG_U1604_FILENAME

        if not u2004_dpkg.exists() or not u1604_dpkg.exists():
            build_logger.debug(
                f"Required dpkgs {cls.DPKG_U2004_FILENAME} and {cls.DPKG_U1604_FILENAME} are missing. Downloading them from S3."
            )
            if not cls._get_s3_object(
                "purplellama-cyberseceval",
                cls.DPKG_U2004_FILENAME,
                u2004_dpkg,
            ) or not cls._get_s3_object(
                "purplellama-cyberseceval",
                cls.DPKG_U1604_FILENAME,
                u1604_dpkg,
            ):
                return False

        return True

    @classmethod
    def _build_container_images_for_arvo_id(
        cls, arvo_id: int, container_repository: str
    ) -> int:  # return arvo_id if successful, -1 otherwise
        build_logger = logging.getLogger("container_build")
        build_logger.debug(f"Building container images for ARVO ID: {arvo_id}.")
        # Simultaneously build both images via multithreading
        with ThreadPoolExecutor() as executor:
            future_vul = executor.submit(
                cls._build_image_from_dockerfile,
                container_repository,
                arvo_id,
                cls.CONTAINER_TYPE_VUL,
            )
            future_fix = executor.submit(
                cls._build_image_from_dockerfile,
                container_repository,
                arvo_id,
                cls.CONTAINER_TYPE_FIX,
            )
            image_tag_vul, build_result_vul = future_vul.result()
            image_tag_fix, build_result_fix = future_fix.result()

        # Check if either build failed
        if build_result_vul.returncode != 0 or build_result_fix.returncode != 0:
            build_logger.error("One or both image builds failed. Cleaning up images.")
            # Delete both images
            for image_tag in [image_tag_vul, image_tag_fix]:
                try:
                    subprocess.run(
                        ["podman", "rmi", image_tag],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    build_logger.debug(f"Deleted image {image_tag}.")
                except subprocess.CalledProcessError as e:
                    build_logger.error(f"Failed to delete image {image_tag}: {e}")
            return -1
        else:
            build_logger.debug(f"Image builds for ARVO ID {arvo_id} succeeded.")
            return arvo_id

    @classmethod
    def _build_image_from_dockerfile(
        cls, container_repository: str, arvo_id: int, container_type: str
    ) -> Tuple[str, CompletedProcess[str]]:
        build_logger = logging.getLogger("container_build")
        dockerfile_template = cls._get_dockerfile_template(container_type)
        dockerfile_content = dockerfile_template.replace("{ARVO_ID}", str(arvo_id))
        # Create temporary files for the Dockerfile
        with tempfile.NamedTemporaryFile(mode="w") as dockerfile:
            dockerfile.write(dockerfile_content)
            dockerfile.flush()

            image_tag = f"{container_repository}:{arvo_id}-{container_type}"
            build_cmd_args = [
                "podman",
                "build",
                "-t",
                image_tag,
                "-f",
                dockerfile.name,
                "--platform",
                "linux/amd64",
                str(cls._get_build_dir()),
            ]
            # Run the build command synchronously using subprocess
            result = subprocess.run(
                build_cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stdout and stderr
                text=True,
            )
            if result.returncode != 0:
                build_command = " ".join(build_cmd_args)
                build_logger.error(
                    f"Build command {build_command} failed with return code {result.returncode}. Command output:\n{result.stdout}"
                )
            else:
                build_logger.debug(f"Image {image_tag} built successfully.")

            return image_tag, result

    @classmethod
    @lru_cache()
    def _get_dockerfile_template(cls, container_type: str) -> str:
        filename = (
            cls.DOCKERFILE_VUL_TEMPLATE_FILENAME
            if container_type == ArvoContainer.CONTAINER_TYPE_VUL
            else cls.DOCKERFILE_FIX_TEMPLATE_FILENAME
        )
        template_filepath = cls._get_build_dir() / filename
        with open(template_filepath, "r") as file:
            return file.read()

    @classmethod
    def _get_arvo_ids_that_exist_in_container_repository(
        cls,
        container_repository: str,
    ) -> Set[int]:
        if "localhost" in container_repository:
            return cls._get_arvo_ids_that_exist_in_local_repository(
                container_repository
            )
        return cls._get_arvo_ids_that_exist_in_remote_repository(container_repository)

    @classmethod
    def _get_arvo_ids_that_exist_in_local_repository(
        cls,
        container_repository: str,
    ) -> Set[int]:
        result = subprocess.run(
            ["podman", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            logging.error("Failed to list images using podman.")
            return set()

        images_output = result.stdout
        tags = [
            line.strip()
            for line in images_output.splitlines()
            if line.startswith(f"{container_repository}:")
        ]
        paired_numbers = set()

        # Extract ARVO IDs with both vul and fix tags
        vul_tags = {
            int(tag.split(":")[1].split("-")[0])
            for tag in tags
            if tag.endswith(cls.CONTAINER_TYPE_VUL)
        }
        fix_tags = {
            int(tag.split(":")[1].split("-")[0])
            for tag in tags
            if tag.endswith(cls.CONTAINER_TYPE_FIX)
        }
        paired_numbers = vul_tags.intersection(fix_tags)
        return paired_numbers

    @classmethod
    def _get_arvo_ids_that_exist_in_remote_repository(
        cls,
        container_repository: str,
    ) -> Set[int]:
        # Get existing images using `podman search --list-tags <container_repository>``
        result = subprocess.run(
            ["podman", "search", "--list-tags", container_repository],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            logging.error(f"Failed to list tags for repository {container_repository}.")
            return set()

        tags_output = result.stdout
        tags = [line.split(" ")[-1] for line in tags_output.splitlines()]
        paired_numbers = set()

        # Extract ARVO IDs with both vul and fix tags
        vul_tags = {
            int(tag.split("-")[0])
            for tag in tags
            if tag.endswith(cls.CONTAINER_TYPE_VUL)
        }
        fix_tags = {
            int(tag.split("-")[0])
            for tag in tags
            if tag.endswith(cls.CONTAINER_TYPE_FIX)
        }
        paired_numbers = vul_tags.intersection(fix_tags)
        return paired_numbers

    async def start_container(
        self,
    ) -> None:
        # Grab a container from the pool if a suitable one exists
        self.logger.debug(
            f"Attempting to start a container for {self.arvo_id}-{self.container_type} to test {self.model_under_test}."
        )
        self.container_id = await ArvoContainer.get_or_create_container_id(
            self.arvo_id, self.container_type, self.model_under_test
        )
        if self.container_type == ArvoContainer.CONTAINER_TYPE_VUL:
            await self._ensure_git()
        # Update the logger with the new container ID
        self.logger = self._get_logger(
            self.arvo_id,
            self.container_type,
            self.container_id,
            self._log_level,
            self._log_filepath,
        )

    async def _ensure_git(self) -> None:
        """Ensure git is initialized in the container."""
        self.logger.debug("Checking if git is initialized in the container...")
        result = await self.exec_command(
            cmd_args=["git", "rev-parse", "--is-inside-work-tree"],
            combine_outputs=True,
        )

        if result.returncode != 0:
            self.logger.info("Git repository not initialized. Initializing git...")
            # remove any existing .git directories because `git add` may fail if
            # there are multiple .git directories
            await self.exec_command(
                cmd_args=[
                    "find",
                    ".",
                    "-type",
                    "d",
                    "-name",
                    ".git",
                    "-exec",
                    "rm",
                    "-rf",
                    "{}",
                    "\\;",
                ]
            )
            # Initialize git repository
            init_result = await self.exec_command(cmd_args=["git", "init"])
            if init_result.returncode != 0:
                self.logger.error(
                    f"Failed to initialize git repository: {init_result.stderr.decode()}"
                )
                raise Exception("Failed to initialize git repository")
            self.logger.debug("Git repository successfully initialized.")
        else:
            self.logger.debug("Git repository already initialized.")

    async def ensure_tracked_by_git(self, filepath: str) -> None:
        result = await self.exec_command(
            cmd_args=["git", "ls-files", "--error-unmatch", filepath]
        )
        if result.returncode != 0:
            await self.exec_command(cmd_args=["git", "add", filepath])

    async def restart_container(self) -> None:
        """Stop, remove, and start a new container fresh."""
        self.logger.info("Restarting container to clear state.")
        await self.stop_container(self.container_id)
        await self.remove_container()
        await self.start_container()

    @classmethod
    async def get_or_create_container_id(
        cls, arvo_id: int, container_type: str, model_under_test: str
    ) -> str:
        """Get an existing container instance or create a new one if it doesn't exist."""
        assert container_type in {
            ArvoContainer.CONTAINER_TYPE_VUL,
            ArvoContainer.CONTAINER_TYPE_FIX,
        }, "container_type must be vul or fix"

        key = (arvo_id, container_type, model_under_test)
        logging.debug(f"Searching for existing container ID with key {key}.")
        if key in cls._container_ids:
            logging.debug(
                f"Found existing container ID {cls._container_ids[key]} for {arvo_id}-{container_type} to test {model_under_test}."
            )
            return cls._container_ids[key]

        logging.debug(
            f"Creating new container for {arvo_id}-{container_type} to test {model_under_test}."
        )
        cmd_to_run = cls.RUN_CMD_ARGS.copy()
        image_name = cls.get_image_name(
            ArvoContainer._get_container_repository(), arvo_id, container_type
        )
        cmd_to_run.extend([image_name])

        container_id = ArvoContainer.CONTAINER_ID_NOT_INITIALIZED
        # Retry with exponential backoff, up to approximately an hour
        retries = 0
        delay = ArvoContainer.CONTAINER_CREATION_INITIAL_RETRY_DELAY
        while retries < ArvoContainer.CONTAINER_CREATION_MAX_RETRIES:
            result = await asyncio.create_subprocess_exec(
                *cmd_to_run,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()
            stdout_decoded = stdout.decode("utf-8").strip()
            container_id = stdout_decoded

            # Health check
            if (
                container_id != ArvoContainer.CONTAINER_ID_NOT_INITIALIZED
                and result.returncode == 0
            ):
                break

            stderr_decoded = stderr.decode("utf-8")
            logging.error(f"Error running container image {image_name}.")
            logging.error(f"stdout: {stdout_decoded}")
            logging.error(f"stderr: {stderr_decoded}")
            if "not found" in stderr_decoded:
                return ArvoContainer.CONTAINER_ID_NOT_INITIALIZED

            retries += 1
            logging.debug(
                f"Retrying... ({retries}/{ArvoContainer.CONTAINER_CREATION_MAX_RETRIES}) with delay {delay}s"
            )
            await asyncio.sleep(delay)
            delay *= (
                ArvoContainer.CONTAINER_CREATION_RETRY_BACKOFF_RATE
            )  # Exponential backoff

        if retries == ArvoContainer.CONTAINER_CREATION_MAX_RETRIES:
            return ArvoContainer.CONTAINER_ID_NOT_INITIALIZED

        logging.debug(
            f"Container created with ID {container_id} for {arvo_id}-{container_type} to test {model_under_test}."
        )
        cls._container_ids[key] = container_id
        return cls._container_ids[key]

    @classmethod
    async def cleanup_all_containers(cls) -> None:
        """Clean up all outstanding container instances."""
        logging.debug("Starting cleanup of all containers.")
        for container_id in list(cls._container_ids.values()):
            try:
                if container_id != ArvoContainer.CONTAINER_ID_NOT_INITIALIZED:
                    await cls.stop_container(container_id)
            except Exception as e:
                # Log error but continue with other containers
                logging.error(f"Error cleaning up container {container_id}: {str(e)}")
        cls._container_ids.clear()
        logging.debug("Cleanup of all containers completed.")

    @classmethod
    async def stop_container(cls, container_id: str) -> None:
        """Stop the container."""
        logging.debug(f"Stopping container with ID: {container_id}.")
        stop_cmd_args = [
            "podman",
            "stop",
            container_id,
        ]
        result = await asyncio.create_subprocess_exec(
            *stop_cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()

    async def remove_container(self) -> None:
        """Remove the container."""
        logging.debug(f"Removing container with ID: {self.container_id}.")
        stop_cmd_args = [
            "podman",
            "rm",
            self.container_id,
        ]
        result = await asyncio.create_subprocess_exec(
            *stop_cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()

        key = (self.arvo_id, self.container_type, self.model_under_test)
        self._container_ids.pop(key)

    async def check_container_status(self) -> str:
        """Check status of the container."""
        self.logger.debug(f"Checking status of container with ID: {self.container_id}.")
        result = await asyncio.create_subprocess_exec(
            "podman",
            "container",
            "inspect",
            "-f",
            "'{{.State.Status}}'",
            self.container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        status = stdout.decode("utf-8").strip()
        self.logger.debug(f"Container status: {status} for ID: {self.container_id}.")
        return status

    @staticmethod
    def get_image_name(
        container_repository: str, arvo_id: int, container_type: str
    ) -> str:
        image_name = f"{container_repository}:{arvo_id}-{container_type}"
        logging.debug(
            f"Generated image name: {image_name} for {arvo_id}-{container_type}."
        )
        return image_name

    async def get_current_commit_hash(self) -> str:
        result = await self.exec_command(
            cmd_args=[
                'sh -c \'cd "$(dirname "$(find . -type d -name .git -print -quit)")" && git rev-parse HEAD\''
            ]
        )
        return result.stdout.decode().strip()

    async def get_git_diff(self, commit_hash: str) -> str:
        result = await self.exec_command(
            cmd_args=[
                "git",
                "diff",
                commit_hash + "^",
                commit_hash,
            ],
            workdir=await self.get_git_dir(),
        )
        return result.stdout.decode()

    async def get_git_summary(self, commit_hash: str) -> str:
        result = await self.exec_command(
            cmd_args=["git", "show", commit_hash, "--no-patch"],
            workdir=await self.get_git_dir(),
        )
        return result.stdout.decode()

    async def get_commit_timestamp(self, commit_hash: str) -> str:
        result = await self.exec_command(
            cmd_args=[
                "git",
                "show",
                "-s",
                "--format=%ci",
                commit_hash,
            ],
            workdir=await self.get_git_dir(),
        )
        return result.stdout.decode().strip()

    async def get_git_dir(self) -> Optional[str]:
        result = await self.exec_command(
            cmd_args=[
                'sh -c \'realpath "$(dirname "$(find . -type d -name .git -print -quit)")"\''
            ]
            # dirname "$(find . -type d -name .git -print -quit)"
        )
        workdir = result.stdout.decode().strip()
        if workdir == ".":
            workdir = None
        elif workdir.startswith("./"):
            workdir = workdir[2:]
        return workdir

    async def git_bisect_groundtruth_fix(
        self, start_commit: str, end_commit: str
    ) -> Optional[str]:
        """Run git bisect to find the commit that introduced a bug."""
        self.logger.debug(f"Running git bisect from {start_commit} to {end_commit}")
        workdir = await self.get_git_dir()
        if workdir == ".":
            workdir = None
        elif workdir and workdir.startswith("./"):
            workdir = workdir[2:]

        # Change directory
        result = await self.exec_command(
            cmd_args=[
                f'sh -c \'cd "$(dirname "$(find . -type d -name .git -print -quit)")"\
                && git checkout {start_commit}\''
            ],
            workdir=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout {start_commit}: {result.stdout.decode()} {result.stderr.decode()}"
            )

        # start from one commit before start_commit
        result = await self.exec_command(
            cmd_args=[
                "git",
                "rev-parse",
                f"{start_commit}^",
            ],
            workdir=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to get commit hash for {start_commit}: {result.stdout.decode()} {result.stderr.decode()}"
            )
        parent_start_commit = result.stdout.decode().strip()

        result = await self.exec_command(
            cmd_args=["git", "bisect", "start"], workdir=workdir
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start git bisect: {result.stdout.decode()} {result.stderr.decode()}"
            )
        # good: arvo fails
        result = await self.exec_command(
            cmd_args=["git", "bisect", "good", parent_start_commit],
            workdir=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to mark {parent_start_commit} as good: {result.stdout.decode()} {result.stderr.decode()}"
            )
        # bad: arvo passes
        result = await self.exec_command(
            cmd_args=["git", "bisect", "bad", end_commit], workdir=workdir
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to mark {end_commit} as bad: {result.stdout.decode()} {result.stderr.decode()}"
            )
        # bisect: find last arvo fail
        result = await self.exec_command(
            cmd_args=["git", "bisect", "run", "bash", "-c", "! arvo"],
            workdir=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to run git bisect: {result.stdout.decode()} {result.stderr.decode()}"
            )

        self.logger.debug(
            f"Git bisect result: {result.stdout.decode()}, {result.stderr.decode()}"
        )

        # Get the first bad commit
        bisect_log = await self.exec_command(
            cmd_args=["git", "bisect", "log"], workdir=workdir
        )
        # Reset bisect state
        await self.exec_command(cmd_args=["git", "bisect", "reset"], workdir=workdir)
        # Extract and return the first bad commit hash
        log_output = bisect_log.stdout.decode("utf-8")
        match = re.search(r"first bad commit: \[([a-f0-9]+)\]", log_output)

        if match:
            return match.group(1)

    async def exec_command(
        self,
        extra_args: Optional[List[str]] = None,
        cmd_args: Optional[List[str]] = None,
        combine_outputs: bool = False,
        env_vars: Optional[MutableMapping[str, str]] = None,
        workdir: Optional[str] = None,
    ) -> CompletedProcess[bytes]:
        exec_cmd_args = [
            "podman",  # Use podman instead of docker to allow for use on devservers
            "exec",
        ]
        if workdir:
            exec_cmd_args.extend(["-w", workdir])
        if env_vars is not None:
            for key, value in env_vars.items():
                exec_cmd_args.extend(["-e", f"{key}='{value}'"])
        if extra_args is not None:
            exec_cmd_args.extend(extra_args)
        exec_cmd_args.extend([self.container_id])
        if cmd_args is not None:
            exec_cmd_args.extend(cmd_args)

        cmd_str = " ".join(exec_cmd_args)
        self.logger.debug(f"Executing command: {cmd_str}.")
        stderr = subprocess.STDOUT if combine_outputs else subprocess.PIPE
        result = await asyncio.create_subprocess_shell(
            cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
        )
        stdout, stderr = await result.communicate()
        returncode = result.returncode if result.returncode is not None else -1
        self.logger.debug(
            f"Command `{cmd_str}` executed with return code: {returncode}."
        )
        return CompletedProcess(exec_cmd_args, returncode, stdout, stderr)

    def exec_command_blocking(
        self,
        extra_args: Optional[List[str]] = None,
        cmd_args: Optional[List[str]] = None,
        combine_outputs: bool = False,
    ) -> CompletedProcess[bytes]:
        exec_cmd_args = [
            "podman",  # Use podman instead of docker to allow for use on devservers
            "exec",
        ]
        if extra_args is not None:
            exec_cmd_args.extend(extra_args)
        exec_cmd_args.extend([self.container_id])
        if cmd_args is not None:
            exec_cmd_args.extend(cmd_args)

        cmd_str = " ".join(exec_cmd_args)
        self.logger.debug(f"Executing command: {cmd_str}.")
        stderr = subprocess.STDOUT if combine_outputs else subprocess.PIPE
        result = subprocess.run(
            exec_cmd_args,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        self.logger.debug(
            f"Command `{cmd_str}` executed with return code: {result.returncode}."
        )
        return result

    async def copy_to_container(
        self, filepath_on_host: str, filepath_in_container: str
    ) -> None:
        """Copy the contents of filepath_on_host to filepath_in_container in the container. If local_filepath specifies a file and dest_path ends with /, the directory must already exist on the container, and the local file will be copied there.
        If local_filpath is a directory, dest_path will be created if needed and the contents of local_filepath copied there.  For detailed documentation, see https://docs.docker.com/reference/cli/docker/container/cp/
        """
        self.logger.debug(
            f"Copying {filepath_on_host} on host to {filepath_in_container} in container."
        )
        cmd_args = [
            "podman",
            "cp",
            filepath_on_host,
            f"{self.container_id}:{filepath_in_container}",
        ]
        cmd_str = " ".join(cmd_args)
        result = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await result.communicate()
        self.logger.debug(
            f"Command `{cmd_str}` executed with return code: {result.returncode}."
        )
        if result.returncode != 0:
            error_msg = f"Error copying file: {stdout.decode().strip()} using the command `{cmd_str}`."
            self.logger.error(error_msg)
            raise Exception(error_msg)

    async def copy_from_container(
        self, filepath_in_container: str, filepath_on_host: str
    ) -> None:
        """Copy the file at filepath_in_container in the container to filepath_on_host on the host."""
        self.logger.debug(
            f"Copying {filepath_in_container} from container to {filepath_on_host} on host."
        )
        result = await asyncio.create_subprocess_exec(
            "podman",
            "cp",
            f"{self.container_id}:{filepath_in_container}",
            filepath_on_host,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()

    async def submit_patch(self, local_filepath: str) -> bytes:
        """Submit a proposed patch to the container and apply with git apply."""
        self.logger.debug(f"Submitting patch from {local_filepath} to container.")
        await self.copy_to_container(local_filepath, "/tmp/patch")

        result = await self.exec_command(cmd_args=["git", "apply", "/tmp/patch"])
        self.logger.debug(f"Patch applied with return code: {result.returncode}.")
        return result.stdout

    async def recompile(self, combine_outputs: bool = False) -> CompletedProcess[bytes]:
        """Recompile the repo."""
        self.logger.debug("Recompiling...")
        result = await self.exec_command(
            cmd_args=["arvo", "compile"],
            combine_outputs=combine_outputs,
            env_vars={
                "CFLAGS": await self._get_updated_compiler_flags("CFLAGS"),
                "CXXFLAGS": await self._get_updated_compiler_flags("CXXFLAGS"),
            },
        )
        self.logger.debug(f"Recompile completed with return code: {result.returncode}.")
        return result

    @property
    def build_status(self) -> BuildStatus:
        """Get the build status of this container."""
        status = self._container_build_status.get(
            self.container_id, BuildStatus.UNBUILT
        )
        self.logger.debug(f"Build status for container: {status}.")
        return status

    @build_status.setter
    def build_status(self, status: BuildStatus) -> None:
        """Set the build status of this container."""
        self.logger.debug(f"Setting build status for container to {status}.")
        self._container_build_status[self.container_id] = status

    async def does_program_crash_in_full_fuzzing(self) -> bool:
        """Run the fuzzer for 60s.  Return a boolean indicating if the fuzzer found a crash."""
        self.logger.debug("Running full fuzzing over the patched binary.")
        # Update fuzzing time limit at evaluation time
        await self.string_replace_in_container_file(
            "/bin/arvo",
            "-max_total_time=60",
            f"-max_total_time={self.FULL_FUZZING_DURATION}",
        )
        # Make sure /bin/arvo is executable
        await self.exec_command(cmd_args=["chmod", "+x", "/bin/arvo"])
        # Run the fuzzer
        fuzzing_result = await self.exec_command(cmd_args=["arvo", "fuzz_fix"])
        crash_found = fuzzing_result.returncode != 0
        self.logger.debug(f"Full fuzzing completed. Crash found? {crash_found}.")
        if self._log_level == logging.DEBUG:
            await self.copy_from_container(
                self.CONTAINER_FILEPATH_TO_FUZZING_CORPUS, str(self._artifacts_dir)
            )
        return crash_found

    def _load_ground_truth_patches(self) -> List[Dict[str, Any]]:
        """Load the ground truth patch data from the arvo_meta directory."""
        gt_patch_path = pkg_resources.resource_filename(
            __name__, f"arvo_meta/{self.arvo_id}-patch.json"
        )
        with open(gt_patch_path, "r") as f:
            return json.load(f)

    def _load_arvo_metadata(self) -> Dict[str, Any]:
        """Load ARVO metadata"""
        gt_patch_path = pkg_resources.resource_filename(
            __name__, f"arvo_meta/{self.arvo_id}-meta.json"
        )
        with open(gt_patch_path, "r") as f:
            return json.load(f)

    def _get_ground_truth_fix_commit(self) -> str:
        """Get the commit hash of the ground truth fix."""
        metadata_dict = self._load_arvo_metadata()
        if "arvo_metadata" not in metadata_dict:
            raise RuntimeError(f"No arvo_metadata found in metadata for {self.arvo_id}")
        arvo_metadata = metadata_dict["arvo_metadata"]
        fix_commit = arvo_metadata.get("fix_commit", "")
        if not fix_commit:
            raise RuntimeError(f"No fix_commit found in metadata for {self.arvo_id}")
        return fix_commit

    async def get_stacktraces(
        self, name: str, binary: str, bps: List[str]
    ) -> List[List[str]]:
        bp_args = []
        for bp in bps:
            bp_args.extend(["--patched-line", bp])
        cmd_args = [
            "python3",
            "/microsnapshots/main.py",
            "get-stacktraces",
            "--binary",
            binary,
            *bp_args,
            "--input",
            self.CONTAINER_FILEPATH_TO_FUZZING_CORPUS,
            "--out",
            "/tmp/stacktraces.json",
        ]
        get_stacktraces = await self.exec_command(
            cmd_args=cmd_args,
            env_vars={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/out:/usr/local/lldb13/bin",
                "ASAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer in ASan-instrumented programs
                "LSAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer when using LSan standalone
            },
            combine_outputs=True,
        )
        if self._log_level == logging.DEBUG:
            self._write_to_file(
                get_stacktraces.stdout,
                self._artifacts_dir / f"get_stacktraces_combined_{name}.txt",
            )
            await self.copy_from_container(
                "/tmp/stacktraces.json",
                str(self._artifacts_dir / f"stacktraces_{name}.json"),
            )
        if get_stacktraces.returncode != 0:
            self.logger.error(
                f"stacktraces.returncode is {get_stacktraces.returncode}."
            )
            self.logger.error(f"stdout is {get_stacktraces.stdout}")
            self.logger.error(f"stderr is {get_stacktraces.stderr}")
            raise Exception(
                f"get_stacktraces failed with return code {get_stacktraces.returncode}"
            )
        with open(self._artifacts_dir / f"stacktraces_{name}.json") as stacktraces_file:
            return json.load(stacktraces_file)

    def get_non_contiguous_ints(self, int_list: List[int]) -> List[int]:
        return [
            int_list[i]
            for i in range(len(int_list))
            if i == 0 or int_list[i] - int_list[i - 1] > 1
        ]

    def get_groundtruth_bps(self) -> List[str]:
        res = []
        for p in self._load_ground_truth_patches():
            for line in self.get_non_contiguous_ints(p["patched_line_numbers"]):
                res.append(f"{Path(p['filename']).name}:{line}")
        self.logger.debug(f"ground truth breakpoints: {res}")
        return res

    async def run_differential_debugging(self, patched_func_name: str) -> bool:
        """Run differential debugging scripts.  Return a Tuple of booleans
        indicating if 1) the rebuilt binary appears to be of identical
        functionality to the correct patch and 2) if there was an error in
        attempting to run the debugging script (in which case the first boolean
        defaults to False)."""

        # Remove existing microsnapshots directory in the container
        await self.exec_command(cmd_args=["rm", "-rf", "/microsnapshots"])
        # Recreate the microsnapshots directory in the container
        await self.exec_command(cmd_args=["mkdir", "-p", "/microsnapshots"])
        # Copy files in the microsnapshots directory into the container, making sure to resolve any symlinks
        microsnapshots_dir = self._get_build_dir() / "microsnapshots"
        await asyncio.gather(
            *[
                self.copy_to_container(
                    str(file_path.resolve()),
                    f"/microsnapshots/{file_path.resolve().name}",
                )
                for file_path in microsnapshots_dir.iterdir()
                if file_path.suffix in {".py", ".json"}
            ]
        )
        try:
            gt_commit = self._get_ground_truth_fix_commit()
            self.logger.debug(f"Checking out ground truth fix commit {gt_commit}")
            # check out the fix commit
            await self.exec_command(
                cmd_args=["git", "checkout", self._get_ground_truth_fix_commit()],
                workdir=await self.get_git_dir(),
            )
        except Exception as e:
            self.logger.warning(f"Failed to get ground truth fix commit: {e}")

        # Recompile canonical binary to make sure it includes the necessary flags
        await self.recompile()
        functionality_preserved = False
        self.logger.debug("Obtaining stacktraces for differential debugging")

        try:
            gt_patches = [self.get_groundtruth_bps()[0]]

            ground_truth_stacktraces = await self.get_stacktraces(
                "ground_truth", await self.get_binary_filepath(), gt_patches
            )
            if len(ground_truth_stacktraces) == 0:
                self.logger.error("No ground truth stacktraces found")
                raise Exception("No ground truth stacktraces found")

            llm_stacktraces = await self.get_stacktraces(
                "llm", "/tmp/rebuilt_binary", [patched_func_name]
            )
            if len(llm_stacktraces) == 0:
                self.logger.error("No llm stacktraces found")
                raise Exception("No llm stacktraces found")

            lcas = get_all_least_common_ancestors(
                ground_truth_stacktraces,
                llm_stacktraces,
                self._artifacts_dir / "diff_log.txt",
            )
            self.logger.debug(f"done getting stacktraces, found LCAs {lcas}")
            if len(lcas) == 0:
                raise Exception(
                    "Unable to find least common ancestor function for the groundtruth and the llm patch"
                )
            if lcas == ["LLVMFuzzerTestOneInput"]:
                # if the least common ancestor is the entry point the llm patch and the ground truth patch are in completely different places
                raise Exception(
                    "Unable to find least common ancestor function for the groundtruth and the llm patch"
                )

        except Exception as e:
            self.logger.error(f"in stacktrace extraction: {e} {traceback.format_exc()}")
            raise e

        self.logger.debug(
            f"Obtaining noisy-config for differential debugging, hooking after patched function {patched_func_name}"
        )
        cmd_args = [
            "python3",
            "/microsnapshots/main.py",
            "get-noisy-config",
            "--binary",
            "/tmp/rebuilt_binary",
            "--hook-after",
            lcas[0],
            "--input",
            self.CONTAINER_FILEPATH_TO_FUZZING_CORPUS,
            "--out",
            "skipped_fields.json",
        ]
        get_noisy_config = await self.exec_command(
            cmd_args=cmd_args,
            env_vars={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/out:/usr/local/lldb13/bin",
                "ASAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer in ASan-instrumented programs
                "LSAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer when using LSan standalone
            },
            combine_outputs=True,
        )
        if self._log_level == logging.DEBUG:
            self._write_to_file(
                get_noisy_config.stdout,
                self._artifacts_dir / "get_noisy_config_stdout_stderr_combined.txt",
            )
            await self.copy_from_container(
                "skipped_fields.json", str(self._artifacts_dir / "skipped_fields.json")
            )
        if get_noisy_config.returncode != 0:
            self.logger.error(
                f"get_noisy_config.returncode is {get_noisy_config.returncode}."
            )
            self.logger.error(f"stdout is {get_noisy_config.stdout}")
            self.logger.error(f"stderr is {get_noisy_config.stderr}")
            raise Exception(
                f"get_noisy_config failed with return code {get_noisy_config.returncode}"
            )
        else:
            self.logger.debug(
                f"Running differential debugging to obtain deltas, hooking after patched function {patched_func_name}"
            )
            cmd_args = [
                "python3",
                "/microsnapshots/main.py",
                "get-delta",
                "--binary-correct",
                await self.get_binary_filepath(),
                "--binary-suggested",
                "/tmp/rebuilt_binary",
                "--hook-after",
                lcas[0],
                "--input",
                self.CONTAINER_FILEPATH_TO_FUZZING_CORPUS,
                "--skip-file",
                "skipped_fields.json",
                "--out",
                "/tmp/list_of_deltas.json",
            ]
            get_delta = await self.exec_command(
                cmd_args=cmd_args,
                env_vars={
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/out:/usr/local/lldb13/bin",
                    "ASAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer in ASan-instrumented programs
                    "LSAN_OPTIONS": "detect_leaks=0",  # Disable LeakSanitizer when using LSan standalone
                },
                combine_outputs=True,
            )
            if self._log_level == logging.DEBUG:
                self._write_to_file(
                    get_delta.stdout,
                    self._artifacts_dir / "get_delta_stdout_stderr_combined.txt",
                )
                await self.copy_from_container(
                    "/tmp/list_of_deltas.json",
                    str(self._artifacts_dir / "list_of_deltas.json"),
                )
            if get_delta.returncode != 0:
                self.logger.error(
                    f"get_delta.returncode is {str(get_delta.returncode)}."
                )
                self.logger.error(f"stdout is {get_delta.stdout}")
                self.logger.error(f"stderr is {get_delta.stderr}")
                raise Exception(
                    f"get_delta failed with return code {get_delta.returncode}"
                )
            else:
                # Check if the contents of list_of_deltas.json is an empty list, indicating there is no difference between the suggested and the correct patch.
                file = (await self.get_content("/tmp/list_of_deltas.json")).decode(
                    "utf-8"
                )
                data = json.loads(file)
                filtered_data = self.filter_deltas(data)
                functionality_preserved = len(filtered_data) == 0
                self.logger.debug(
                    f"Differential debugging completed. Functionality preserved? {functionality_preserved}."
                )

        return functionality_preserved

    def filter_deltas(self, deltas: List[str]) -> List[str]:
        # remove all deltas that match "build" or "time" since values like that are likely to change sporadically.
        filtered_deltas = []
        for line in deltas:
            if re.search(r"build|time", line, re.IGNORECASE) is None:
                filtered_deltas.append(line)
        return filtered_deltas

    async def does_program_crash(
        self,
    ) -> bool:
        self.logger.debug("Checking if crash reproduces...")
        fuzzing_result = await self.exec_command(cmd_args=["arvo"])
        crash_detected = fuzzing_result.returncode != 0
        self.logger.debug(f"Crash detected? {crash_detected}.")
        return crash_detected

    async def clean_up_image(self) -> None:
        """Clean up podman/docker image associated with this issue"""
        self.logger.debug(
            f"Cleaning up image for {self.arvo_id}-{self.container_type}."
        )
        result = await asyncio.create_subprocess_exec(
            "podman",
            "image",
            "rm",
            "-f",
            ArvoContainer.get_image_name(
                ArvoContainer._get_container_repository(),
                self.arvo_id,
                self.container_type,
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()
        self.logger.debug(
            f"Image cleanup completed for {self.arvo_id}-{self.container_type}."
        )

    async def qa_checks(self) -> bool:
        """Run QA checks on the container."""
        self.logger.debug("Running QA checks...")
        if self.container_id == ArvoContainer.CONTAINER_ID_NOT_INITIALIZED:
            self.logger.debug("Failed QA checks as container is not initialized.")
            return False
        crash_reproduces_expectedly = (
            await self.does_program_crash()
            if self.container_type == ArvoContainer.CONTAINER_TYPE_VUL
            else not await self.does_program_crash()
        )
        recompile_results = await self.recompile()
        compiles_successfully = recompile_results.returncode == 0
        qa_passed = crash_reproduces_expectedly and compiles_successfully
        self.logger.debug(f"QA checks completed. Passed? {qa_passed}.")
        return qa_passed

    async def string_replace_in_container_file(
        self, filepath: str, old_string: str, new_string: str
    ) -> None:
        """Replace all instances of old_string with new_string in the file at filepath in the container."""
        self.logger.debug(
            f"Replacing all instances of {old_string} with {new_string} in {filepath} in container."
        )
        old_content = (await self.get_content(filepath)).decode()
        new_content = old_content.replace(old_string, new_string)
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(new_content.encode())
            temp_file.flush()
            await self.copy_to_container(temp_file.name, filepath)

    @staticmethod
    def _is_crash(output: str) -> bool:
        """Check if the output from running 'arvo' contains a crash message."""
        crash_detected = bool(
            re.search(
                r"==[1-9]+==(ERROR|WARNING): (AddressSanitizer|MemorySanitizer)", output
            )
        )
        logging.debug(f"Crash detected in output? {crash_detected}.")
        return crash_detected
        # There might be more error messages this is missing.

    async def get_content(self, filepath: str) -> bytes:
        """Return the contents of the file at filepath in the container."""
        self.logger.debug(f"Getting content of file {filepath} in container...")
        result = await self.exec_command(cmd_args=["cat", filepath])
        return result.stdout

    async def get_binary_filepath(self) -> str:
        self.logger.debug("Getting filepath of binary in container...")
        result = await self.exec_command(cmd_args=["bash", "-c", "'cat $(which arvo)'"])
        script_contents = result.stdout.decode()
        pattern = r"(\S+)\s+/tmp/poc"
        filepath = re.findall(pattern, script_contents)[0]
        self.logger.debug(f"Binary filepath obtained: {filepath}.")
        return filepath

    @classmethod
    def _get_logger(
        cls,
        arvo_id: int,
        container_type: str,
        container_id: str,
        log_level: int,
        log_filepath: Path,
        # pyre-ignore[24]: Generic type `logging.LoggerAdapter` expects 1 type parameter.
    ) -> logging.LoggerAdapter:
        """Returns a logger with ARVO ID, container type, and container ID context"""
        logger = logging.getLogger(f"arvo_{arvo_id}_{container_type}")
        logger.setLevel(log_level)

        # Check if the handler is already added to avoid duplicate handlers
        if not any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == log_filepath.as_posix()
            for h in logger.handlers
        ):
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        return ArvoContainerContextAdapter(
            logger,
            {
                "container_id": container_id[:12],
            },
        )

    @staticmethod
    def _get_build_dir() -> Path:
        return Path(__file__).parent.resolve() / "autopatch" / "build"

    @alru_cache()
    async def _get_updated_compiler_flags(self, env_name: str) -> str:
        flags = (
            (await self.exec_command(cmd_args=["printenv", env_name]))
            .stdout.decode()
            .strip()
        )
        # Set optimization flag to `-O0``
        flags = self._update_flags(
            flags, self.OPT_FLAG_PATTERN, self.OPT_FLAG_FOR_BINARIES
        )
        # Set debug flag to `-g2`
        flags = self._update_flags(
            flags, self.DEBUG_FLAG_PATTERN, self.DEBUG_FLAG_FOR_BINARIES
        )
        return flags

    @staticmethod
    def _update_flags(
        original_flags: str, flags_pattern: str, replacement_flag: str
    ) -> str:
        updated_flags = re.sub(flags_pattern, replacement_flag, original_flags)
        if replacement_flag in updated_flags:
            return updated_flags
        # Flag doesn't exist in original flags, so add it
        return (
            replacement_flag
            if original_flags == ""
            else f"{original_flags} {replacement_flag}"
        )

    @staticmethod
    def _write_to_file(content: bytes, filepath: Path) -> None:
        """Write the content to a file in a specified path."""
        with open(filepath, "wb") as file:
            file.write(content)


# pyre-ignore[24]: Generic type `logging.LoggerAdapter` expects 1 type parameter.
class ArvoContainerContextAdapter(logging.LoggerAdapter):
    """Adapter that adds ARVO ID, container type, and container ID context to log messages"""

    # pyre-fixme[3]: Return type must be specified as type that does not contain `Any`.
    def process(
        self,
        msg: str,
        # pyre-fixme[2]: Parameter `kwargs` must have a type that does not contain `Any`.
        kwargs: MutableMapping[str, Any],
    ) -> tuple[str, MutableMapping[str, Any]]:
        if self.extra is None:
            return msg, kwargs

        container_id = self.extra.get("container_id")
        return f"[{container_id}] {msg}", kwargs
