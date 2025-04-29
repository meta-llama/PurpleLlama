# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import signal
from pathlib import Path
from types import FrameType
from typing import cast, Dict, List, Optional, Union
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from .benchmark import Benchmark, BenchmarkConfig
from .llm import LLM, RetriesExceededException
from .query_llm import query_llm_to_generate_responses
from .ssh import SshClient


class AutonomousUpliftAI:
    def __init__(
        self,
        llm: LLM,
        ssh_client: SshClient,
        system_prompt: str = "",
        shots_per_run: int = 100,
        run_id: str = "??",
    ) -> None:
        self.llm = llm
        self.ssh_client = ssh_client
        self.system_prompt = system_prompt
        self.shots_per_run = shots_per_run
        self.run_id = run_id
        self.messages_ctx: List[str] = []
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def update_message_context(self, prompt: str) -> None:
        self.messages_ctx.append(prompt)

    def prompt_llm(self, prompt: str) -> str:
        self.logger.debug(f"[Pre-LLM prompting] Prompt: {prompt}")

        # Add user prompt (previous command's output) to the context
        self.messages_ctx.append(prompt)
        self.logger.debug(f"[Pre-LLM prompting] Context: {self.messages_ctx}")

        # Prompt LLM with the context
        try:
            result = self.llm.chat_with_system_prompt_with_retries(
                system_prompt=self.system_prompt, prompt_with_history=self.messages_ctx
            )
        except RetriesExceededException as e:
            self.logger.error(e)
            result = "[!] Retries exceeded on LLM. No valid response."

        # Add AI's response to the context
        self.messages_ctx.append(result)
        self.logger.debug(f"[Post-LLM prompting] Context: {self.messages_ctx}")

        return result

    def execute_llm_command(self, prompt: str) -> str:
        llm_response = self.prompt_llm(prompt)
        self.logger.debug(f"LLM Response: {llm_response}")

        output = self.ssh_client.execute_command(llm_response)
        self.logger.debug(f"SSH Output: {output}")

        return output

    def run(self, initial_prompt: str = "") -> bool:
        self.logger.debug(
            f"Starting run {self.run_id} with initial prompt: {initial_prompt}"
        )

        cmd_output = self.execute_llm_command(initial_prompt)

        for shot in tqdm(
            range(1, self.shots_per_run + 1),
            desc=f"[Run {self.run_id}] Shots",
            unit="shot",
            nrows=40,
        ):
            self.logger.debug(
                f"Starting shot {shot}/{self.shots_per_run} in run {self.run_id}"
            )

            cmd_output = self.execute_llm_command(cmd_output)

        # Before finishing, add the final command output to the context
        self.update_message_context(cmd_output)

        self.logger.debug(f"Finished run {self.run_id}")

        return True

    def get_context(self) -> str:
        messages = []
        for i in range(len(self.messages_ctx)):
            if i % 2 == 0:
                messages.append(f">>> USER: {self.messages_ctx[i]}\n")
            else:
                messages.append(f">>> AI: {self.messages_ctx[i]}\n")

        return "".join(messages)

    def write_context(self, filepath: Optional[str] = None) -> None:
        if filepath is None:
            filepath = f"context_{self.ssh_client.hostname}_{self.llm.model.replace('/', '-')}_{self.run_id}.txt"
        assert filepath is not None
        with open(filepath, "w") as file:
            file.write(self.get_context())

        self.logger.debug(f"Context written to {filepath}")


def process_autonomous_run(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
) -> Dict[str, Union[int, str]]:
    config = {}
    for key in [
        "attacker",
        "target",
        "attacker_ssh_username",
        "attacker_ssh_key",
        "system_prompt",
        "initial_prompt",
    ]:
        config[key] = cast(str, test_case.get(key))
    for key in ["ssh_command_timeout", "shots", "id"]:
        config[key] = cast(int, test_case.get(key))

    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug(f"Running with config: {config}")
    else:
        logging.info(
            f"Running with config: { {k:v for k,v in config.items() if k not in ['system_prompt', 'attacker_ssh_key']} }"
        )

    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")

    with SshClient(
        hostname=config["attacker"],
        user=config["attacker_ssh_username"],
        private_key=config["attacker_ssh_key"],
        command_timeout=config["ssh_command_timeout"],
    ) as ssh_client:
        autonomous = AutonomousUpliftAI(
            llm=llm,
            ssh_client=ssh_client,
            shots_per_run=config["shots"],
            system_prompt=config["system_prompt"],
            run_id=f"{config['id']}.{pass_id}",
        )
        autonomous.run(initial_prompt=config["initial_prompt"])
        autonomous.write_context()
        operation_log = autonomous.get_context()

    response = {
        "pass_id": pass_id,
        "operation_log": operation_log,
        "model": llm.model,
    }
    test_case_fields = [
        "attacker",
        "target",
        "system_prompt",
        "initial_prompt",
        "ssh_command_timeout",
        "id",
    ]
    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]
    return response


class AutonomousUpliftBenchmark(Benchmark):
    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.in_sig_handler: bool = False
        # pyre-ignore
        self.response_result = []

        super().__init__(config)

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["autonomous-uplift"]

    def write_responses(self) -> None:
        self.response_path.write_text(json.dumps(self.response_result, indent=4))
        print(f"Wrote responses to {self.response_path}")

    def generator_interrupt_handler(
        self, signum: int, frame: Optional[FrameType]
    ) -> None:
        # Prevent re-entrancy
        if self.in_sig_handler:
            return
        self.in_sig_handler = True

        print(
            "Ctrl+C pressed while generating responses. Writing existing outputs to file before exiting..."
        )
        self.write_responses()

        # Reset CTRL+C handler to default, so user can exit all other threads
        signal.signal(signal.SIGINT, signal.default_int_handler)

        # Attempt to exit, it won't work, but attempt anyway
        exit(0)

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: int = 1
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the responses in a file. This is called before run().
        """
        self.response_result = []

        signal.signal(signal.SIGINT, self.generator_interrupt_handler)

        for llm in self.llms_under_test:
            self.logger.info(f"Querying {llm.model}")
            self.response_result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    prompt_path,
                    run_llm_in_parallel,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                    to_process_prompt=process_autonomous_run,
                )
            )

        self.write_responses()

    async def run(
        self,
        num_test_cases: int = 0,
        run_llm_in_parallel: int = 16,
        should_cleanup_after_eval: bool = True,
    ) -> None:
        """
        This method runs the grading part of the benchmark after the LLM has been prompted for responses.
        Args:
            run_llm_in_parallel (bool, optional): A flag indicating whether to run the language model in parallel. Defaults to True.
        """
        self.logger.warning("Grading is not implemented yet.")

        return
