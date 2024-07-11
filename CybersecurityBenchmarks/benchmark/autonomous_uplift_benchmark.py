# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from pathlib import Path
from typing import cast, Dict, List, Optional, Union
from warnings import filterwarnings

from tqdm import tqdm

from .llm import LLM
from .query_llm import query_llm_to_generate_responses

filterwarnings("ignore", category=DeprecationWarning)

import json
import logging

from .benchmark import Benchmark

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

    def update_message_context(self, prompt: str) -> None:
        self.messages_ctx.append(prompt)

    def prompt_llm(self, prompt: str) -> str:
        logging.debug(f"[Pre-LLM prompting] Prompt: {prompt}")

        # Add user prompt (previous command's output) to the context
        self.messages_ctx.append(prompt)
        logging.debug(f"[Pre-LLM prompting] Context: {self.messages_ctx}")

        # Prompt LLM with the context
        result = self.llm.chat_with_system_prompt_with_retries(
            system_prompt=self.system_prompt, prompt_with_history=self.messages_ctx
        )

        # Add AI's response to the context
        self.messages_ctx.append(result)
        logging.debug(f"[Post-LLM prompting] Context: {self.messages_ctx}")

        return result

    def execute_llm_command(self, prompt: str) -> str:
        llm_response = self.prompt_llm(prompt)
        logging.debug(f"LLM Response: {llm_response}")

        output = self.ssh_client.execute_command(llm_response)
        logging.debug(f"SSH Output: {output}")

        return output

    def run(self, initial_prompt: str = "") -> bool:
        logging.debug(
            f"Starting run {self.run_id} with initial prompt: {initial_prompt}"
        )

        cmd_output = self.execute_llm_command(initial_prompt)

        for shot in tqdm(
            range(1, self.shots_per_run + 1),
            desc=f"[Run {self.run_id}] Shots",
            unit="shot",
        ):
            logging.debug(
                f"Starting shot {shot}/{self.shots_per_run} in run {self.run_id}"
            )

            cmd_output = self.execute_llm_command(cmd_output)

        # Before finishing, add the final command output to the context
        self.update_message_context(cmd_output)

        logging.debug(f"Finished run {self.run_id}")

        return True

    def get_context(self) -> str:
        messages = []
        for i in range(len(self.messages_ctx)):
            if i % 2 == 0:
                messages.append(f">>> USER: {self.messages_ctx[i]}\n")
            else:
                messages.append(f">>> AI: {self.messages_ctx[i]}\n")

        return "".join(messages)

    def write_context(self, filepath: Union[str | None] = None) -> None:
        if filepath is None:
            filepath = (
                f"context_{self.ssh_client.hostname}_{self.llm.model}_{self.run_id}.txt"
            )
        with open(filepath, "w") as file:
            file.write(self.get_context())

        logging.debug(f"Context written to {filepath}")


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
        *,
        llms_under_test: list[LLM],
        response_path: Path,
        judge_response_path: Optional[Path] = None,
        stat_path: Optional[Path] = None,
        num_test_cases: int = 0,
        expansion_llm: Optional[LLM] = None,
        judge_llm: Optional[LLM] = None,
        pass_k: int = 1,
    ) -> None:

        # TODO: add class validation logic here, e.g. check if judge_llm is not None. If there
        # is no such requirement, there is no need to override the __init__ function

        super().__init__(
            llms_under_test=llms_under_test,
            response_path=response_path,
            judge_response_path=judge_response_path,
            stat_path=stat_path,
            num_test_cases=num_test_cases,
            judge_llm=judge_llm,
            pass_k=pass_k,
        )

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["autonomous-uplift"]

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: bool = False
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the responses in a file. This is called before run().
        """
        response_result = []
        for llm in self.llms_under_test:
            logging.info(f"Querying {llm.model}")
            response_result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    prompt_path,
                    run_llm_in_parallel,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                    to_process_prompt=process_autonomous_run,
                )
            )

        self.response_path.write_text(json.dumps(response_result, indent=4))

    async def run(
        self, num_test_cases: int = 0, run_llm_in_parallel: bool = True
    ) -> None:
        """
        This method runs the grading part of the benchmark after the LLM has been prompted for responses.
        Args:
            run_llm_in_parallel (bool, optional): A flag indicating whether to run the language model in parallel. Defaults to True.
        """
        logging.warning("Grading is not implemented yet.")
        return
