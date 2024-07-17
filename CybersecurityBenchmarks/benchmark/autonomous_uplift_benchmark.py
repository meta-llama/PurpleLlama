# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import re
import signal
from pathlib import Path
from types import FrameType
from typing import cast, Dict, List, Optional, Tuple, Union
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm

from .benchmark import Benchmark
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

    def write_context(self, filepath: Union[str | None] = None) -> None:
        if filepath is None:
            filepath = (
                f"context_{self.ssh_client.hostname}_{self.llm.model}_{self.run_id}.txt"
            )
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
        *,
        llms_under_test: list[LLM],
        response_path: Path,
        judge_prompt_path: Optional[Path] = None,
        judge_response_path: Optional[Path] = None,
        stat_path: Optional[Path] = None,
        num_test_cases: int = 0,
        expansion_llm: Optional[LLM] = None,
        judge_llm: Optional[LLM] = None,
        pass_k: int = 1,
    ) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.in_sig_handler: bool = False
        self.judge_prompt_path = judge_prompt_path
        # pyre-ignore
        self.judge_prompts = []
        self.judge_challenge_ids: List[str] = []
        # pyre-ignore
        self.response_result = []

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

    def process_judge_run(
        self,
        prompt_id: int,
        pass_id: int,
        llms: Dict[str, LLM],
        test_case: Dict[str, Union[int, str]],
        guided_decode_json_schema: Optional[str] = None,
    ) -> Dict[str, Union[int, str]]:
        judge_llm = llms["judge_llm"]

        # JSON format: [ {challenge_id: 1a, success: True, ai_command: whoami, user_output: root}, {challenge_id: 2b, success: False}, {...} ]
        schema = '{"type":"array","items":{"type":"object","properties":{"challenge_id":{"type":"string"},"success":{"type":"boolean"},"ai_command":{"type":"string"},"user_output":{"type":"string"}},"required":["challenge_id","success"]}}'

        result = {}
        result["pass_id"] = pass_id
        result["model"] = test_case["model"]
        result["attacker"] = test_case["attacker"]
        result["target"] = test_case["target"]
        result["id"] = test_case["id"]
        result["challenges"] = []

        returned_challenges = []
        for judge_prompt in tqdm(
            self.judge_prompts, desc="Judge prompts", unit="judge"
        ):
            judge_result = judge_llm.query_with_system_prompt_with_retries(
                system_prompt=judge_prompt,
                prompt=str(test_case["operation_log"]),
                guided_decode_json_schema=schema,
            )

            # Judge results are returned in a list, but we just want the items in that list
            for challenge in json.loads(judge_result):
                returned_challenges.append(challenge)

        self.logger.debug(f"Judge results: {returned_challenges}")

        """
        returned_challenges is a list of all successful challenges, and some failed challenges.
        We need to make a complete list of every challenge's state. Unless it was explicitly successful, assume failure.

        for every challenge_id
            check if that was successful in judge's outputs
                if yes: add it to results with full details
            else
                add it to results with false
        """
        for challenge_id in self.judge_challenge_ids:
            for returned_challenge in returned_challenges:
                try:
                    if returned_challenge["challenge_id"] == challenge_id:
                        if returned_challenge.get("success") is True:
                            result["challenges"].append(
                                {
                                    "challenge_id": challenge_id,
                                    "success": True,
                                    "ai_command": returned_challenge.get("ai_command"),
                                    "user_output": returned_challenge.get(
                                        "user_output"
                                    ),
                                }
                            )

                            # Use the first success and continue with the next challenge_id
                            break
                except KeyError:
                    pass

            # Didn't find a success for this one in the judge results
            result["challenges"].append(
                {"challenge_id": challenge_id, "success": False}
            )

        return result

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: bool = False
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
        self, num_test_cases: int = 0, run_llm_in_parallel: bool = True
    ) -> None:
        """
        This method runs the grading part of the benchmark after the LLM has been prompted for responses.
        Args:
            run_llm_in_parallel (bool, optional): A flag indicating whether to run the language model in parallel. Defaults to True.
        """
        if not (self.judge_prompt_path and self.judge_response_path and self.judge_llm):
            raise ValueError(
                "Please provide --judge-prompt-path, --judge-response-path, and --judge-llm to automatically grade benchmark (or run benchmark with --only-generate-llm-responses)."
            )

        # Fixing annotation errors because the instance objects are Optional
        judge_prompt_path: Path = self.judge_prompt_path
        judge_response_path: Path = self.judge_response_path
        judge_llm: LLM = self.judge_llm

        self.logger.debug(f"Judging with {self.judge_llm.model}")

        self.judge_prompts = json.loads(judge_prompt_path.read_text())

        # Extract challenge IDs from judge prompts and sort them neatly
        for judge_prompt in self.judge_prompts:
            ids = re.findall(r"id:\s*([^\s]+)", judge_prompt)
            for idx in ids:
                self.judge_challenge_ids.append(idx)

        # Sort inputs like 1a, 1b, 2a, 3f
        def sort_num_alpha(s: str) -> Tuple[int, str]:
            match = re.match(r"([a-zA-Z]+)(\d+)([a-zA-Z]*)", s)
            if match:
                return (  # Return a tuple of (number, trailing letter)
                    int(match.group(2)),
                    match.group(3),
                )
            return (  # Fallback for any strings that don't match the expected pattern
                0,
                "",
            )

        self.judge_challenge_ids = sorted(self.judge_challenge_ids, key=sort_num_alpha)

        self.logger.debug(
            f"Loaded {len(self.judge_prompts)} judge prompts with {len(self.judge_challenge_ids)} challenges from {self.judge_prompt_path}"
        )

        judge_results = []
        judge_results.extend(
            query_llm_to_generate_responses(
                {"judge_llm": judge_llm},
                self.response_path,
                run_llm_in_parallel,
                pass_k=self.pass_k,
                num_test_cases=num_test_cases,
                to_process_prompt=self.process_judge_run,
            )
        )

        judge_response_path.write_text(json.dumps(judge_results, indent=4))
        print(
            f"Wrote {len(judge_results)} judge responses to {self.judge_response_path}"
        )

        return
