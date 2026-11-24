"""
Dataset-specific evaluators for checking correctness of agent responses.
"""

import contextlib
import faulthandler
import multiprocessing
import queue
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager to limit execution time."""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def reliability_guard():
    """
    Disables destructive functions to prevent generated code from interfering with tests.
    WARNING: This is NOT a security sandbox.
    """
    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None

    import shutil

    shutil.rmtree = None
    shutil.move = None

    import subprocess

    subprocess.Popen = None


def _get_execution_context():
    """Use fork when available so HumanEval checks also work from CLI snippets."""
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _unsafe_execute_humaneval(check_program: str, timeout: float, result_queue) -> None:
    """Execute a HumanEval check program in a child process."""
    try:
        reliability_guard()

        with time_limit(timeout):
            exec_globals = {}
            exec(check_program, exec_globals)
            result_queue.put({"passed": True, "result": "passed"})
    except TimeoutException:
        result_queue.put({"passed": False, "result": "timed out"})
    except AssertionError as e:
        result_queue.put({"passed": False, "result": f"failed: {str(e)}"})
    except BaseException as e:
        result_queue.put({"passed": False, "result": f"error: {type(e).__name__}: {str(e)}"})


class DatasetEvaluator:
    """Evaluates agent responses against ground truth answers."""

    def __init__(self, dataset_name: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            dataset_name: Name of the dataset being evaluated
        """
        self.dataset_name = dataset_name

    def evaluate(
        self,
        response: str,
        correct_answer: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate if response is correct.

        Args:
            response: Agent's response
            correct_answer: Ground truth answer
            metadata: Additional metadata (choices, answer_index, etc.)

        Returns:
            Dict with 'is_correct' and optional 'predicted_answer' keys
        """
        if metadata is None:
            metadata = {}

        result = {
            "is_correct": False,
            "predicted_answer": None,
        }

        # Dataset-specific evaluation
        if self.dataset_name and "gpqa" in self.dataset_name.lower():
            result = self._evaluate_gpqa(response, correct_answer, metadata)
        elif self.dataset_name and "mmlu" in self.dataset_name.lower():
            result = self._evaluate_mmlu(response, correct_answer, metadata)
        elif self.dataset_name and "aime" in self.dataset_name.lower():
            result = self._evaluate_aime(response, correct_answer, metadata)
        elif self.dataset_name and "human_eval" in self.dataset_name.lower():
            result = self._evaluate_human_eval(response, correct_answer, metadata)
        else:
            # Default: simple string matching
            result["is_correct"] = str(correct_answer).lower() in response.lower()
            result["predicted_answer"] = response

        return result

    def _evaluate_gpqa(self, response: str, correct_answer: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate GPQA response.

        GPQA format: choices[answer_index] == answer
        """
        choices = metadata.get("choices", [])
        correct_answer_index = metadata.get("correct_answer_index")

        # Extract answer from response (look for the final (A), (B), (C), (D) pattern)
        answer_matches = re.findall(r"\(([A-D])\)", response)
        if answer_matches:
            predicted_letter = answer_matches[-1]
            predicted_index = ord(predicted_letter) - ord("A")
            is_correct = predicted_index == correct_answer_index
            predicted_answer = predicted_letter
        else:
            # Fallback: check if correct answer text appears in response
            is_correct = str(correct_answer).lower() in response.lower()
            predicted_answer = None

        return {
            "is_correct": is_correct,
            "predicted_answer": predicted_answer,
        }

    def _evaluate_mmlu(self, response: str, correct_answer: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate MMLU response.

        MMLU format: direct comparison with answer option (A-J)
        """
        # Look for the final answer pattern (A) through (J)
        answer_matches = re.findall(r"\(([A-J])\)", response)
        if answer_matches:
            predicted_answer = answer_matches[-1]
            is_correct = predicted_answer.lower() == str(correct_answer).lower()
        else:
            answer_pattern = rf"(?:answer is|answer:)\s*\(?{re.escape(str(correct_answer))}\)?\b"
            is_correct = re.search(answer_pattern, response, flags=re.IGNORECASE) is not None
            predicted_answer = None

        return {
            "is_correct": is_correct,
            "predicted_answer": predicted_answer,
        }

    def _evaluate_aime(self, response: str, correct_answer: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate AIME response.

        AIME format: numerical answer
        """
        # Look for numbers in response
        numbers = re.findall(r"\b\d+\.?\d*\b", response)
        if numbers:
            try:
                predicted_number = float(numbers[-1])  # Take last number
                correct_number = float(correct_answer)
                is_correct = abs(predicted_number - correct_number) < 1e-6
                predicted_answer = predicted_number
            except (ValueError, TypeError):
                is_correct = False
                predicted_answer = None
        else:
            is_correct = str(correct_answer).lower() in response.lower()
            predicted_answer = None

        return {
            "is_correct": is_correct,
            "predicted_answer": predicted_answer,
        }

    def _evaluate_human_eval(self, response: str, correct_answer: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate HumanEval response by executing the code and running tests.

        Args:
            response: Generated code completion
            correct_answer: Not used (test function is in metadata)
            metadata: Must contain 'prompt', 'test', and 'entry_point'

        Returns:
            Dict with 'is_correct', 'predicted_answer', and 'result' keys
        """
        prompt = metadata.get("prompt", "")
        test = metadata.get("test", "")
        entry_point = metadata.get("entry_point", "")
        timeout = metadata.get("timeout", 3.0)

        if not prompt or not test or not entry_point:
            return {
                "is_correct": False,
                "predicted_answer": response,
                "result": "Missing required metadata (prompt, test, or entry_point)",
            }

        # Construct the check program
        check_program = prompt + response + "\n" + test + "\n" + f"check({entry_point})"

        # Run in separate process
        context = _get_execution_context()
        result_queue = context.Queue()
        process = context.Process(target=_unsafe_execute_humaneval, args=(check_program, timeout, result_queue))
        process.start()
        process.join(timeout=timeout + 1)

        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "is_correct": False,
                "predicted_answer": response,
                "result": "timed out (process killed)",
            }

        # Get result from child process
        try:
            child_result = result_queue.get_nowait()
        except queue.Empty:
            child_result = {
                "passed": False,
                "result": f"error: process exited with code {process.exitcode}",
            }

        passed = child_result.get("passed", False)
        result_msg = child_result.get("result", "unknown error")

        return {
            "is_correct": passed,
            "predicted_answer": response,
            "result": result_msg,
        }
