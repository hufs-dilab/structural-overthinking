"""
Dataset evaluation runner.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd
from smolagents import CodeAgent, Tool, ToolCallingAgent
from tqdm import tqdm

from .data_utils import QueryItem, prepare_dataset_items
from .logger import ExperimentLogger
from .model_config import ModelConfig

if TYPE_CHECKING:
    from src.parsers import ToolCallParser


@dataclass
class ProcessResult:
    """Result of processing a single query."""

    idx: int
    query: str
    metadata: Dict[str, Any]
    response: Optional[str]
    steps: List[Dict[str, Any]]
    error: Optional[Exception]


def create_agent(
    agent_type: str,
    tools: List[Tool],
    model,
    step_callbacks: Optional[List] = None,
) -> CodeAgent | ToolCallingAgent:
    """Create an agent of the specified type."""
    AgentClass = CodeAgent if agent_type == "code" else ToolCallingAgent
    return AgentClass(
        tools=tools,
        model=model,
        add_base_tools=True,
        step_callbacks=step_callbacks or [],
    )


class DatasetRunner:
    """Runs evaluation on a dataset."""

    def __init__(
        self,
        agent_type: str,
        tools: List[Tool],
        logger: ExperimentLogger,
        model_config: ModelConfig,
        num_workers: int = 1,
        tool_parser: Optional["ToolCallParser"] = None,
        step_callback_factory: Optional[Callable[[QueryItem, Optional[int], Optional[List[Dict[str, Any]]]], Any]] = None,
    ):
        self.agent_type = agent_type
        self.tools = tools
        self.logger = logger
        self.model_config = model_config
        self.num_workers = num_workers
        self.tool_parser = tool_parser
        self.step_callback_factory = step_callback_factory
        self._thread_local = threading.local()

    def _create_model(self):
        """Create model instance, with parser if specified."""
        if self.tool_parser:
            from src.models import ParsedOpenAIModel

            model = ParsedOpenAIModel(
                model_id=self.model_config.model_id,
                api_base=self.model_config.api_base,
                api_key=self.model_config.api_key,
                parser=self.tool_parser,
                max_tokens=self.model_config.max_tokens,
                client_kwargs={"timeout": self.model_config._create_timeout()},
            )
            return model
        return self.model_config.create_model()

    def run(self, dataset: pd.DataFrame) -> None:
        """Run evaluation on the dataset."""
        data_items = prepare_dataset_items(dataset)
        print(f"\n🚀 Starting evaluation on {len(data_items)} samples with {self.num_workers} workers...\n")

        if self.num_workers == 1:
            self._run_sequential(data_items)
        else:
            self._run_parallel(data_items)

    def _run_sequential(self, data_items: List[QueryItem]) -> None:
        """Run evaluation sequentially."""
        model = self._create_model()
        shared_agent = None
        if self.step_callback_factory is None:
            shared_agent = create_agent(
                self.agent_type,
                self.tools,
                model,
                step_callbacks=[self.logger.step_callback],
            )

        for item in tqdm(data_items, desc="Processing samples"):
            query_idx = self.logger.start_query(item.query, metadata=item.metadata)
            agent = shared_agent
            if agent is None:
                callbacks = [self.logger.step_callback]
                callbacks.append(self.step_callback_factory(item, query_idx, None))
                agent = create_agent(
                    self.agent_type,
                    self.tools,
                    model,
                    step_callbacks=callbacks,
                )
            try:
                response = agent.run(item.query, reset=True)
                self.logger.end_query(query_idx, str(response))
            except Exception as e:
                self.logger.log_error(query_idx, e)

    def _run_parallel(self, data_items: List[QueryItem]) -> None:
        """Run evaluation in parallel."""
        results: List[ProcessResult] = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_item, item): item for item in data_items}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                result = future.result()
                results.append(result)

        # Sort and log results
        results.sort(key=lambda x: x.idx)
        for result in results:
            query_idx = self.logger.start_query(result.query, metadata=result.metadata)
            if result.steps:
                self.logger.log_steps(query_idx, result.steps)
            if result.error is not None:
                self.logger.log_error(query_idx, result.error)
            else:
                self.logger.end_query(query_idx, result.response)

    def _process_item(self, item: QueryItem) -> ProcessResult:
        """Process a single item with thread-local agent."""
        agent, steps = self._get_thread_agent(item)

        try:
            response = agent.run(item.query, reset=True)
            return ProcessResult(
                idx=item.idx,
                query=item.query,
                metadata=item.metadata,
                response=str(response),
                steps=steps,
                error=None,
            )
        except Exception as e:
            return ProcessResult(
                idx=item.idx,
                query=item.query,
                metadata=item.metadata,
                response=None,
                steps=steps,
                error=e,
            )

    def _get_thread_agent(self, item: QueryItem) -> tuple:
        """Get or create agent for current thread with fresh step collector."""
        steps, step_callback = self.logger.create_step_collector()
        callbacks = [step_callback]
        if self.step_callback_factory is not None:
            callbacks.append(self.step_callback_factory(item, None, steps))

        if not hasattr(self._thread_local, "model"):
            self._thread_local.model = self._create_model()

        agent = create_agent(
            self.agent_type,
            self.tools,
            self._thread_local.model,
            step_callbacks=callbacks,
        )

        return agent, steps
