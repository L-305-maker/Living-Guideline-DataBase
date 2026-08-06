# 集中化 GPU 推理队列: 把 SentenceTransformer.encode / CrossEncoder.predict
# 等非线程安全的 GPU 调用收口到一个 consumer 线程里串行执行,
# 并对同一窗口内的请求做 batch 合并, 从而在高并发下显著提升 GPU 利用率。
#
# 设计动机(背景):
# - 大模型 (Qwen3-Embedding-8B / Reranker-4B) 共享单卡 Ampere 16GB 显存。
#   多线程并发调 SentenceTransformer.encode 会触发 tensor parallel 与 CUDA
#   上下文冲突, 表现为偶发 NaN / OOM / 慢路径 (静默退化到 CPU)。
# - 现状: 每个 MCP 请求都单条调用 model.encode(query), 完全没利用 batch。
# - 改造后: 请求 producer 把 (texts, future) 投进队列; consumer 在
#   [GPU_BATCH_WAIT_MS] 窗口内攒批, 单次 model.encode(batch) 吃满 batch。
#
# 关键约束:
# - 严格 1 个 consumer 线程, 不并发调 GPU 端 (即"GPU 独占"保证)。
# - 每个 (model_name, op) 一个独立队列, 互不阻塞 (embedder 不被 reranker 拖)。
# - 单 request 失败时只 fail 自己, 其它 request 继续。
# - 模块级 lazy start: 第一次 submit 才起 consumer 线程。
# - daemon=True: 进程退出时不卡。
#
# 与 MCP/检索链路集成:
#   query_embedding.query_vector_literal()  => GPUQueue.submit_embed()
#   reranker.BgeM3Document/ChunkReranker.rerank()  => GPUQueue.submit_predict_one()
# 重构后两个调用点都不直接 model.encode / model.predict,
# 而是把单条请求推到 GPU queue, 在 consumer 里合并并执行。

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

# 队列/批处理默认值 (环境变量覆盖)
DEFAULT_MAX_QUEUE = int(os.getenv("GPU_QUEUE_MAX", "1024"))         # 每个 (model,op) 队列上限
DEFAULT_BATCH_WAIT_MS = float(os.getenv("GPU_BATCH_WAIT_MS", "8"))  # 攒批最大等待
DEFAULT_MAX_BATCH = int(os.getenv("GPU_BATCH_MAX", "32"))           # 单次 GPU 调用最大批
DEFAULT_SUBMIT_TIMEOUT = float(os.getenv("GPU_SUBMIT_TIMEOUT", "30"))


class _Future:
    """轻量 future: set_result / set_exception / result(timeout)。

    不借 threading.Future 是为了: (1) 避免 GIL/异常路径; (2) 提交时直接同步
    trigger consumer 启动而不是依赖 __init__ 的 side-effect; (3) dict 友好。
    """

    __slots__ = ("_value", "_exc", "_event")

    def __init__(self) -> None:
        self._value: Any = None
        self._exc: BaseException | None = None
        self._event = threading.Event()

    def set_result(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exc = exc
        self._event.set()

    def result(self, timeout: float | None = None) -> Any:
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("gpu_queue: result() timed out")
        if self._exc is not None:
            raise self._exc
        return self._value


class _QueueKey:
    """(model_name, op) 复合键, 用于把不同模型/操作的请求路由到不同 consumer。"""

    __slots__ = ("model_name", "op")

    def __init__(self, model_name: str, op: str) -> None:
        self.model_name = model_name
        self.op = op

    def __hash__(self) -> int:
        return hash((self.model_name, self.op))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _QueueKey):
            return NotImplemented
        return self.model_name == other.model_name and self.op == other.op


class _Lane:
    """单 (model, op) lane: 一个 FIFO queue + 一个 consumer 线程。"""

    def __init__(
        self,
        key: _QueueKey,
        runner: Callable[[list[Any]], list[Any]],
        max_queue: int,
        batch_wait_ms: float,
        max_batch: int,
    ) -> None:
        self.key = key
        self._runner = runner
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._batch_wait_ms = batch_wait_ms
        self._max_batch = max_batch
        self._consumer: threading.Thread | None = None
        # 启动动作在 producer 第一次 submit 时触发 (lazy)

    def submit(self, payload: Any, fut: _Future) -> None:
        # queue.Queue.put_nowait 在满时直接抛 queue.Full, 由 caller 决定是否重试
        self._q.put_nowait((payload, fut))
        self._ensure_consumer()

    def _ensure_consumer(self) -> None:
        if self._consumer is not None and self._consumer.is_alive():
            return
        # 双检锁避免多线程同时启动两次 consumer
        t = threading.Thread(
            target=self._run,
            name=f"gpu-queue-{self.key.model_name}-{self.key.op}-{uuid.uuid4().hex[:6]}",
            daemon=True,
        )
        t.start()
        self._consumer = t

    def _drain_batch(self) -> list[tuple[Any, _Future]]:
        """攒一批请求: 等满 max_batch 或 batch_wait_ms 到期。

        返回的 batch: [(payload, future), ...]
        每个 payload 由调用方语义决定, 但 runner 必须:
          batch_results = runner([p for p, _ in batch])
        返回的 batch_results 必须与 batch 等长 (1:1 对应)。
        """
        batch: list[tuple[Any, _Future]] = []
        deadline = time.perf_counter() + self._batch_wait_ms / 1000.0
        try:
            first = self._q.get_nowait()
        except queue.Empty:  # pragma: no cover - put_nowait 后立即 get
            return batch
        batch.append(first)
        while len(batch) < self._max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        # 单 consumer 线程: 这里就是 GPU 独占的临界区
        LOGGER.debug("gpu_queue lane start key=%s", (self.key.model_name, self.key.op))
        while True:
            batch = self._drain_batch()
            if not batch:
                # 没有 item: 重新等 (理论上 producer 一定会 put)
                try:
                    pair = self._q.get()
                except (OSError, ValueError):
                    return
                batch = [pair]
            payloads = [payload for payload, _fut in batch]
            try:
                # runner 一次吃完整批: 输入 list[payload], 输出 list[result] 1:1
                # 典型实现: 用 stack/pad 把 list 合并到一次 GPU 调用, 拆结果按顺序返回
                results = self._runner(payloads)
            except BaseException as exc:  # 整批失败: 全部 future fail
                LOGGER.exception("gpu_queue runner failed key=%s", self.key)
                for _payload, fut in batch:
                    fut.set_exception(exc)
                continue
            if len(results) != len(batch):
                # runner 应当 1:1 输出; 不等就 fail 全部, 避免 silently misalign
                err = RuntimeError(
                    f"gpu_queue runner returned {len(results)} for {len(batch)} inputs (key={self.key})"
                )
                for _payload, fut in batch:
                    fut.set_exception(err)
                continue
            for (_payload, fut), result in zip(batch, results):
                fut.set_result(result)


class GPUQueue:
    """进程级 GPU 推理入口: 按 (model_name, op) 分 lane 提交批处理任务。

    典型用法:
        gpu = GPUQueue()
        fut = gpu.submit_embed("Qwen/...", "doc A")   # 单条 producer, 跨请求合并
        vec = fut.result(timeout=30)
    """

    def __init__(
        self,
        max_queue: int = DEFAULT_MAX_QUEUE,
        batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        self._max_queue = max_queue
        self._batch_wait_ms = batch_wait_ms
        self._max_batch = max_batch
        self._lock = threading.Lock()
        self._lanes: dict[_QueueKey, _Lane] = {}
        # 注册的 runner: 模型 + op -> callable (list[inputs]) -> list[results]
        self._runners: dict[_QueueKey, Callable[[list[Any]], list[Any]]] = {}
        self._disabled = os.getenv("GPU_QUEUE_DISABLED", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

    # ----- 注册 runner (由应用启动时一次性注册) -----
    def is_registered(self, model_name: str, op: str) -> bool:
        with self._lock:
            return _QueueKey(model_name, op) in self._runners

    def register(self, model_name: str, op: str, runner: Callable[[list[Any]], list[Any]]) -> None:
        """注册 (model_name, op) 的 runner: 输入 list -> 输出 list (对齐)。

        通常: op = "embed" 或 op = "predict", runner 内部 import torch / 模型。
        """
        with self._lock:
            self._runners[_QueueKey(model_name, op)] = runner

    # ----- producer 接口 -----
    def submit(self, model_name: str, op: str, payload: Any, timeout: float = DEFAULT_SUBMIT_TIMEOUT) -> _Future:
        """异步提交一个任务, 返回 _Future (不可跨进程使用)。

        queue 满 / runner 未注册 / GPU 异常 都会通过 future.set_exception 传播。
        """
        if self._disabled:
            raise RuntimeError("GPU_QUEUE_DISABLED is set; direct calls required")
        with self._lock:
            runner = self._runners.get(_QueueKey(model_name, op))
            if runner is None:
                raise RuntimeError(
                    f"GPUQueue: no runner for model={model_name!r} op={op!r}; "
                    "register first via gpu.register(model_name, op, runner)"
                )
            lane = self._lanes.get(_QueueKey(model_name, op))
            if lane is None:
                lane = _Lane(
                    _QueueKey(model_name, op), runner,
                    max_queue=self._max_queue,
                    batch_wait_ms=self._batch_wait_ms,
                    max_batch=self._max_batch,
                )
                self._lanes[_QueueKey(model_name, op)] = lane
        fut = _Future()
        try:
            lane.submit(payload, fut)
        except queue.Full:
            fut.set_exception(RuntimeError(
                f"GPUQueue lane full: key=({model_name!r},{op!r}) "
                f"max_queue={self._max_queue}; raise MAX or reduce concurrency"
            ))
        return fut

    # ----- 高级接口 -----
    def submit_embed(self, model_name: str, text: str, timeout: float = DEFAULT_SUBMIT_TIMEOUT) -> list[float]:
        """同步嵌入单条 query: 等 GPU 完成, 返回向量。

        在 consumer 视角: 多个 producer 的单条请求被合并成一个 batch,
        单次 model.encode(batch) 完成, 然后按请求顺序拆回。
        Runner 必须满足 1:1 对齐:
          inputs = [q1, q2, ..., qn]   # 每项是单条 str
          outputs = [v1, v2, ..., vn]  # 每项是 list[float], 与 inputs 一一对应
        """
        fut = self.submit(model_name, "embed", text, timeout=timeout)
        vec = fut.result(timeout=timeout)
        if not isinstance(vec, list):
            raise RuntimeError("GPUQueue embed runner returned unexpected shape")
        return list(vec)

    def submit_embed_texts(
        self,
        model_name: str,
        texts: list[str],
        timeout: float = DEFAULT_SUBMIT_TIMEOUT,
    ) -> list[list[float]]:
        """同步嵌入多条 query: 把整批作为单 request 提交。

        适用: caller 已经攒批 (比如 vector_search_*_pg 一次跑多 query)。
        跨请求合并见 submit_embed。
        """
        if not texts:
            return []
        fut = self.submit(model_name, "embed", texts, timeout=timeout)
        result = fut.result(timeout=timeout)
        if not isinstance(result, list):
            raise RuntimeError("GPUQueue embed runner returned unexpected shape")
        return result

    def submit_predict_one(
        self,
        model_name: str,
        pair: tuple[str, str],
        timeout: float = DEFAULT_SUBMIT_TIMEOUT,
    ) -> float:
        """同步预测单对: 返回该 pair 的 score。

        与 submit_predict_pairs 不同: 每个 producer 单 pair, 跨请求合并。
        """
        fut = self.submit(model_name, "predict", pair, timeout=timeout)
        s = fut.result(timeout=timeout)
        if not isinstance(s, (int, float)):
            raise RuntimeError("GPUQueue predict runner returned non-scalar")
        return float(s)

    def submit_predict_pairs(
        self,
        model_name: str,
        pairs: list[tuple[str, str]],
        timeout: float = DEFAULT_SUBMIT_TIMEOUT,
    ) -> list[float]:
        """同步预测多对 (常用于 reranker 一次性 batch): 返回与 pairs 等长的 score 列表。

        整批作为单 request 提交, 适合 caller 已经攒批的场景。
        """
        if not pairs:
            return []
        fut = self.submit(model_name, "predict", pairs, timeout=timeout)
        scores = fut.result(timeout=timeout)
        if not isinstance(scores, list):
            raise RuntimeError("GPUQueue predict runner returned unexpected shape")
        return [float(s) for s in scores]  # type: ignore[arg-type]  # runtime-checked


# module-level singleton: 单进程共享一个 GPU queue
_QUEUE_SINGLETON: GPUQueue | None = None
_SINGLETON_LOCK = threading.Lock()


def get_gpu_queue() -> GPUQueue:
    """获取进程级 GPUQueue 单例 (lazy)。"""
    global _QUEUE_SINGLETON
    with _SINGLETON_LOCK:
        if _QUEUE_SINGLETON is None:
            _QUEUE_SINGLETON = GPUQueue()
    return _QUEUE_SINGLETON
