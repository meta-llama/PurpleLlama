# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import threading
from typing import List, Tuple

class ValidationBufferPool:
    """
    Pool de buffers contíguos de memória para evitar a fragmentação do heap
    e reduzir o overhead do Garbage Collector no laço crítico do LlamaFirewall.
    Conformidade estrita com as regras de tipagem Pyre e thread-safe.
    """
    def __init__(self, buffer_size: int = 8192, pool_capacity: int = 64) -> None:
        self._capacity: int = pool_capacity
        self._pool: List[bytearray] = [bytearray(buffer_size) for _ in range(pool_capacity)]
        self._lock: threading.Lock = threading.Lock()
        self._available: List[int] = list(range(pool_capacity))

    def acquire(self) -> Tuple[int, bytearray]:
        with self._lock:
            if not self._available:
                return -1, bytearray()
            index: int = self._available.pop()
            return index, self._pool[index]

    def release(self, index: int) -> None:
        if index == -1:
            return
        with self._lock:
            self._available.append(index)
