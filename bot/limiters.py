# bot/limiters.py

import collections
import math
import random
import threading
import time


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = max(0.01, rate_per_sec)
        self.capacity = max(1, burst)
        self.tokens = self.capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            # need to wait for deficit
            deficit = 1.0 - self.tokens
            return deficit / self.rate


class SlidingWindowCounter:
    def __init__(self, window_sec: int):
        self.window = window_sec
        self.events = collections.deque()  # (ts, key)
        self.per_key = collections.Counter()
        self.lock = threading.Lock()

    def add(self, key):
        now = time.time()
        with self.lock:
            self.events.append((now, key))
            self.per_key[key] += 1
            self._gc(now)

    def count(self, key):
        now = time.time()
        with self.lock:
            self._gc(now)
            return self.per_key.get(key, 0)

    def _gc(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            _, k = self.events.popleft()
            self.per_key[k] -= 1
            if self.per_key[k] <= 0:
                self.per_key.pop(k, None)


class CircuitBreaker429:
    CLOSED, OPEN, HALF = "closed", "open", "half-open"

    def __init__(self, threshold: int, open_secs: int, half_probe_secs: int):
        self.th = max(1, threshold)
        self.open_secs = open_secs
        self.half_probe_secs = half_probe_secs
        self.state = self.CLOSED
        self.strikes = 0
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def before(self):
        with self.lock:
            now = time.time()
            if self.state == self.OPEN:
                if now >= self.next_allowed:
                    self.state = self.HALF
                else:
                    raise RuntimeError("circuit-open")
            if self.state == self.HALF:
                # allow one probe
                return

    def on_success(self):
        with self.lock:
            self.state = self.CLOSED
            self.strikes = 0

    def on_429(self):
        with self.lock:
            self.strikes += 1
            if self.state in (self.CLOSED, self.HALF) and self.strikes >= self.th:
                self.state = self.OPEN
                self.next_allowed = time.time() + self.open_secs
            elif self.state == self.HALF:
                self.state = self.OPEN
                self.next_allowed = time.time() + self.open_secs

    def on_other_error(self):
        # no-op; only 429 trips circuit
        pass


def full_jitter_sleep(base: float, attempt: int, cap: float = 60.0):
    delay = min(cap, base * (2 ** attempt))
    time.sleep(random.uniform(0, delay))
