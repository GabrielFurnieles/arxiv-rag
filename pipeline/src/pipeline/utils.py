import asyncio
import time


class AsyncRateLimiter:
    def __init__(self, max_rate: int, time_period: int):
        assert max_rate > 0, f"max_rate should be greater than 0, got {max_rate}"
        assert (
            time_period > 0
        ), f"time_period should be greater than 0, got {time_period}"

        self.max_rate = max_rate
        self.time_period = time_period

        self._lock = asyncio.Lock()
        self._start_time = None
        self._capacity = 0

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        pass

    async def acquire(self, amount: int = 1):
        assert amount > 0, f"amount should be greater than 0, got {amount}"

        if amount > self.max_rate:
            raise ValueError(
                f"Task's amount {amount} is over the maximum allowed rate {self.max_rate}"
            )

        while True:
            async with self._lock:
                now = time.monotonic()

                if (
                    self._start_time is None
                    or (now - self._start_time) > self.time_period
                ):
                    self._start_time = now
                    self._capacity = 0

                if amount + self._capacity <= self.max_rate:
                    self._capacity += amount
                    return

                wait_time = self.time_period - (now - self._start_time)

            await asyncio.sleep(wait_time)
