"""Focused tests for OpenRouter generation wait semantics."""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from acestep.openrouter_adapter import (
    _GenerationWaitTimeout,
    _generation_timeout_seconds,
    _openrouter_stream_generator,
    _wait_for_generation,
)


class OpenRouterGenerationTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Long healthy generations must not inherit a fixed ten-minute deadline."""

    @staticmethod
    def _record(**overrides: Any) -> SimpleNamespace:
        values = {
            "status": "running",
            "stage": "diffusion",
            "progress": 0.10,
            "updated_at": 0.0,
            "done_event": asyncio.Event(),
            "progress_queue": asyncio.Queue(),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_timeout_is_disabled_unless_positive(self):
        self.assertIsNone(_generation_timeout_seconds(None))
        self.assertIsNone(_generation_timeout_seconds(""))
        self.assertIsNone(_generation_timeout_seconds("bad"))
        self.assertIsNone(_generation_timeout_seconds("0"))
        self.assertIsNone(_generation_timeout_seconds("-5"))
        self.assertEqual(_generation_timeout_seconds("3600"), 3600.0)

    async def test_wait_without_timeout_completes_normally(self):
        rec = self._record()
        asyncio.get_running_loop().call_soon(rec.done_event.set)
        await _wait_for_generation(rec, None)
        self.assertTrue(rec.done_event.is_set())

    async def test_repeated_timestamp_heartbeats_do_not_hide_a_stall(self):
        rec = self._record()
        now = [0.0]

        async def heartbeat_only(_event: asyncio.Event, timeout: float) -> bool:
            now[0] += timeout
            rec.updated_at = now[0]
            return False

        with self.assertRaisesRegex(_GenerationWaitTimeout, "no status, stage, or progress"):
            await _wait_for_generation(
                rec,
                None,
                inactivity_timeout=3.0,
                poll_interval=1.0,
                clock=lambda: now[0],
                wait_once=heartbeat_only,
            )
        self.assertEqual(now[0], 3.0)

    async def test_progress_advance_resets_inactivity_deadline(self):
        rec = self._record()
        now = [0.0]
        waits = [0]

        async def advancing_then_done(_event: asyncio.Event, timeout: float) -> bool:
            now[0] += timeout
            waits[0] += 1
            if waits[0] == 2:
                rec.progress = 0.20
            if waits[0] == 5:
                rec.done_event.set()
                return True
            return False

        await _wait_for_generation(
            rec,
            None,
            inactivity_timeout=3.0,
            poll_interval=1.0,
            clock=lambda: now[0],
            wait_once=advancing_then_done,
        )
        self.assertEqual(now[0], 5.0)
        self.assertGreater(now[0], 3.0)

    async def test_stage_change_resets_deadline_even_when_progress_restarts(self):
        rec = self._record()
        now = [0.0]
        waits = [0]

        async def next_stage_then_done(_event: asyncio.Event, timeout: float) -> bool:
            now[0] += timeout
            waits[0] += 1
            if waits[0] == 2:
                rec.stage = "decoding"
                rec.progress = 0.01
            if waits[0] == 5:
                rec.done_event.set()
                return True
            return False

        await _wait_for_generation(
            rec,
            None,
            inactivity_timeout=3.0,
            poll_interval=1.0,
            clock=lambda: now[0],
            wait_once=next_stage_then_done,
        )
        self.assertEqual(now[0], 5.0)

    async def test_positive_hard_timeout_remains_an_overall_ceiling(self):
        rec = self._record()
        now = [0.0]

        async def always_progressing(_event: asyncio.Event, timeout: float) -> bool:
            now[0] += timeout
            rec.progress += 0.01
            return False

        with self.assertRaisesRegex(_GenerationWaitTimeout, "configured 2 seconds hard limit"):
            await _wait_for_generation(
                rec,
                2.0,
                inactivity_timeout=100.0,
                poll_interval=0.5,
                clock=lambda: now[0],
                wait_once=always_progressing,
            )

    async def test_stream_heartbeats_end_after_same_inactivity_limit(self):
        rec = self._record()
        now = [0.0]

        async def heartbeat_only(_queue: asyncio.Queue, timeout: float):
            now[0] += timeout
            rec.updated_at = now[0]
            return None

        chunks = []
        async for chunk in _openrouter_stream_generator(
            rec,
            "acestep/model",
            "mp3",
            inactivity_timeout=3.0,
            hard_timeout=None,
            poll_interval=1.0,
            clock=lambda: now[0],
            queue_wait=heartbeat_only,
        ):
            chunks.append(chunk)

        payload = "".join(chunks)
        self.assertIn("no status, stage, or progress change", payload)
        self.assertIn('"finish_reason": "error"', payload)
        self.assertTrue(payload.endswith("data: [DONE]\n\n"))
        self.assertEqual(now[0], 3.0)


if __name__ == "__main__":
    unittest.main()
