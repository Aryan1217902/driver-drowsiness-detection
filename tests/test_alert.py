import threading
import unittest

from alert import DrowsinessAlert


class DrowsinessAlertTests(unittest.TestCase):
    def setUp(self):
        self.play_calls = []
        self.playback_started = threading.Event()
        self.alert = DrowsinessAlert(
            play_warning=self.record_play,
            duration_ms=50,
        )

    def record_play(self):
        self.play_calls.append("play")
        self.playback_started.set()

    def test_plays_only_once_while_same_event_remains_active(self):
        self.assertTrue(self.alert.update(True))
        self.assertTrue(self.playback_started.wait(1))
        self.assertFalse(self.alert.update(True))
        self.assertFalse(self.alert.update(True))

        self.assertEqual(self.play_calls, ["play"])
        self.alert.close()

    def test_event_end_rearms_alert(self):
        self.alert.update(True)
        self.assertTrue(self.playback_started.wait(1))
        self.assertFalse(self.alert.update(False))

        self.alert._worker.join(1)
        self.playback_started.clear()
        self.assertTrue(self.alert.update(True))
        self.assertTrue(self.playback_started.wait(1))
        self.assertEqual(self.play_calls, ["play", "play"])
        self.alert.close()

    def test_false_without_active_event_does_not_play_or_stop(self):
        self.assertFalse(self.alert.update(False))

        self.assertEqual(self.play_calls, [])

    def test_new_event_does_not_overlap_still_playing_beep(self):
        release_playback = threading.Event()

        def blocked_playback():
            self.play_calls.append("play")
            self.playback_started.set()
            release_playback.wait(1)

        alert = DrowsinessAlert(play_warning=blocked_playback, duration_ms=50)
        self.assertTrue(alert.update(True))
        self.assertTrue(self.playback_started.wait(1))
        alert.update(False)

        # A separate event during the previous beep is suppressed, not overlapped.
        self.assertFalse(alert.update(True))
        self.assertEqual(self.play_calls, ["play"])

        release_playback.set()
        alert.close()
        alert.update(False)
        self.playback_started.clear()
        self.assertTrue(alert.update(True))
        self.assertTrue(self.playback_started.wait(1))
        alert.close()
        self.assertEqual(self.play_calls, ["play", "play"])

    def test_close_waits_for_warning_worker(self):
        self.alert.update(True)
        self.alert.close()

        self.assertEqual(self.play_calls, ["play"])
        self.assertFalse(self.alert._worker.is_alive())

    def test_play_error_is_reported(self):
        attempts = []

        def fail_playback():
            attempts.append("attempt")
            raise OSError("audio unavailable")

        alert = DrowsinessAlert(
            play_warning=fail_playback,
            duration_ms=50,
        )
        alert.update(True)
        alert._worker.join(1)
        with self.assertRaisesRegex(
            RuntimeError, "Could not play"
        ) as raised_error:
            alert.close()
        self.assertIsInstance(raised_error.exception.__cause__, OSError)
        self.assertEqual(len(attempts), 1)

    def test_invalid_audio_parameters_are_rejected(self):
        for kwargs in (
            {"frequency_hz": 0},
            {"frequency_hz": -1},
            {"duration_ms": 0},
            {"duration_ms": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    DrowsinessAlert(**kwargs)

    def test_close_without_playback_is_safe_and_idempotent(self):
        self.alert.close()
        self.alert.close()

        self.assertEqual(self.play_calls, [])

    def test_worker_playback_error_is_reported_by_update(self):
        def fail_playback():
            raise OSError("audio unavailable")

        alert = DrowsinessAlert(play_warning=fail_playback, duration_ms=50)
        self.assertTrue(alert.update(True))
        alert._worker.join(1)

        with self.assertRaisesRegex(RuntimeError, "Could not play"):
            alert.update(True)

        alert.close()


if __name__ == "__main__":
    unittest.main()
