"""Windows sound warning for a newly detected drowsiness event."""

import threading

from config import ALERT_DURATION_MS, ALERT_FREQUENCY_HZ


class DrowsinessAlert:
    """Play one short background-thread warning per drowsiness event."""

    def __init__(
        self,
        play_warning=None,
        frequency_hz=ALERT_FREQUENCY_HZ,
        duration_ms=ALERT_DURATION_MS,
    ):
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be greater than zero")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be greater than zero")

        self._play_warning = (
            play_warning
            if play_warning is not None
            else lambda: self._play_windows_warning(frequency_hz, duration_ms)
        )
        self._duration_ms = duration_ms
        self._event_active = False
        self._worker = None
        self._worker_error = None
        self._error_lock = threading.Lock()

    def update(self, drowsiness_triggered):
        """Start a warning on the event's rising edge; never overlap sounds."""
        self._raise_worker_error()

        if drowsiness_triggered:
            if self._event_active:
                return False

            if self._worker is not None and self._worker.is_alive():
                self._event_active = True
                return False

            worker = threading.Thread(
                target=self._play_in_background,
                name="drowsiness-alert",
                daemon=True,
            )
            worker.start()
            self._worker = worker
            self._event_active = True
            return True

        self._event_active = False
        return False

    def close(self):
        """Wait briefly for the short beep and report any playback failure."""
        self._event_active = False
        if self._worker is not None:
            self._worker.join(self._duration_ms / 1000 + 0.1)
        self._raise_worker_error()

    def _play_in_background(self):
        try:
            self._play_warning()
        except Exception as error:
            with self._error_lock:
                self._worker_error = error

    def _raise_worker_error(self):
        with self._error_lock:
            error = self._worker_error
            self._worker_error = None
        if error is not None:
            raise RuntimeError("Could not play the drowsiness warning") from error

    @staticmethod
    def _play_windows_warning(frequency_hz, duration_ms):
        import winsound

        winsound.Beep(frequency_hz, duration_ms)
