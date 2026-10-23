/**
 * Driver Drowsiness Detection (DDD) - Frontend Dashboard Controller
 * Real-time telemetry integration with FastAPI backend (/api/status).
 */

(() => {
  // DOM References
  const drowsinessBanner = document.getElementById("drowsinessBanner");
  const cameraDot = document.getElementById("cameraDot");
  const cameraStatusText = document.getElementById("cameraStatusText");
  const monitoringDot = document.getElementById("monitoringDot");
  const monitoringStatusText = document.getElementById("monitoringStatusText");
  const faceDot = document.getElementById("faceDot");
  const faceStatusText = document.getElementById("faceStatusText");
  const backendDot = document.getElementById("backendDot");
  const backendStatusText = document.getElementById("backendStatusText");
  const liveDot = document.getElementById("liveDot");
  const landmarksState = document.getElementById("landmarksState");
  const pipelineState = document.getElementById("pipelineState");
  const audioState = document.getElementById("audioState");

  const fpsBadge = document.getElementById("fpsBadge");
  const sessionElapsedBadge = document.getElementById("sessionElapsedBadge");
  const videoStatusTag = document.getElementById("videoStatusTag");

  const drowsinessCard = document.getElementById("drowsinessCard");
  const drowsinessBadge = document.getElementById("drowsinessBadge");
  const closureStateText = document.getElementById("closureStateText");
  const closureDurationText = document.getElementById("closureDurationText");
  const durationFill = document.getElementById("durationFill");

  const leftEarValue = document.getElementById("leftEarValue");
  const leftRawEarValue = document.getElementById("leftRawEarValue");
  const leftMeasurementState = document.getElementById("leftMeasurementState");
  const leftThresholdValue = document.getElementById("leftThresholdValue");
  const leftEyeStatus = document.getElementById("leftEyeStatus");
  const leftEarGauge = document.getElementById("leftEarGauge");
  const leftThresholdMarker = document.getElementById("leftThresholdMarker");

  const rightEarValue = document.getElementById("rightEarValue");
  const rightRawEarValue = document.getElementById("rightRawEarValue");
  const rightMeasurementState = document.getElementById("rightMeasurementState");
  const rightThresholdValue = document.getElementById("rightThresholdValue");
  const rightEyeStatus = document.getElementById("rightEyeStatus");
  const rightEarGauge = document.getElementById("rightEarGauge");
  const rightThresholdMarker = document.getElementById("rightThresholdMarker");

  const calibrationBadge = document.getElementById("calibrationBadge");
  const calibrationStatusText = document.getElementById("calibrationStatusText");
  const calibrationFill = document.getElementById("calibrationFill");
  const sampleCountsText = document.getElementById("sampleCountsText");
  const recalibrateBtn = document.getElementById("recalibrateBtn");
  const calibrationInstructionText = document.getElementById("calibrationInstructionText");
  const baselineValuesText = document.getElementById("baselineValuesText");
  const startMonitoringBtn = document.getElementById("startMonitoringBtn");
  const stopMonitoringBtn = document.getElementById("stopMonitoringBtn");
  const monitoringMessage = document.getElementById("monitoringMessage");
  const videoFeed = document.getElementById("videoFeed");
  const videoFeedMessage = document.getElementById("videoFeedMessage");
  const VIDEO_STALE_AFTER_MS = 3000;
  const VIDEO_RETRY_AFTER_MS = 5000;
  let pollPending = false;
  let monitoringActionPending = false;
  let recalibrationPending = false;
  let monitoringRunning = false;
  let latestStatus = null;
  let lastVideoUpdateAt = Date.now();
  let lastVideoRecoveryAt = 0;
  let recalibrationTimer = null;
  const recalibrateLabel = recalibrateBtn
    ? recalibrateBtn.textContent.trim()
    : "Recalibrate";

  function setOperationMessage(message, isError = false) {
    if (!monitoringMessage) return;
    monitoringMessage.textContent = message || "";
    monitoringMessage.classList.toggle("error", Boolean(isError && message));
  }

  function setVideoFeedMessage(message, isError = false) {
    if (!videoFeedMessage) return;
    videoFeedMessage.textContent = message || "";
    videoFeedMessage.hidden = !message;
    videoFeedMessage.classList.toggle("error", Boolean(isError && message));
  }

  function reconnectVideoFeed() {
    if (!videoFeed) return;
    const now = Date.now();
    lastVideoRecoveryAt = now;
    lastVideoUpdateAt = now;
    videoFeed.src = `/video_feed?refresh=${now}`;
  }

  if (videoFeed) {
    videoFeed.addEventListener("load", () => {
      lastVideoUpdateAt = Date.now();
      if (latestStatus) updateVideoFeedStatus(latestStatus);
    });
    videoFeed.addEventListener("error", () => {
      setVideoFeedMessage("Video connection failed. Reconnecting…", true);
      if (Date.now() - lastVideoRecoveryAt >= VIDEO_RETRY_AFTER_MS) {
        reconnectVideoFeed();
      }
    });
  }

  function renderMonitoringControls(running) {
    monitoringRunning = Boolean(running);
    if (liveDot) liveDot.classList.toggle("live", monitoringRunning);
    if (startMonitoringBtn) startMonitoringBtn.disabled = monitoringRunning || monitoringActionPending;
    if (stopMonitoringBtn) stopMonitoringBtn.disabled = !monitoringRunning || monitoringActionPending;
    if (recalibrateBtn) recalibrateBtn.disabled = !monitoringRunning || recalibrationPending;
  }

  async function requestMonitoringAction(path) {
    if (monitoringActionPending) return;
    monitoringActionPending = true;
    setOperationMessage("");
    if (recalibrationTimer !== null) {
      clearTimeout(recalibrationTimer);
      recalibrationTimer = null;
    }
    recalibrationPending = false;
    if (recalibrateBtn) recalibrateBtn.textContent = recalibrateLabel;
    renderMonitoringControls(monitoringRunning);
    try {
      const response = await fetch(path, { method: "POST" });
      const result = await response.json();
      if (!response.ok || result.status !== "ok") {
        throw new Error(result.message || `Request failed (${response.status})`);
      }
      setOperationMessage(result.message || "Monitoring request completed.");
    } catch (err) {
      setOperationMessage(err.message || "Monitoring request failed.", true);
    } finally {
      monitoringActionPending = false;
      renderMonitoringControls(monitoringRunning);
      await pollStatus();
    }
  }

  if (startMonitoringBtn) {
    startMonitoringBtn.addEventListener("click", () => requestMonitoringAction("/api/start"));
  }
  if (stopMonitoringBtn) {
    stopMonitoringBtn.addEventListener("click", () => requestMonitoringAction("/api/stop"));
  }

  if (recalibrateBtn) {
    recalibrateBtn.addEventListener("click", async () => {
      if (recalibrateBtn.disabled || !monitoringRunning) return;
      setOperationMessage("");
      recalibrationPending = true;
      renderMonitoringControls(monitoringRunning);
      try {
        recalibrateBtn.disabled = true;
        recalibrateBtn.textContent = "Calibrating...";
        const response = await fetch("/api/recalibrate", { method: "POST" });
        const result = await response.json();
        if (!response.ok || result.status !== "ok") {
          throw new Error(result.message || `Request failed (${response.status})`);
        }
        setOperationMessage(result.message || "Recalibration initiated.");
        recalibrationTimer = setTimeout(() => {
          recalibrationPending = false;
          recalibrationTimer = null;
          renderMonitoringControls(monitoringRunning);
          recalibrateBtn.textContent = recalibrateLabel;
        }, 5500);
      } catch (err) {
        setOperationMessage(err.message || "Recalibration request failed.", true);
        recalibrationPending = false;
        renderMonitoringControls(monitoringRunning);
        recalibrateBtn.textContent = recalibrateLabel;
      }
    });
  }

  // Format seconds into MM:SS display
  function formatSessionTime(seconds) {
    if (typeof seconds !== "number" || isNaN(seconds) || seconds < 0) {
      return "00:00";
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  // Update Eye Card helper
  function formatEar(value) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(3)
      : "Unavailable";
  }

  function updateEyeCard(
    measurement,
    rawValueEl,
    smoothedValueEl,
    measurementStateEl,
    threshEl,
    badgeEl,
    gaugeEl,
    markerEl
  ) {
    const rawEar = measurement && measurement.raw_ear !== undefined
      ? measurement.raw_ear
      : measurement && measurement.measurement_available === undefined
        ? measurement.ear
        : null;
    const smoothedEar = measurement && Object.prototype.hasOwnProperty.call(measurement, "smoothed_ear")
      ? measurement.smoothed_ear
      : measurement && measurement.ear;
    const measurementAvailable = Boolean(
      measurement &&
      (measurement.measurement_available === undefined
        ? rawEar !== null && rawEar !== undefined
        : measurement.measurement_available) &&
      rawEar !== null && rawEar !== undefined
    );
    const threshold = measurement ? measurement.threshold : null;

    rawValueEl.textContent = formatEar(measurementAvailable ? rawEar : null);
    smoothedValueEl.textContent = formatEar(smoothedEar);
    threshEl.textContent = typeof threshold === "number" && Number.isFinite(threshold)
      ? `Threshold: ${threshold.toFixed(3)}`
      : "Threshold: --";

    if (measurementStateEl) {
      measurementStateEl.textContent = !measurementAvailable
        ? "Measurement unavailable"
        : smoothedEar === null || smoothedEar === undefined
          ? "Raw available; smoothing starts after calibration"
          : "Measurement available";
      measurementStateEl.classList.toggle(
        "unavailable",
        !measurementAvailable || smoothedEar === null || smoothedEar === undefined
      );
    }

    const status = measurement ? measurement.status : "UNAVAILABLE";
    badgeEl.textContent = status || "UNAVAILABLE";
    if (status === "OPEN") {
      badgeEl.className = "eye-status-badge badge-open";
      gaugeEl.style.backgroundColor = "var(--accent-green)";
    } else if (status === "CLOSED") {
      badgeEl.className = "eye-status-badge badge-closed";
      gaugeEl.style.backgroundColor = "var(--accent-red)";
    } else {
      badgeEl.className = "eye-status-badge badge-neutral";
      gaugeEl.style.backgroundColor = "var(--accent-yellow)";
    }

    // Gauge visualization: scale EAR from 0.0 to 0.5 (normal open eye ~0.30 - 0.35)
    const gaugePercent = typeof smoothedEar === "number" && Number.isFinite(smoothedEar)
      ? Math.min(100, Math.max(0, (smoothedEar / 0.5) * 100))
      : 0;
    gaugeEl.style.width = `${gaugePercent}%`;

    if (typeof threshold === "number" && Number.isFinite(threshold)) {
      const markerPercent = Math.min(100, Math.max(0, (threshold / 0.5) * 100));
      markerEl.style.left = `${markerPercent}%`;
    } else {
      markerEl.style.left = "0%";
    }
  }

  function updateVideoFeedStatus(data) {
    if (!videoFeedMessage) return;

    if (data.processing_error) {
      setVideoFeedMessage(data.processing_error, true);
      return;
    }
    if (!data.running) {
      setVideoFeedMessage("Monitoring is stopped. Start monitoring to open the live feed.");
      return;
    }
    if (!data.camera_available) {
      setVideoFeedMessage(
        data.camera_error || "Camera unavailable. Check the camera connection and permissions.",
        true
      );
      return;
    }

    const now = Date.now();
    if (now - lastVideoUpdateAt > VIDEO_STALE_AFTER_MS) {
      setVideoFeedMessage("The live feed stopped updating. Reconnecting...", true);
      if (now - lastVideoRecoveryAt >= VIDEO_RETRY_AFTER_MS) {
        reconnectVideoFeed();
      }
      return;
    }
    setVideoFeedMessage("");
  }

  // Render Telemetry
  function updateDashboard(data) {
    latestStatus = data;
    renderMonitoringControls(Boolean(data.running));
    if (monitoringDot) monitoringDot.className = `status-dot ${data.running ? "green" : "gray"}`;
    if (monitoringStatusText) monitoringStatusText.textContent = `Monitoring: ${data.running ? "Running" : "Stopped"}`;
    if (pipelineState) pipelineState.textContent = data.running ? "Running" : "Stopped";
    if (audioState) audioState.textContent = data.alert_active ? "Alert event active" : "No active alert event";
    if (landmarksState) {
      landmarksState.textContent = data.face_landmarks_available
        ? "Available"
        : "Unavailable";
    }
    updateVideoFeedStatus(data);

    // 1. Backend health
    backendDot.className = "status-dot green";
    backendStatusText.textContent = "Backend: Online";

    // 2. Camera health
    if (data.camera_available) {
      cameraDot.className = "status-dot green";
      cameraStatusText.textContent = "Camera: Connected";
      const cameraPill = document.getElementById("cameraHealthPill");
      if (cameraPill) cameraPill.removeAttribute("title");
    } else {
      cameraDot.className = "status-dot red";
      cameraStatusText.textContent = data.camera_error ? "Camera: Error" : "Camera: Unavailable";
      const cameraPill = document.getElementById("cameraHealthPill");
      if (cameraPill) cameraPill.title = data.camera_error || "Camera unavailable";
    }

    // 3. Face detection status
    if (!data.running) {
      faceDot.className = "status-dot gray";
      faceStatusText.textContent = "Face: Inactive";
    } else if (data.face_detected) {
      faceDot.className = "status-dot green";
      faceStatusText.textContent = "Face: Detected";
    } else {
      faceDot.className = "status-dot yellow";
      faceStatusText.textContent = "Face: Searching";
    }
    const facePill = document.getElementById("faceStatusPill");
    if (facePill) facePill.title = data.face_landmarks_available
      ? "Face detected and landmarks available"
      : "Face landmarks unavailable";

    // 4. Video & session stats
    fpsBadge.textContent = `${(data.fps || 0).toFixed(1)} FPS`;
    sessionElapsedBadge.textContent = `Session: ${formatSessionTime(data.elapsed_seconds)}`;

    // 5. Drowsiness indicator & Banner
    const isTriggered = Boolean(data.drowsiness_triggered);
    const closureState = data.closure_state || "MEASUREMENT UNAVAILABLE";
    const duration = data.closure_duration || 0.0;
    const maxDuration = 2.0;

    closureStateText.textContent = !data.running
      ? "MONITORING STOPPED"
      : (isTriggered ? "DROWSINESS DETECTED" : closureState);
    closureDurationText.textContent = `${duration.toFixed(1)}s / ${maxDuration.toFixed(1)}s`;

    const durationPercent = Math.min(100, (duration / maxDuration) * 100);
    durationFill.style.width = `${durationPercent}%`;

    if (isTriggered) {
      drowsinessBanner.classList.remove("hidden");
      drowsinessCard.className = "card drowsiness-card alert-active";
      drowsinessBadge.textContent = "DANGER";
      drowsinessBadge.className = "badge tag-alert";
      closureStateText.className = "state-display state-alert";
      durationFill.className = "progress-bar-fill duration-fill danger";
      videoStatusTag.className = "tag tag-alert";
      videoStatusTag.textContent = "DROWSINESS ALERT";
    } else if (closureState === "EYES CLOSED" || closureState === "VERIFYING CLOSURE") {
      drowsinessBanner.classList.add("hidden");
      drowsinessCard.className = "card drowsiness-card warning-active";
      drowsinessBadge.textContent = "WARNING";
      drowsinessBadge.className = "badge";
      drowsinessBadge.style.backgroundColor = "var(--accent-yellow-bg)";
      drowsinessBadge.style.color = "var(--accent-yellow)";
      closureStateText.className = "state-display state-warning";
      durationFill.className = "progress-bar-fill duration-fill";
      videoStatusTag.className = "tag tag-normal";
      videoStatusTag.textContent = "EYES CLOSED";
    } else {
      drowsinessBanner.classList.add("hidden");
      drowsinessCard.className = "card drowsiness-card";
      drowsinessBadge.textContent = "NORMAL";
      drowsinessBadge.className = "badge";
      drowsinessBadge.style.backgroundColor = "var(--accent-green-bg)";
      drowsinessBadge.style.color = "var(--accent-green)";
      closureStateText.className = "state-display";
      durationFill.className = "progress-bar-fill duration-fill";
      videoStatusTag.className = "tag tag-normal";
      videoStatusTag.textContent = "SYSTEM ACTIVE";
    }

    if (!data.running) {
      drowsinessBanner.classList.add("hidden");
      drowsinessCard.className = "card drowsiness-card";
      drowsinessBadge.textContent = "STOPPED";
      drowsinessBadge.className = "badge";
      closureStateText.className = "state-display state-muted";
      videoStatusTag.className = "tag tag-normal";
      videoStatusTag.textContent = "MONITORING STOPPED";
      durationFill.style.width = "0%";
    }

    // 6. Left and Right Eye Cards
    updateEyeCard(
      data.left_eye,
      leftRawEarValue,
      leftEarValue,
      leftMeasurementState,
      leftThresholdValue,
      leftEyeStatus,
      leftEarGauge,
      leftThresholdMarker
    );
    updateEyeCard(
      data.right_eye,
      rightRawEarValue,
      rightEarValue,
      rightMeasurementState,
      rightThresholdValue,
      rightEyeStatus,
      rightEarGauge,
      rightThresholdMarker
    );

    // 7. Calibration Card
    const isCalibrated = Boolean(data.calibrated);
    const calibStatus = data.calibration_status || "NOT STARTED";
    const calibElapsed = data.calibration_elapsed_seconds !== undefined ? data.calibration_elapsed_seconds : (data.elapsed_seconds || 0);
    const remaining = Math.max(0, Math.ceil(5.0 - calibElapsed));

    if (!data.running) {
      calibrationBadge.textContent = "STOPPED";
      calibrationBadge.className = "badge";
      calibrationFill.style.width = "0%";
      calibrationStatusText.textContent = data.camera_error || "Monitoring stopped";
      if (calibrationInstructionText) {
        calibrationInstructionText.textContent = "Start monitoring to begin a fresh calibration.";
        calibrationInstructionText.style.color = "var(--text-muted)";
      }
    } else if (isCalibrated) {
      calibrationBadge.textContent = "COMPLETE";
      calibrationBadge.className = "badge";
      calibrationBadge.style.backgroundColor = "var(--accent-green-bg)";
      calibrationBadge.style.color = "var(--accent-green)";
      calibrationFill.style.width = "100%";
      calibrationFill.style.backgroundColor = "var(--accent-green)";
      calibrationStatusText.textContent = calibStatus;
      if (calibrationInstructionText) {
        calibrationInstructionText.textContent = "Calibration complete. Real-time monitoring active.";
        calibrationInstructionText.style.color = "var(--text-muted)";
      }
    } else {
      calibrationBadge.textContent = "CALIBRATING";
      calibrationBadge.className = "badge";
      calibrationBadge.style.backgroundColor = "var(--accent-yellow-bg)";
      calibrationBadge.style.color = "var(--accent-yellow)";
      const calibProgress = Math.min(100, (calibElapsed / 5.0) * 100);
      calibrationFill.style.width = `${calibProgress}%`;
      calibrationFill.style.backgroundColor = "var(--accent-yellow)";
      calibrationStatusText.textContent = `Calibrating: ${remaining}s remaining`;
      if (calibrationInstructionText) {
        calibrationInstructionText.textContent = "Face the camera, keep both eyes comfortably open, and look forward.";
        calibrationInstructionText.style.color = "var(--accent-yellow)";
      }
    }

    if (data.sample_counts) {
      sampleCountsText.textContent = `Left: ${data.sample_counts.left || 0} | Right: ${data.sample_counts.right || 0} (Min: 30)`;
    }

    if (baselineValuesText) {
      const leftBase = data.left_eye && data.left_eye.baseline !== null && data.left_eye.baseline !== undefined
        ? data.left_eye.baseline.toFixed(2)
        : "--";
      const rightBase = data.right_eye && data.right_eye.baseline !== null && data.right_eye.baseline !== undefined
        ? data.right_eye.baseline.toFixed(2)
        : "--";
      baselineValuesText.textContent = `Left: ${leftBase} | Right: ${rightBase}`;
    }
  }

  // Handle Backend Offline / Disconnected
  function handleBackendError() {
    backendDot.className = "status-dot red";
    backendStatusText.textContent = "Backend: Offline";
    if (monitoringRunning) {
      setVideoFeedMessage("Backend status is unavailable. The live feed may be stale.", true);
    }
  }

  // Keep one status request in flight so a slow response cannot queue requests.
  async function pollStatus() {
    if (pollPending) return;
    pollPending = true;
    const controller = new AbortController();
    const requestTimeout = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch("/api/status", {
        cache: "no-store",
        signal: controller.signal,
      });
      if (response.ok) {
        const data = await response.json();
        updateDashboard(data);
      } else {
        throw new Error(`Status request failed (${response.status})`);
      }
    } catch (_err) {
      handleBackendError();
    } finally {
      clearTimeout(requestTimeout);
      pollPending = false;
    }
  }

  // A single poll loop serves startup and all monitoring states.
  setInterval(pollStatus, 250);
  pollStatus();
})();
