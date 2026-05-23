"""
MindPong: a quick Pygame + BrainFlow blink-control prototype.

Target board: NeuroPawn Knight Board IMU on COM3, matching the NeuroPawn EXG
Visualizer profile:
    - Board Type: Knight Board IMU
    - EXG channels 1-4 active
    - EXG channels 5-8 disabled to avoid floating-channel noise
    - Gain 12
    - RLD routed on channels 1-4

Close the NeuroPawn EXG Visualizer before starting this script. Only one process
can own COM3 at a time.

Controls:
    W / S       Move player paddle up / down
    C           Toggle control pair between Ch1/Ch2 and Ch3/Ch4
    R           Recalibrate EEG blink detector
    Esc / close Quit

Install:
    pip install pygame brainflow

If BrainFlow is not installed or the board cannot be opened, the game still runs
in keyboard-only mode.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from statistics import median, pstdev
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required. Install it with: pip install pygame") from exc


SERIAL_PORT = "COM3"
EXG_GAIN = 12
ACTIVE_EXG_CHANNELS = (1, 2, 3, 4)
DISABLED_EXG_CHANNELS = (5, 6, 7, 8)
CONTROL_CHANNEL_PAIRS = ((1, 2), (3, 4))
CONFIG_COMMAND_PAUSE_SECONDS = 0.10
CALIBRATION_SECONDS = 5.0
DOUBLE_BLINK_SECONDS = 0.55
BLINK_REFRACTORY_SECONDS = 0.18
BASELINE_ALPHA = 0.004
HISTORY_SECONDS = 6.0

WIDTH = 960
HEIGHT = 540
FPS = 60
PADDLE_W = 16
PADDLE_H = 112
BALL_SIZE = 14
PADDLE_SPEED = 420.0
AI_SPEED = 330.0
BLINK_STEP = 74

BG = (9, 13, 18)
PANEL = (18, 25, 33)
TEXT = (229, 235, 241)
MUTED = (143, 155, 168)
ACCENT = (93, 215, 181)
WARN = (255, 191, 87)
DANGER = (255, 105, 105)
LINE = (70, 83, 98)
PLAYER = (122, 166, 255)
OPPONENT = (240, 172, 103)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class EEGFrame:
    ch1: float
    ch2: float
    signal: float
    timestamp: float


class BrainFlowEEG:
    """Small wrapper that keeps all BrainFlow failures non-fatal."""

    def __init__(self, serial_port: str = SERIAL_PORT) -> None:
        self.serial_port = serial_port
        self.board = None
        self.board_id = None
        self.sample_rate = 125
        self.exg_rows: List[int] = []
        self.channel_rows = {}
        self.control_pair = CONTROL_CHANNEL_PAIRS[0]
        self.control_rows: List[int] = []
        self.config_warnings: List[str] = []
        self.status = "Keyboard-only mode"
        self.ok = False
        self.streaming = False

        try:
            from brainflow.board_shim import (  # type: ignore
                BoardIds,
                BoardShim,
                BrainFlowInputParams,
            )
        except Exception as exc:
            self.status = f"BrainFlow unavailable: {exc.__class__.__name__}"
            return

        try:
            board_enum = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU
            self.board_id = getattr(board_enum, "value", board_enum)
        except AttributeError:
            self.status = "BrainFlow too old: NEUROPAWN_KNIGHT_BOARD_IMU missing"
            return

        try:
            params = BrainFlowInputParams()
            params.serial_port = serial_port
            params.other_info = f'{{"gain": {EXG_GAIN}}}'

            self.sample_rate = int(BoardShim.get_sampling_rate(self.board_id))
            get_exg_channels = getattr(BoardShim, "get_exg_channels", None)
            if get_exg_channels is not None:
                exg_rows = list(get_exg_channels(self.board_id))
            else:
                exg_rows = list(BoardShim.get_eeg_channels(self.board_id))
            if len(exg_rows) < 2:
                self.status = "Board exposes fewer than 2 EXG channels"
                return

            self.exg_rows = [int(row) for row in exg_rows]
            self.channel_rows = {
                channel_num: row
                for channel_num, row in enumerate(self.exg_rows, start=1)
            }
            self._select_control_pair(CONTROL_CHANNEL_PAIRS[0])
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.board.start_stream(45000, "")
            self.streaming = True
            self._configure_visualizer_profile()
            self.ok = True
            self.status = (
                f"BrainFlow Knight Board IMU on {serial_port}; "
                f"EXG Ch{self.control_pair[0]}/Ch{self.control_pair[1]}, gain {EXG_GAIN}"
            )
            if self.config_warnings:
                self.status += f"; config warnings {len(self.config_warnings)}"
        except Exception as exc:
            self.status = f"BrainFlow keyboard-only fallback: {exc.__class__.__name__}: {exc}"
            self.shutdown()

    def _select_control_pair(self, pair: Tuple[int, int]) -> bool:
        if all(channel in self.channel_rows for channel in pair):
            self.control_pair = pair
            self.control_rows = [self.channel_rows[pair[0]], self.channel_rows[pair[1]]]
            return True

        fallback_rows = self.exg_rows[:2]
        if len(fallback_rows) < 2:
            return False

        self.control_pair = (1, 2)
        self.control_rows = fallback_rows
        return True

    def cycle_control_pair(self) -> bool:
        usable_pairs = [
            pair
            for pair in CONTROL_CHANNEL_PAIRS
            if all(channel in self.channel_rows for channel in pair)
        ]
        if not usable_pairs:
            return False

        try:
            current_index = usable_pairs.index(self.control_pair)
        except ValueError:
            current_index = -1
        next_pair = usable_pairs[(current_index + 1) % len(usable_pairs)]
        if not self._select_control_pair(next_pair):
            return False

        self.status = (
            f"BrainFlow Knight Board IMU on {self.serial_port}; "
            f"EXG Ch{self.control_pair[0]}/Ch{self.control_pair[1]}, gain {EXG_GAIN}"
        )
        if self.config_warnings:
            self.status += f"; config warnings {len(self.config_warnings)}"
        return True

    def _configure_visualizer_profile(self) -> None:
        if self.board is None:
            return

        commands = []
        for channel in DISABLED_EXG_CHANNELS:
            commands.append(f"rldremove_{channel}")
            commands.append(f"choff_{channel}")
        for channel in ACTIVE_EXG_CHANNELS:
            commands.append(f"chon_{channel}_{EXG_GAIN}")
            commands.append(f"rldadd_{channel}")

        for command in commands:
            try:
                self.board.config_board(command)
                time.sleep(CONFIG_COMMAND_PAUSE_SECONDS)
            except Exception as exc:
                self.config_warnings.append(f"{command}: {exc.__class__.__name__}")

        try:
            self.board.get_board_data()
        except Exception:
            pass

    def read_frames(self) -> List[EEGFrame]:
        if not self.ok or self.board is None or len(self.control_rows) < 2:
            return []

        try:
            data = self.board.get_board_data()
        except Exception as exc:
            self.status = f"BrainFlow read failed; keyboard-only: {exc.__class__.__name__}"
            self.ok = False
            self.shutdown()
            return []

        if data is None:
            return []

        try:
            sample_count = int(data.shape[1])
        except Exception:
            return []
        if sample_count <= 0:
            return []

        ch1_row, ch2_row = self.control_rows
        try:
            ch1_values = data[ch1_row]
            ch2_values = data[ch2_row]
        except Exception as exc:
            self.status = f"EEG channel read failed: {exc.__class__.__name__}"
            return []

        now = time.monotonic()
        first_ts = now - (sample_count - 1) / max(1, self.sample_rate)
        frames: List[EEGFrame] = []
        for idx in range(sample_count):
            ch1 = float(ch1_values[idx])
            ch2 = float(ch2_values[idx])
            frames.append(
                EEGFrame(
                    ch1=ch1,
                    ch2=ch2,
                    signal=(ch1 + ch2) * 0.5,
                    timestamp=first_ts + idx / max(1, self.sample_rate),
                )
            )
        return frames

    def shutdown(self) -> None:
        if self.board is None:
            return
        try:
            if self.streaming:
                self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass
        self.streaming = False


class BlinkDetector:
    def __init__(self, sample_rate: int = 125) -> None:
        self.sample_rate = max(1, sample_rate)
        self.reset()

    def reset(self) -> None:
        now = time.monotonic()
        self.started_at = now
        self.calibrated = False
        self.calibration_values: List[float] = []
        self.baseline = 0.0
        self.threshold = 0.0
        self.current_ch1 = 0.0
        self.current_ch2 = 0.0
        self.current_signal = 0.0
        self.current_metric = 0.0
        self.above_threshold = False
        self.last_blink_time = -999.0
        self.pending_single_time: Optional[float] = None
        self.last_action = "none"
        self.last_action_time = 0.0
        self.events: Deque[str] = deque(maxlen=7)
        self.metric_history: Deque[float] = deque(
            maxlen=int(self.sample_rate * HISTORY_SECONDS)
        )
        self.signal_history: Deque[float] = deque(
            maxlen=int(self.sample_rate * HISTORY_SECONDS)
        )

    def calibration_remaining(self) -> float:
        if self.calibrated:
            return 0.0
        return max(0.0, CALIBRATION_SECONDS - (time.monotonic() - self.started_at))

    def update(self, frames: Iterable[EEGFrame]) -> List[str]:
        actions: List[str] = []
        now = time.monotonic()

        for frame in frames:
            self.current_ch1 = frame.ch1
            self.current_ch2 = frame.ch2
            self.current_signal = frame.signal
            self.signal_history.append(frame.signal)

            if not self.calibrated:
                self.calibration_values.append(frame.signal)
                self.current_metric = abs(frame.signal - self.baseline)
                self.metric_history.append(self.current_metric)
                continue

            self.current_metric = abs(frame.signal - self.baseline)
            self.metric_history.append(self.current_metric)

            if self.threshold <= 0.0:
                continue

            low_gate = self.threshold * 0.45
            high_gate = self.threshold

            if self.above_threshold:
                if self.current_metric < low_gate:
                    self.above_threshold = False
            elif (
                self.current_metric >= high_gate
                and frame.timestamp - self.last_blink_time >= BLINK_REFRACTORY_SECONDS
            ):
                self.above_threshold = True
                self.last_blink_time = frame.timestamp
                actions.extend(self._register_blink(frame.timestamp))

            if self.current_metric < self.threshold * 0.65:
                self.baseline += BASELINE_ALPHA * (frame.signal - self.baseline)

        if not self.calibrated and now - self.started_at >= CALIBRATION_SECONDS:
            self._finish_calibration()

        if (
            self.pending_single_time is not None
            and now - self.pending_single_time > DOUBLE_BLINK_SECONDS
        ):
            self.pending_single_time = None
            self.last_action = "single blink: up"
            self.last_action_time = now
            self.events.appendleft("single blink/jaw -> paddle up")
            actions.append("up")

        return actions

    def _finish_calibration(self) -> None:
        values = self.calibration_values[-int(self.sample_rate * CALIBRATION_SECONDS) :]
        if len(values) < 8:
            self.baseline = 0.0
            self.threshold = 80.0
        else:
            self.baseline = float(median(values))
            deviations = [abs(value - self.baseline) for value in values]
            mad = float(median(deviations))
            std = float(pstdev(values)) if len(values) > 1 else 0.0
            p95 = float(percentile(deviations, 95.0))
            self.threshold = max(35.0, mad * 8.0, std * 4.0, p95 * 2.2)

        self.current_metric = abs(self.current_signal - self.baseline)
        self.calibrated = True
        self.above_threshold = False
        self.pending_single_time = None
        self.events.appendleft(f"calibrated threshold {self.threshold:.1f} uV")

    def _register_blink(self, timestamp: float) -> List[str]:
        if (
            self.pending_single_time is not None
            and timestamp - self.pending_single_time <= DOUBLE_BLINK_SECONDS
        ):
            self.pending_single_time = None
            self.last_action = "double blink: down"
            self.last_action_time = timestamp
            self.events.appendleft("double blink/jaw -> paddle down")
            return ["down"]

        self.pending_single_time = timestamp
        self.events.appendleft("blink/jaw detected")
        return []


def reset_ball(direction: Optional[int] = None) -> Tuple[pygame.Rect, pygame.Vector2]:
    if direction is None:
        direction = random.choice([-1, 1])
    ball = pygame.Rect(0, 0, BALL_SIZE, BALL_SIZE)
    ball.center = (WIDTH // 2, HEIGHT // 2)
    speed_x = direction * random.uniform(260.0, 330.0)
    speed_y = random.choice([-1, 1]) * random.uniform(120.0, 210.0)
    return ball, pygame.Vector2(speed_x, speed_y)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(text, True, color), pos)


def draw_graph(
    surface: pygame.Surface,
    rect: pygame.Rect,
    detector: BlinkDetector,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, PANEL, rect, border_radius=6)
    pygame.draw.rect(surface, LINE, rect, 1, border_radius=6)

    history = list(detector.metric_history)
    if len(history) >= 2:
        max_value = max(max(history), detector.threshold * 1.25, 1.0)
        points = []
        for idx, value in enumerate(history):
            x = rect.left + 8 + idx * (rect.width - 16) / max(1, len(history) - 1)
            y = rect.bottom - 22 - clamp(value / max_value, 0.0, 1.0) * (
                rect.height - 44
            )
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(surface, ACCENT, False, points, 2)

        if detector.threshold > 0.0:
            threshold_y = rect.bottom - 22 - clamp(
                detector.threshold / max_value, 0.0, 1.0
            ) * (rect.height - 44)
            pygame.draw.line(
                surface,
                WARN,
                (rect.left + 8, threshold_y),
                (rect.right - 8, threshold_y),
                1,
            )

    draw_text(surface, font, "blink/jaw metric", (rect.left + 10, rect.top + 8), MUTED)


def draw_hud(
    surface: pygame.Surface,
    detector: BlinkDetector,
    reader: BrainFlowEEG,
    small_font: pygame.font.Font,
    font: pygame.font.Font,
    score: Tuple[int, int],
) -> None:
    panel = pygame.Rect(18, 18, 438, 180)
    pygame.draw.rect(surface, PANEL, panel, border_radius=6)
    pygame.draw.rect(surface, LINE, panel, 1, border_radius=6)

    status_color = ACCENT if reader.ok else WARN
    draw_text(surface, font, "MindPong", (34, 30), TEXT)
    draw_text(surface, small_font, reader.status[:68], (34, 58), status_color)

    pair_a, pair_b = reader.control_pair
    draw_text(
        surface,
        small_font,
        f"Visualizer profile: IMU, active 1-4, off 5-8, gain {EXG_GAIN}",
        (34, 82),
        MUTED,
    )

    y = 106
    if reader.ok and not detector.calibrated:
        elapsed = CALIBRATION_SECONDS - detector.calibration_remaining()
        draw_text(
            surface,
            small_font,
            f"calibrating {elapsed:0.1f}/{CALIBRATION_SECONDS:0.1f}s",
            (34, y),
            WARN,
        )
    elif reader.ok:
        draw_text(surface, small_font, "EEG active", (34, y), ACCENT)
    else:
        draw_text(surface, small_font, "keyboard-only active", (34, y), WARN)

    y += 24
    draw_text(
        surface,
        small_font,
        f"Ch{pair_a} {detector.current_ch1:8.1f} uV   Ch{pair_b} {detector.current_ch2:8.1f} uV",
        (34, y),
        TEXT,
    )
    y += 22
    draw_text(
        surface,
        small_font,
        f"raw avg {detector.current_signal:8.1f} uV   metric {detector.current_metric:7.1f}",
        (34, y),
        TEXT,
    )
    y += 22
    threshold = "calibrating" if not detector.calibrated and reader.ok else f"{detector.threshold:.1f} uV"
    draw_text(
        surface,
        small_font,
        f"baseline {detector.baseline:8.1f} uV   threshold {threshold}",
        (34, y),
        MUTED,
    )

    score_text = font.render(f"{score[0]} : {score[1]}", True, TEXT)
    surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 24))

    events_panel = pygame.Rect(WIDTH - 326, 18, 308, 180)
    pygame.draw.rect(surface, PANEL, events_panel, border_radius=6)
    pygame.draw.rect(surface, LINE, events_panel, 1, border_radius=6)
    draw_text(surface, font, "Events", (events_panel.left + 14, events_panel.top + 12), TEXT)
    if detector.events:
        for idx, event in enumerate(list(detector.events)[:5]):
            color = ACCENT if "->" in event else MUTED
            draw_text(
                surface,
                small_font,
                event[:34],
                (events_panel.left + 14, events_panel.top + 44 + idx * 24),
                color,
            )
    else:
        draw_text(
            surface,
            small_font,
            "waiting for input",
            (events_panel.left + 14, events_panel.top + 48),
            MUTED,
        )

    graph_rect = pygame.Rect(18, HEIGHT - 116, WIDTH - 36, 94)
    draw_graph(surface, graph_rect, detector, small_font)


def run_game() -> int:
    pygame.init()
    pygame.display.set_caption("MindPong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 15)

    reader = BrainFlowEEG(SERIAL_PORT)
    detector = BlinkDetector(reader.sample_rate)

    player = pygame.Rect(42, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
    ai = pygame.Rect(WIDTH - 58, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
    ball, velocity = reset_ball()
    score = [0, 0]

    try:
        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        detector.reset()
                        detector.events.appendleft("recalibration started")
                    elif event.key == pygame.K_c:
                        if reader.cycle_control_pair():
                            detector.reset()
                            pair = reader.control_pair
                            detector.events.appendleft(
                                f"using Ch{pair[0]}/Ch{pair[1]}; recalibrating"
                            )
                        else:
                            detector.events.appendleft("no alternate EXG pair")

            frames = reader.read_frames()
            actions = detector.update(frames)
            for action in actions:
                if action == "up":
                    player.y -= BLINK_STEP
                elif action == "down":
                    player.y += BLINK_STEP

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                player.y -= int(PADDLE_SPEED * dt)
            if keys[pygame.K_s]:
                player.y += int(PADDLE_SPEED * dt)
            player.y = int(clamp(player.y, 0, HEIGHT - player.height))

            ai_target = ball.centery - ai.height / 2
            if ai.y + ai.height / 2 < ai_target:
                ai.y += int(AI_SPEED * dt)
            elif ai.y + ai.height / 2 > ai_target:
                ai.y -= int(AI_SPEED * dt)
            ai.y = int(clamp(ai.y, 0, HEIGHT - ai.height))

            ball.x += int(velocity.x * dt)
            ball.y += int(velocity.y * dt)

            if ball.top <= 0:
                ball.top = 0
                velocity.y *= -1
            elif ball.bottom >= HEIGHT:
                ball.bottom = HEIGHT
                velocity.y *= -1

            if ball.colliderect(player) and velocity.x < 0:
                ball.left = player.right
                offset = (ball.centery - player.centery) / (player.height / 2)
                velocity.x = abs(velocity.x) * 1.04
                velocity.y = 290.0 * clamp(offset, -1.0, 1.0)
            elif ball.colliderect(ai) and velocity.x > 0:
                ball.right = ai.left
                offset = (ball.centery - ai.centery) / (ai.height / 2)
                velocity.x = -abs(velocity.x) * 1.04
                velocity.y = 290.0 * clamp(offset, -1.0, 1.0)

            if ball.right < 0:
                score[1] += 1
                ball, velocity = reset_ball(direction=1)
            elif ball.left > WIDTH:
                score[0] += 1
                ball, velocity = reset_ball(direction=-1)

            screen.fill(BG)
            pygame.draw.line(screen, (33, 41, 52), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 1)
            pygame.draw.rect(screen, PLAYER, player, border_radius=3)
            pygame.draw.rect(screen, OPPONENT, ai, border_radius=3)
            pygame.draw.ellipse(screen, TEXT, ball)

            draw_hud(screen, detector, reader, small_font, font, (score[0], score[1]))
            pygame.display.flip()
    finally:
        reader.shutdown()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(run_game())
