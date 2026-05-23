"""
Move Ball With Mind: a simple Pygame + BrainFlow EXG focus-ball prototype.

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
    SPACE       Keyboard fallback: simulate focus/activation
    C           Toggle control pair between Ch1/Ch2 and Ch3/Ch4
    R           Recalibrate and reset the ball
    Esc / close Quit

Install:
    pip install pygame brainflow

If BrainFlow is not installed or the board cannot be opened, the game still runs
in keyboard-only mode.
"""

from __future__ import annotations

import math
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
BASELINE_ALPHA = 0.003
ACTIVATION_EMA_ALPHA = 0.12
HISTORY_SECONDS = 6.0
KEYBOARD_FOCUS_TARGET = 1.0

WIDTH = 960
HEIGHT = 540
FPS = 60
BALL_RADIUS = 22
MAX_BALL_SPEED = 430.0
ROLLBACK_SPEED = 42.0
ROLLBACK_BELOW_ACTIVATION = 0.12

ROAD_LEFT = 96
ROAD_RIGHT = 864
ROAD_Y = 298
ROAD_HEIGHT = 122
START_X = ROAD_LEFT + 50
FINISH_X = ROAD_RIGHT - 50

BG = (9, 13, 18)
PANEL = (18, 25, 33)
ROAD = (31, 38, 48)
ROAD_EDGE = (82, 96, 112)
TEXT = (229, 235, 241)
MUTED = (143, 155, 168)
ACCENT = (93, 215, 181)
WARN = (255, 191, 87)
DANGER = (255, 105, 105)
LINE = (70, 83, 98)
BALL = (122, 166, 255)
BALL_HIGHLIGHT = (210, 229, 255)


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
class EXGFrame:
    ch1: float
    ch2: float
    signal: float
    timestamp: float


class BrainFlowEXG:
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
            self._update_status()
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

    def _update_status(self) -> None:
        self.status = (
            f"BrainFlow Knight Board IMU on {self.serial_port}; "
            f"EXG Ch{self.control_pair[0]}/Ch{self.control_pair[1]}, gain {EXG_GAIN}"
        )
        if self.config_warnings:
            self.status += f"; config warnings {len(self.config_warnings)}"

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

        self._update_status()
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

    def read_frames(self) -> List[EXGFrame]:
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
            self.status = f"EXG channel read failed: {exc.__class__.__name__}"
            return []

        now = time.monotonic()
        first_ts = now - (sample_count - 1) / max(1, self.sample_rate)
        frames: List[EXGFrame] = []
        for idx in range(sample_count):
            ch1 = float(ch1_values[idx])
            ch2 = float(ch2_values[idx])
            frames.append(
                EXGFrame(
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


class ActivationMeter:
    """Continuous baseline-normalized EXG activation for ball movement."""

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
        self.activation_metric = 0.0
        self.raw_activation = 0.0
        self.smoothed_activation = 0.0
        self.events: Deque[str] = deque(maxlen=6)
        self.metric_history: Deque[float] = deque(
            maxlen=int(self.sample_rate * HISTORY_SECONDS)
        )
        self.activation_history: Deque[float] = deque(
            maxlen=int(self.sample_rate * HISTORY_SECONDS)
        )

    def calibration_remaining(self) -> float:
        if self.calibrated:
            return 0.0
        return max(0.0, CALIBRATION_SECONDS - (time.monotonic() - self.started_at))

    def make_keyboard_ready(self) -> None:
        if self.calibrated:
            return
        self.baseline = 0.0
        self.threshold = 1.0
        self.activation_metric = 0.0
        self.raw_activation = 0.0
        self.smoothed_activation = 0.0
        self.calibrated = True
        self.events.appendleft("keyboard activation ready")

    def update(
        self,
        frames: Iterable[EXGFrame],
        keyboard_focus: bool,
        keyboard_only: bool,
    ) -> None:
        frame_seen = False

        for frame in frames:
            frame_seen = True
            self.current_ch1 = frame.ch1
            self.current_ch2 = frame.ch2
            self.current_signal = frame.signal

            if not self.calibrated:
                self.calibration_values.append(frame.signal)
                if self.calibration_values:
                    self.baseline = float(median(self.calibration_values))
                self.activation_metric = abs(frame.signal - self.baseline)
                self.metric_history.append(self.activation_metric)
                self.activation_history.append(0.0)
                continue

            self.activation_metric = abs(frame.signal - self.baseline)
            self.raw_activation = clamp(
                self.activation_metric / max(self.threshold, 0.0001),
                0.0,
                1.0,
            )
            self.metric_history.append(self.activation_metric)

            if self.raw_activation < 0.45:
                self.baseline += BASELINE_ALPHA * (frame.signal - self.baseline)

        if not self.calibrated:
            if time.monotonic() - self.started_at >= CALIBRATION_SECONDS:
                if frame_seen:
                    self._finish_calibration()
                elif keyboard_only:
                    self.make_keyboard_ready()
            self._smooth_toward(0.0)
            return

        target_activation = self.raw_activation if frame_seen else 0.0
        if keyboard_focus:
            target_activation = max(target_activation, KEYBOARD_FOCUS_TARGET)

        self._smooth_toward(target_activation)
        self.activation_history.append(self.smoothed_activation)

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

        self.activation_metric = abs(self.current_signal - self.baseline)
        self.raw_activation = 0.0
        self.smoothed_activation = 0.0
        self.calibrated = True
        self.events.appendleft(f"calibrated threshold {self.threshold:.1f} uV")

    def _smooth_toward(self, target: float) -> None:
        target = clamp(target, 0.0, 1.0)
        self.raw_activation = target
        self.smoothed_activation += ACTIVATION_EMA_ALPHA * (
            target - self.smoothed_activation
        )
        self.smoothed_activation = clamp(self.smoothed_activation, 0.0, 1.0)


@dataclass
class FocusBall:
    x: float = float(START_X)
    finished: bool = False

    def reset(self) -> None:
        self.x = float(START_X)
        self.finished = False

    def update(self, dt: float, activation: float) -> float:
        if self.finished:
            return 0.0

        forward_speed = activation * MAX_BALL_SPEED
        rollback = 0.0
        if activation < ROLLBACK_BELOW_ACTIVATION:
            rollback = ROLLBACK_SPEED * (1.0 - activation / ROLLBACK_BELOW_ACTIVATION)

        velocity = forward_speed - rollback
        self.x += velocity * dt
        self.x = clamp(self.x, float(START_X), float(FINISH_X))
        if self.x >= FINISH_X:
            self.finished = True
        return velocity

    def progress(self) -> float:
        return clamp((self.x - START_X) / max(1.0, FINISH_X - START_X), 0.0, 1.0)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(text, True, color), pos)


def draw_power_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    activation: float,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, PANEL, rect, border_radius=6)
    pygame.draw.rect(surface, LINE, rect, 1, border_radius=6)

    inner = rect.inflate(-12, -34)
    inner.y += 18
    pygame.draw.rect(surface, (11, 15, 21), inner, border_radius=4)
    fill = inner.copy()
    fill.width = int(inner.width * clamp(activation, 0.0, 1.0))
    color = ACCENT if activation >= ROLLBACK_BELOW_ACTIVATION else WARN
    if fill.width > 0:
        pygame.draw.rect(surface, color, fill, border_radius=4)
    pygame.draw.rect(surface, LINE, inner, 1, border_radius=4)

    draw_text(surface, font, "power", (rect.left + 12, rect.top + 8), MUTED)
    draw_text(
        surface,
        font,
        f"{activation * 100:5.1f}%",
        (rect.right - 92, rect.top + 8),
        TEXT,
    )


def draw_activation_graph(
    surface: pygame.Surface,
    rect: pygame.Rect,
    meter: ActivationMeter,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, PANEL, rect, border_radius=6)
    pygame.draw.rect(surface, LINE, rect, 1, border_radius=6)

    history = list(meter.activation_history)
    if len(history) >= 2:
        points = []
        for idx, value in enumerate(history):
            x = rect.left + 8 + idx * (rect.width - 16) / max(1, len(history) - 1)
            y = rect.bottom - 18 - clamp(value, 0.0, 1.0) * (rect.height - 40)
            points.append((x, y))
        pygame.draw.lines(surface, ACCENT, False, points, 2)

        gate_y = rect.bottom - 18 - ROLLBACK_BELOW_ACTIVATION * (rect.height - 40)
        pygame.draw.line(
            surface,
            WARN,
            (rect.left + 8, gate_y),
            (rect.right - 8, gate_y),
            1,
        )

    draw_text(surface, font, "smoothed activation", (rect.left + 10, rect.top + 8), MUTED)


def draw_hud(
    surface: pygame.Surface,
    meter: ActivationMeter,
    reader: BrainFlowEXG,
    small_font: pygame.font.Font,
    font: pygame.font.Font,
    velocity: float,
    ball: FocusBall,
) -> None:
    panel = pygame.Rect(18, 18, 452, 202)
    pygame.draw.rect(surface, PANEL, panel, border_radius=6)
    pygame.draw.rect(surface, LINE, panel, 1, border_radius=6)

    status_color = ACCENT if reader.ok else WARN
    draw_text(surface, font, "Move Ball With Mind", (34, 30), TEXT)
    draw_text(surface, small_font, reader.status[:70], (34, 58), status_color)

    pair_a, pair_b = reader.control_pair
    draw_text(
        surface,
        small_font,
        f"IMU profile: active 1-4, off 5-8, gain {EXG_GAIN}, pair Ch{pair_a}/Ch{pair_b}",
        (34, 82),
        MUTED,
    )

    y = 108
    if reader.ok and not meter.calibrated:
        elapsed = CALIBRATION_SECONDS - meter.calibration_remaining()
        draw_text(
            surface,
            small_font,
            f"calibrating baseline {elapsed:0.1f}/{CALIBRATION_SECONDS:0.1f}s",
            (34, y),
            WARN,
        )
    elif reader.ok:
        draw_text(surface, small_font, "EXG active", (34, y), ACCENT)
    else:
        draw_text(surface, small_font, "keyboard-only active", (34, y), WARN)

    y += 24
    draw_text(
        surface,
        small_font,
        f"Ch{pair_a} {meter.current_ch1:8.1f} uV   Ch{pair_b} {meter.current_ch2:8.1f} uV",
        (34, y),
        TEXT,
    )
    y += 22
    draw_text(
        surface,
        small_font,
        f"metric {meter.activation_metric:7.1f}   activation {meter.raw_activation:0.3f}",
        (34, y),
        TEXT,
    )
    y += 22
    threshold = "calibrating" if not meter.calibrated and reader.ok else f"{meter.threshold:.1f} uV"
    draw_text(
        surface,
        small_font,
        f"smooth {meter.smoothed_activation:0.3f}   baseline {meter.baseline:8.1f}   threshold {threshold}",
        (34, y),
        MUTED,
    )
    y += 22
    draw_text(
        surface,
        small_font,
        f"speed {velocity:7.1f} px/s   progress {ball.progress() * 100:5.1f}%",
        (34, y),
        TEXT,
    )

    power_rect = pygame.Rect(WIDTH - 334, 24, 308, 82)
    draw_power_bar(surface, power_rect, meter.smoothed_activation, small_font)

    events_panel = pygame.Rect(WIDTH - 334, 118, 308, 102)
    pygame.draw.rect(surface, PANEL, events_panel, border_radius=6)
    pygame.draw.rect(surface, LINE, events_panel, 1, border_radius=6)
    draw_text(surface, font, "Status", (events_panel.left + 14, events_panel.top + 12), TEXT)
    if meter.events:
        for idx, event in enumerate(list(meter.events)[:2]):
            draw_text(
                surface,
                small_font,
                event[:36],
                (events_panel.left + 14, events_panel.top + 44 + idx * 24),
                ACCENT,
            )
    else:
        draw_text(
            surface,
            small_font,
            "SPACE simulates activation",
            (events_panel.left + 14, events_panel.top + 48),
            MUTED,
        )


def draw_tunnel(surface: pygame.Surface, font: pygame.font.Font, ball: FocusBall) -> None:
    road_rect = pygame.Rect(
        ROAD_LEFT,
        ROAD_Y - ROAD_HEIGHT // 2,
        ROAD_RIGHT - ROAD_LEFT,
        ROAD_HEIGHT,
    )
    pygame.draw.rect(surface, ROAD, road_rect, border_radius=8)
    pygame.draw.rect(surface, ROAD_EDGE, road_rect, 2, border_radius=8)

    for x in range(ROAD_LEFT + 32, ROAD_RIGHT, 64):
        pygame.draw.line(
            surface,
            (48, 58, 70),
            (x, ROAD_Y),
            (x + 26, ROAD_Y),
            2,
        )

    pygame.draw.line(
        surface,
        WARN,
        (START_X, road_rect.top - 18),
        (START_X, road_rect.bottom + 18),
        3,
    )
    pygame.draw.line(
        surface,
        ACCENT,
        (FINISH_X, road_rect.top - 18),
        (FINISH_X, road_rect.bottom + 18),
        4,
    )
    draw_text(surface, font, "START", (START_X - 30, road_rect.bottom + 26), WARN)
    draw_text(surface, font, "FINISH", (FINISH_X - 36, road_rect.bottom + 26), ACCENT)

    ball_center = (int(ball.x), ROAD_Y)
    pygame.draw.circle(surface, BALL, ball_center, BALL_RADIUS)
    pygame.draw.circle(
        surface,
        BALL_HIGHLIGHT,
        (ball_center[0] - 7, ball_center[1] - 8),
        7,
    )

    if ball.finished:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((9, 13, 18, 158))
        surface.blit(overlay, (0, 0))
        big_font = pygame.font.SysFont("consolas", 64, bold=True)
        text = big_font.render("FINISH!", True, ACCENT)
        sub = font.render("Press R to recalibrate and reset", True, TEXT)
        surface.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 74))
        surface.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2 + 6))


def run_game() -> int:
    pygame.init()
    pygame.display.set_caption("Move Ball With Mind")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 15)

    reader = BrainFlowEXG(SERIAL_PORT)
    meter = ActivationMeter(reader.sample_rate)
    if not reader.ok:
        meter.make_keyboard_ready()

    ball = FocusBall()
    velocity = 0.0

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
                        meter.reset()
                        if not reader.ok:
                            meter.make_keyboard_ready()
                        ball.reset()
                        velocity = 0.0
                        meter.events.appendleft("reset")
                    elif event.key == pygame.K_c:
                        if reader.cycle_control_pair():
                            meter.reset()
                            ball.reset()
                            pair = reader.control_pair
                            meter.events.appendleft(
                                f"using Ch{pair[0]}/Ch{pair[1]}; recalibrating"
                            )
                        else:
                            meter.events.appendleft("no alternate EXG pair")

            keys = pygame.key.get_pressed()
            keyboard_focus = bool(keys[pygame.K_SPACE])
            frames = reader.read_frames()
            if not reader.ok and not meter.calibrated:
                meter.make_keyboard_ready()
            meter.update(frames, keyboard_focus, keyboard_only=not reader.ok)

            velocity = ball.update(dt, meter.smoothed_activation)

            screen.fill(BG)
            draw_tunnel(screen, font, ball)
            draw_hud(screen, meter, reader, small_font, font, velocity, ball)
            graph_rect = pygame.Rect(18, HEIGHT - 104, WIDTH - 36, 82)
            draw_activation_graph(screen, graph_rect, meter, small_font)
            pygame.display.flip()
    finally:
        reader.shutdown()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(run_game())
