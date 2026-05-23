# NeuroFlow

NeuroFlow is a small Python/Pygame prototype collection for NeuroPawn EXG/EEG
control experiments through BrainFlow.

The main demo is `moveballwithmind.py`: a continuous focus-ball game where
detected EXG/EEG activation pushes a ball through a tunnel toward a finish line.
The repository also includes `mindpong.py`, an earlier blink/jaw-event Pong
prototype.

## Hardware Profile

The scripts are configured for the NeuroPawn Knight Board IMU on `COM3`,
matching the NeuroPawn EXG Visualizer setup:

- Board type: `NEUROPAWN_KNIGHT_BOARD_IMU`
- EXG channels 1-4 enabled
- EXG channels 5-8 disabled to avoid floating-channel noise
- Gain: `12`
- RLD routed on channels 1-4
- Default control pair: Ch1/Ch2
- Alternate control pair: Ch3/Ch4

Close the NeuroPawn EXG Visualizer before running a script. Only one process can
own the serial port at a time.

## Install

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Focus-Ball Game

```powershell
python moveballwithmind.py
```

Gameplay:

- A ball starts on the left side of a horizontal tunnel.
- Higher detected EXG/EEG activation moves the ball faster to the right.
- Low activation slows the ball and can roll it slowly backward.
- Reaching the finish line shows `FINISH!`.

Controls:

- `SPACE`: keyboard fallback, simulates focus/activation
- `C`: toggle control pair between Ch1/Ch2 and Ch3/Ch4
- `R`: reset ball and recalibrate baseline
- `ESC`: quit

The first 5 seconds are used for baseline calibration. Activation is calculated
as absolute deviation from baseline normalized by the calibrated threshold, then
smoothed with an exponential moving average to reduce jitter.

## Run The MindPong Prototype

```powershell
python mindpong.py
```

MindPong uses threshold events instead of continuous control:

- single blink/jaw event moves the paddle up
- double blink/jaw event moves the paddle down
- `W`/`S` provide keyboard fallback
- `C`, `R`, and `ESC` behave similarly to the focus-ball demo

## If BrainFlow Fails

Both scripts degrade to keyboard-only mode if BrainFlow is unavailable, the board
ID is missing, or `COM3` cannot be opened. This is useful for testing gameplay
without the board connected.

Common checks:

- close the NeuroPawn EXG Visualizer
- confirm the board is actually on `COM3`
- confirm `brainflow` is installed in the active Python environment
- try Ch3/Ch4 with `C` if Ch1/Ch2 are noisy

## Files

- `moveballwithmind.py`: primary continuous activation focus-ball game
- `mindpong.py`: blink/jaw event Pong prototype
- `requirements.txt`: Python runtime dependencies
- `LICENSE`: MIT license
