"""Interactive exhibit GUI: drag the source, listen and inspect cues.

A top-view (XY) canvas shows the listener head and the movable source.
Dragging the source or using the yaw slider re-evaluates the geometry on
every block boundary through the real renderer -- there is no volume-pan
shortcut. Preset buttons jump to the verification positions required by the
exhibit: symmetric left/right and directly behind.

On a headless machine the GUI cannot open a window; :class:`HeadlessGUI`
exposes the same event handlers programmatically so the interaction logic
stays testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import relative_direction
from .realtime import RealtimeBinauralPlayer

LISTENER_POS = np.array([0.0, 0.0, 0.0])
CANVAS_LIMIT = 2.0


@dataclass
class GUIState:
    source_position: np.ndarray
    yaw_deg: float
    pitch_deg: float
    azimuth_deg: float
    elevation_deg: float
    distance_m: float


class BinauralGUI:
    def __init__(self, player: RealtimeBinauralPlayer,
                 initial_source=(1.2, 0.0, 0.0)):
        self.player = player
        self.renderer = player.renderer
        self.state = GUIState(
            source_position=np.asarray(initial_source, dtype=float),
            yaw_deg=0.0, pitch_deg=0.0,
            azimuth_deg=0.0, elevation_deg=0.0, distance_m=1.2)
        self._figure = None
        self._dragging = False
        self._refresh_geometry()

    # ------------------------------------------------------------------
    def move_source(self, position: tuple[float, ...], z: float | None = None) -> GUIState:
        pos = np.asarray(position, dtype=float)
        if pos.size == 2:
            x, y = float(pos[0]), float(pos[1])
            z_val = 0.0 if z is None else float(z)
        elif pos.size == 3:
            x, y, z_val = (float(pos[0]), float(pos[1]),
                           float(pos[2]) if z is None else float(z))
        else:
            raise ValueError("position must be 2D or 3D")
        x = float(np.clip(x, -CANVAS_LIMIT, CANVAS_LIMIT))
        y = float(np.clip(y, -CANVAS_LIMIT, CANVAS_LIMIT))
        self.state.source_position = np.array([x, y, z_val])
        self._refresh_geometry()
        return self.state

    def set_yaw(self, yaw_deg: float) -> GUIState:
        self.state.yaw_deg = float(yaw_deg)
        self._refresh_geometry()
        return self.state

    def set_pitch(self, pitch_deg: float) -> GUIState:
        self.state.pitch_deg = float(pitch_deg)
        self._refresh_geometry()
        return self.state

    def jump_preset(self, name: str) -> GUIState:
        presets = {
            "front": (1.2, 0.0, 0.0),
            "left": (0.0, 1.2, 0.0),
            "right": (0.0, -1.2, 0.0),
            "back": (-1.2, 0.0, 0.0),
        }
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; choose from {list(presets)}")
        return self.move_source(presets[name])

    def _refresh_geometry(self) -> None:
        rel = relative_direction(
            self.state.source_position, LISTENER_POS,
            self.state.yaw_deg, self.state.pitch_deg)
        self.state.azimuth_deg = rel.azimuth_deg
        self.state.elevation_deg = rel.elevation_deg
        self.state.distance_m = rel.distance_m
        self.renderer.set_direction(rel.azimuth_deg, rel.elevation_deg,
                                    rel.distance_m)

    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - requires a display
        import matplotlib

        matplotlib.use("TkAgg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.22)
        ax.set_xlim(-CANVAS_LIMIT, CANVAS_LIMIT)
        ax.set_ylim(-CANVAS_LIMIT, CANVAS_LIMIT)
        ax.set_aspect("equal")
        ax.set_xlabel("X  front (+) / back (-)  [m]")
        ax.set_ylabel("Y  left (+) / right (-)  [m]")
        ax.grid(True, alpha=0.3)

        head = ax.scatter([0], [0], s=900, marker="o", c="#4477aa",
                          zorder=3, label="listener head")
        nose, = ax.plot([0, 0.25], [0, 0], lw=2, c="#223344", zorder=4)
        src, = ax.plot([self.state.source_position[0]],
                       [self.state.source_position[1]],
                       marker="*", ms=22, c="#cc3333", zorder=5,
                       label="source (drag me)")
        info = ax.text(-1.95, 1.78, "", fontfamily="monospace", fontsize=9,
                       va="top", bbox=dict(facecolor="white", alpha=0.8))

        def redraw():
            s = self.state
            src.set_data([s.source_position[0]], [s.source_position[1]])
            yaw = np.radians(s.yaw_deg)
            nose.set_data([0, 0.25 * np.cos(yaw)],
                          [0, 0.25 * np.sin(yaw)])
            info.set_text(
                f"az {s.azimuth_deg:+7.1f} deg\n"
                f"el {s.elevation_deg:+7.1f} deg\n"
                f"r  {s.distance_m:6.2f} m\n"
                f"yaw {s.yaw_deg:+7.1f} deg")
            fig.canvas.draw_idle()

        def on_press(event):
            if event.inaxes == ax and event.xdata is not None:
                self._dragging = True
                self.move_source((event.xdata, event.ydata))
                redraw()

        def on_move(event):
            if self._dragging and event.inaxes == ax and event.xdata is not None:
                self.move_source((event.xdata, event.ydata))
                redraw()

        def on_release(event):
            self._dragging = False

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("button_release_event", on_release)

        ax_yaw = plt.axes([0.15, 0.12, 0.7, 0.03])
        yaw_slider = Slider(ax_yaw, "head yaw", -180, 180, valinit=0.0)

        def on_yaw(value):
            self.set_yaw(value)
            redraw()

        yaw_slider.on_changed(on_yaw)

        preset_axes = [plt.axes([x, 0.04, 0.13, 0.05])
                       for x in (0.08, 0.27, 0.46, 0.65)]
        for axi, label in zip(preset_axes,
                              ("front", "left", "right", "back")):
            btn = Button(axi, label)
            btn.on_clicked(
                lambda _event, name=label: (self.jump_preset(name), redraw()))

        ax_play = plt.axes([0.82, 0.04, 0.13, 0.05])
        playing = {"on": False}
        play_btn = Button(ax_play, "play")

        def toggle(_event):
            if playing["on"]:
                self.player.stop()
                play_btn.label.set_text("play")
                playing["on"] = False
            else:
                backend = self.player.start()
                play_btn.label.set_text(f"stop [{backend}]")
                playing["on"] = True

        play_btn.on_clicked(toggle)
        redraw()
        self._figure = fig
        plt.show()


class HeadlessGUI:
    """Programmable twin of :class:`BinauralGUI` for CI and exhibit tests."""

    def __init__(self, player: RealtimeBinauralPlayer):
        self.gui = BinauralGUI(player)

    def scenario(self, moves: list[tuple[str, object]],
                 blocks_per_move: int = 4) -> np.ndarray:
        chunks = []
        for action, value in moves:
            if action == "source":
                self.gui.move_source(value)
            elif action == "yaw":
                self.gui.set_yaw(value)
            elif action == "preset":
                self.gui.jump_preset(value)
            else:
                raise KeyError(action)
            chunks.append(self.gui.player.simulate_blocks(blocks_per_move))
        return np.concatenate(chunks)
