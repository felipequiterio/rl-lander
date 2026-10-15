"""Procedural sound effects for the viewer (no audio files: everything is synthesised
with numpy at startup, in well under a second).

Two kinds of sound:

- **Loops** ("engine", "rcs") play continuously on reserved channels; their volume
  follows a level in [0, 1] that the scene reports every frame (e.g. throttle).
  They're generated with FFT-shaped noise, which is periodic by construction, so
  they loop without clicks.
- **One-shots** ("touchdown", "success", "explosion", "fail", "chime") play once.

Every sound can be panned left/right (-1..1) to follow the lander. If no audio
device is available the viewer simply runs silent.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

LOOPS = ("engine", "rcs")
SMOOTHING = 0.3  # per-frame approach to the target loop volume; avoids zipper noise


def _normalize(x: np.ndarray, peak: float) -> np.ndarray:
    return x * (peak / (np.abs(x).max() + 1e-9))


def _pan(volume: float, pan: float) -> tuple[float, float]:
    """Equal-power pan: -1 = hard left, +1 = hard right."""
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return volume * math.cos(a), volume * math.sin(a)


class Synth:
    def __init__(self, sr: int, seed: int = 0):
        self.sr = sr
        self.rng = np.random.default_rng(seed)

    def t(self, seconds: float) -> np.ndarray:
        return np.arange(int(seconds * self.sr)) / self.sr

    def noise(self, seconds: float, lo: float, hi: float, tilt: float = 0.0) -> np.ndarray:
        """Band-limited noise shaped in the frequency domain (periodic, so loopable).
        tilt > 0 tilts the spectrum towards the lows (1 ~ pink, 2 ~ brown)."""
        n = int(seconds * self.sr)
        spec = np.fft.rfft(self.rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / self.sr)
        f[0] = 1.0
        band = 1 / (1 + (f / hi) ** 4) * (1 - 1 / (1 + (f / lo) ** 4)) * f ** (-tilt / 2)
        x = np.fft.irfft(spec * band, n)
        return x / (np.abs(x).max() + 1e-9)

    def sweep(self, seconds: float, f0: float, f1: float, decay: float) -> np.ndarray:
        """Sine whose pitch glides exponentially f0 -> f1, with exponential decay."""
        t = self.t(seconds)
        freq = f1 + (f0 - f1) * np.exp(-t / (seconds / 4))
        phase = 2 * np.pi * np.cumsum(freq) / self.sr
        return np.sin(phase) * np.exp(-t / decay)

    def bell(self, seconds: float, freq: float, decay: float = 0.35) -> np.ndarray:
        t = self.t(seconds)
        tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2.01 * t) + 0.12 * np.sin(2 * np.pi * freq * 3.98 * t)
        return tone * np.minimum(1, t / 0.004) * np.exp(-t / decay)

    @staticmethod
    def place(total: np.ndarray, x: np.ndarray, at: float, sr: int) -> None:
        i = int(at * sr)
        n = min(len(x), len(total) - i)
        total[i : i + n] += x[:n]

    # --- the sounds --------------------------------------------------------------

    def engine(self) -> np.ndarray:
        d = 2.0
        t = self.t(d)
        rumble = self.noise(d, 20, 160, tilt=1.0)
        roar = self.noise(d, 140, 1600, tilt=1.0)
        hiss = self.noise(d, 1500, 6000)
        flutter = 0.75 + 0.25 * self.noise(d, 2, 9)  # slow periodic wobble so it doesn't sound static
        hum = np.sin(2 * np.pi * 55 * t)  # 110 whole cycles in 2 s: loops seamlessly
        return (0.95 * rumble + 0.4 * roar * flutter + 0.06 * hiss * flutter + 0.15 * hum) * 0.6

    def rcs(self) -> np.ndarray:
        d = 1.0
        return 0.7 * self.noise(d, 1800, 9000) + 0.3 * self.noise(d, 500, 1800)

    def touchdown(self) -> np.ndarray:
        d = 0.6
        t = self.t(d)
        thud = self.sweep(d, 95, 42, 0.13)
        dirt = self.noise(d, 30, 500) * np.exp(-t / 0.05)
        clank = sum(np.sin(2 * np.pi * f * t) for f in (523, 811, 1307, 1789)) / 4 * np.exp(-t / 0.07)
        return _normalize(0.9 * thud + 0.5 * dirt + 0.18 * clank, 0.85)

    def success(self) -> np.ndarray:
        out = np.zeros(int(1.4 * self.sr))
        self.place(out, self.touchdown(), 0.0, self.sr)
        for i, f in enumerate((659.3, 830.6, 987.8)):  # E major arpeggio
            self.place(out, 0.22 * self.bell(0.9, f), 0.22 + 0.11 * i, self.sr)
        return _normalize(out, 0.85)

    def explosion(self) -> np.ndarray:
        d = 2.6
        t = self.t(d)
        attack = np.minimum(1, t / 0.008)
        boom = self.sweep(d, 75, 28, 0.55)
        body = self.noise(d, 25, 900, tilt=1.0) * np.exp(-t / 0.4)
        tail = self.noise(d, 20, 220, tilt=1.0) * np.exp(-t / 1.1)
        crack = self.noise(d, 800, 7000) * np.exp(-t / 0.05)
        # crackle: sparse, decaying bursts of bright noise
        crackle = np.zeros_like(t)
        burst = self.noise(0.02, 1500, 8000) * np.exp(-self.t(0.02) / 0.004)
        for at in np.sort(self.rng.exponential(0.45, 40)):
            if at < d - 0.05:
                self.place(crackle, burst * self.rng.uniform(0.2, 1.0) * math.exp(-at / 0.8), at, self.sr)
        mix = attack * (0.9 * boom + 0.9 * body + 0.6 * tail + 0.35 * crack) + 0.25 * crackle
        return _normalize(np.tanh(mix * 1.6), 0.95)  # soft-clip for punch

    def fail(self) -> np.ndarray:
        out = np.zeros(int(1.0 * self.sr))
        for i, f in enumerate((392.0, 293.7)):  # G4 -> D4, a falling "nope"
            t = self.t(0.45)
            tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.minimum(1, t / 0.01) * np.exp(-t / 0.18)
            self.place(out, 0.4 * tone, 0.2 * i, self.sr)
        return out

    def chime(self) -> np.ndarray:
        out = np.zeros(int(1.0 * self.sr))
        for i, f in enumerate((1046.5, 1318.5, 1568.0)):  # C6 E6 G6, rising
            self.place(out, 0.25 * self.bell(0.7, f, decay=0.25), 0.09 * i, self.sr)
        return out

    def blip(self) -> np.ndarray:
        t = self.t(0.09)
        return 0.3 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, t / 0.003) * np.exp(-t / 0.03)


class SoundFx:
    """The viewer's sound system. Safe to use when audio is unavailable (it no-ops)."""

    def __init__(self, enabled: bool = True, volume: float = 0.7):
        self.available = False
        self.enabled = enabled
        self.volume = volume
        self.levels = {name: 0.0 for name in LOOPS}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 2, 512)
        except pygame.error as e:
            print(f"sound unavailable ({e}); running silent", flush=True)
            return
        sr, _, channels = pygame.mixer.get_init()
        synth = Synth(sr)
        self.sounds = {
            name: self._to_sound(getattr(synth, name)(), channels)
            for name in ("engine", "rcs", "touchdown", "success", "explosion", "fail", "chime", "blip")
        }
        pygame.mixer.set_num_channels(12)
        pygame.mixer.set_reserved(len(LOOPS))
        self.loop_channels = {name: pygame.mixer.Channel(i) for i, name in enumerate(LOOPS)}
        for name, ch in self.loop_channels.items():
            ch.set_volume(0.0, 0.0)
            ch.play(self.sounds[name], loops=-1)
        self.available = True

    @staticmethod
    def _to_sound(mono: np.ndarray, channels: int) -> pygame.mixer.Sound:
        pcm = (np.clip(mono, -1, 1) * 32000).astype(np.int16)
        return pygame.sndarray.make_sound(np.ascontiguousarray(np.repeat(pcm[:, None], channels, axis=1)))

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            self.silence()
        elif self.available:
            self.sounds["blip"].play()
        return self.enabled

    def update(self, levels: dict[str, float], pan: float = 0.0) -> None:
        """Called every frame with the scene's loop levels (missing loops fade out)."""
        if not self.available:
            return
        for name, ch in self.loop_channels.items():
            target = levels.get(name, 0.0) if self.enabled else 0.0
            self.levels[name] += (target - self.levels[name]) * SMOOTHING
            ch.set_volume(*_pan(self.levels[name] * self.volume, pan))

    def silence(self) -> None:
        if not self.available:
            return
        for name, ch in self.loop_channels.items():
            self.levels[name] = 0.0
            ch.set_volume(0.0, 0.0)

    def play(self, name: str | None, pan: float = 0.0) -> None:
        if not (self.available and self.enabled and name):
            return
        ch = pygame.mixer.find_channel(True)  # True: steal the oldest if all are busy
        if ch is not None:
            ch.play(self.sounds[name])
            ch.set_volume(*_pan(self.volume, pan * 0.7))
