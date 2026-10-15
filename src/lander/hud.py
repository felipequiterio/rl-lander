"""Viewer overlay: model card, telemetry gauges, episode strip, banners, toasts.

Env-agnostic: env-specific numbers arrive as a list of ``Gauge`` from the scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

TONES = {
    "normal": (222, 228, 238),
    "dim": (138, 148, 168),
    "good": (110, 231, 160),
    "warn": (250, 204, 90),
    "bad": (255, 110, 100),
    "accent": (120, 200, 255),
}
PANEL_BG = (8, 10, 18)
MARGIN = 16


@dataclass
class Gauge:
    label: str
    text: str
    frac: float | None = None  # if set, draw a 0..1 bar under the row
    tone: str = "normal"


def _font(names: str, size: int, bold: bool = False) -> pygame.font.Font:
    # SysFont takes a comma-separated preference list and falls back to pygame's default.
    return pygame.font.SysFont(names, size, bold=bold)


class Hud:
    def __init__(self) -> None:
        sans = "ubuntu,segoeui,dejavusans,helveticaneue,arial"
        mono = "ubuntumono,consolas,dejavusansmono,menlo,couriernew"
        self.f_label = _font(sans, 12, bold=True)
        self.f_body = _font(sans, 15)
        self.f_title = _font(sans, 19, bold=True)
        self.f_mono = _font(mono, 17)
        self.f_banner = _font(sans, 58, bold=True)
        self.f_sub = _font(sans, 21)
        self.toast_text: str | None = None
        self.toast_left = 0.0

    # --- primitives ---------------------------------------------------------

    def panel(self, surf: pygame.Surface, rect: pygame.Rect, alpha: int = 175) -> None:
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(box, (*PANEL_BG, alpha), box.get_rect(), border_radius=10)
        pygame.draw.rect(box, (255, 255, 255, 26), box.get_rect(), width=1, border_radius=10)
        surf.blit(box, rect)

    def text(self, surf, font, s, pos, tone="normal", anchor="topleft", alpha=255) -> pygame.Rect:
        img = font.render(s, True, TONES.get(tone, tone) if isinstance(tone, str) else tone)
        if alpha < 255:
            img.set_alpha(alpha)
        rect = img.get_rect(**{anchor: pos})
        surf.blit(img, rect)
        return rect

    TRACKING = 1.5

    def label_width(self, s: str) -> int:
        return int(sum(self.f_label.size(ch)[0] for ch in s.upper()) + self.TRACKING * max(len(s) - 1, 0))

    def label(self, surf, s, pos, tone="dim", anchor="topleft") -> pygame.Rect:
        """Small uppercase label with a little letter spacing."""
        color = TONES.get(tone, tone) if isinstance(tone, str) else tone
        glyphs = [self.f_label.render(ch, True, color) for ch in s.upper()]
        rect = pygame.Rect(0, 0, self.label_width(s), self.f_label.get_height())
        setattr(rect, anchor, pos)
        x = float(rect.x)
        for g in glyphs:
            surf.blit(g, (int(x), rect.y))
            x += g.get_width() + self.TRACKING
        return rect

    @staticmethod
    def _fit(font: pygame.font.Font, s: str, width: int) -> str:
        if font.size(s)[0] <= width:
            return s
        while s and font.size(s + "...")[0] > width:
            s = s[:-1]
        return s + "..."

    # --- widgets ------------------------------------------------------------

    def model_card(self, surf, mode: str, record: dict | None, pending: dict | None) -> pygame.Rect:
        w = 340
        rows = 4 if pending else 3
        rect = pygame.Rect(MARGIN, MARGIN, w, 34 + rows * 24)
        self.panel(surf, rect)
        x, y = rect.x + 14, rect.y + 12
        pygame.draw.circle(surf, TONES["accent"], (x + 4, y + 7), 4)
        self.label(surf, mode, (x + 14, y))
        if record is None:
            return rect
        y += 20
        self.text(surf, self.f_title, self._fit(self.f_title, record["run_name"], w - 28), (x, y))
        y += 26
        step = record.get("timestep")
        self.text(surf, self.f_body, f"trained {step:,} steps" if step is not None else "training step unknown", (x, y), "dim")
        y += 22
        if record.get("mean_reward") is not None:
            n = record.get("n_eval_episodes")
            s = f"eval {record['mean_reward']:.1f} ± {record.get('std_reward', 0):.1f}" + (f"  ({n} ep)" if n else "")
            if record.get("success_rate") is not None:
                s += f"  {record['success_rate']:.0%} ok"
        else:
            s = "eval score unknown"
        self.text(surf, self.f_mono, s, (x, y))
        if pending:
            y += 24
            self.text(surf, self.f_body, f"new champion {pending['mean_reward']:.1f} loads next episode", (x, y), "accent")
        return rect

    def gauges(self, surf, gauges: list[Gauge]) -> None:
        if not gauges:
            return
        w = 270
        h = 36 + sum(34 if g.frac is not None else 24 for g in gauges)
        rect = pygame.Rect(surf.get_width() - MARGIN - w, MARGIN, w, h)
        self.panel(surf, rect)
        x0, x1, y = rect.x + 14, rect.right - 14, rect.y + 12
        self.label(surf, "telemetry", (x0, y))
        y += 22
        for g in gauges:
            self.label(surf, g.label, (x0, y + 4))
            self.text(surf, self.f_mono, g.text, (x1, y), g.tone, anchor="topright")
            y += 24
            if g.frac is not None:
                track = pygame.Rect(x0, y, x1 - x0, 4)
                pygame.draw.rect(surf, (48, 54, 70), track, border_radius=2)
                if g.frac > 0:
                    fill = track.copy()
                    fill.width = max(4, int(track.width * min(g.frac, 1.0)))
                    pygame.draw.rect(surf, TONES.get(g.tone, TONES["normal"]), fill, border_radius=2)
                y += 10

    def episode_strip(self, surf, episode: int, seconds: float, reward: float, tally: list[tuple[str, int, str]]) -> None:
        parts = [
            ("label", "episode", "dim"), (self.f_mono, f" {episode}", "normal"),
            ("label", "time", "dim"), (self.f_mono, f" {seconds:5.1f}s", "normal"),
            ("label", "reward", "dim"), (self.f_mono, f" {reward:7.1f}", "good" if reward > 0 else "normal"),
        ]
        chips = [(f"{n} {name}", tone) for name, n, tone in tally]
        gap = 22
        widths = [self.label_width(s) + (gap if i else 0) if f == "label" else f.size(s)[0] for i, (f, s, _) in enumerate(parts)]
        chip_w = sum(self.f_body.size(s)[0] + 34 for s, _ in chips)
        w = sum(widths) + (chip_w + 20 if chips else 0) + 32
        rect = pygame.Rect(0, 0, w, 40)
        rect.midbottom = (surf.get_width() // 2, surf.get_height() - MARGIN)
        self.panel(surf, rect)
        x, cy = rect.x + 16, rect.centery
        for i, ((font, s, tone), fw) in enumerate(zip(parts, widths)):
            if font == "label":
                self.label(surf, s, (x + (gap if i else 0), cy + 1), tone, anchor="midleft")
            else:
                self.text(surf, font, s, (x, cy), tone, anchor="midleft")
            x += fw
        if chips:
            x += 20
            pygame.draw.line(surf, (70, 76, 92), (x - 10, rect.y + 9), (x - 10, rect.bottom - 9))
            for s, tone in chips:
                pygame.draw.circle(surf, TONES[tone], (x + 6, cy), 5)
                r = self.text(surf, self.f_body, s, (x + 16, cy), "normal", anchor="midleft")
                x = r.right + 18

    def banner(self, surf, title: str, subtitle: str, tone: str, fade: float) -> None:
        if fade <= 0:
            return
        w, cy = surf.get_width(), int(surf.get_height() * 0.36)
        band = pygame.Surface((w, 130), pygame.SRCALPHA)
        for i in range(130):  # soft-edged dark band
            a = int(120 * fade * (1 - abs(i - 65) / 65) ** 0.6)
            pygame.draw.line(band, (*PANEL_BG, a), (0, i), (w, i))
        surf.blit(band, (0, cy - 65))
        a = int(255 * fade)
        self.text(surf, self.f_banner, title, (w // 2 + 2, cy - 12 + 2), (0, 0, 0), anchor="center", alpha=a // 2)
        self.text(surf, self.f_banner, title, (w // 2, cy - 12), tone, anchor="center", alpha=a)
        if subtitle:
            self.text(surf, self.f_sub, subtitle, (w // 2, cy + 34), "normal", anchor="center", alpha=a)

    def show_toast(self, text: str, seconds: float = 4.0) -> None:
        self.toast_text, self.toast_left = text, seconds

    def toast(self, surf, below: pygame.Rect, dt: float) -> None:
        if not self.toast_text or self.toast_left <= 0:
            return
        self.toast_left -= dt
        fade = min(1.0, self.toast_left / 0.6)
        if fade <= 0:
            return
        img = self.f_body.render(self.toast_text, True, TONES["accent"])
        rect = pygame.Rect(below.x, below.bottom + 10, img.get_width() + 28, 34)
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(box, (20, 50, 80, int(200 * fade)), box.get_rect(), border_radius=17)
        pygame.draw.rect(box, (*TONES["accent"], int(120 * fade)), box.get_rect(), width=1, border_radius=17)
        surf.blit(box, rect)
        img.set_alpha(int(255 * fade))
        surf.blit(img, img.get_rect(midleft=(rect.x + 14, rect.centery)))

    def sound_hint(self, surf, enabled: bool, available: bool) -> None:
        if not available:
            text, tone = "no audio device", "dim"
        else:
            text, tone = ("sound on", "dim") if enabled else ("sound off", "warn")
        text = f"{text}  [M]"
        w, h = surf.get_size()
        icon = 22 if available else 0
        rect = pygame.Rect(0, 0, self.label_width(text) + icon + 24, 28)
        rect.bottomright = (w - MARGIN, h - MARGIN - 6)
        self.panel(surf, rect, alpha=150)
        self.label(surf, text, (rect.right - 12, rect.centery + 1), tone, anchor="midright")
        if available:  # tiny speaker glyph
            c = TONES[tone]
            x, y = rect.x + 12, rect.centery
            pygame.draw.polygon(surf, c, [(x, y - 2), (x + 3, y - 2), (x + 7, y - 6), (x + 7, y + 6), (x + 3, y + 2), (x, y + 2)])
            if enabled:
                pygame.draw.arc(surf, c, (x + 4, y - 5, 8, 10), -1.0, 1.0, 1)
            else:
                pygame.draw.line(surf, c, (x + 10, y - 3), (x + 15, y + 2), 2)
                pygame.draw.line(surf, c, (x + 10, y + 2), (x + 15, y - 3), 2)

    def waiting(self, surf, path: str, t: float) -> None:
        w, h = surf.get_size()
        surf.fill((6, 8, 16))
        dots = "." * (1 + int(t * 2) % 3)
        self.text(surf, self.f_sub, f"Waiting for the first champion{dots}", (w // 2, h // 2 - 12), "normal", anchor="center")
        self.text(surf, self.f_body, path, (w // 2, h // 2 + 18), "dim", anchor="center")
