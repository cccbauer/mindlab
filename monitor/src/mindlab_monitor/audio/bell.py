"""Start/end meditation bell via flet_audio.Audio.

`Audio` still ships as the separate `flet-audio` add-on package (not merged
into flet core as of 0.86.5), imported as `flet_audio`, not `flet.Audio`.

Uses two Audio control instances per sound, alternating which one plays.
Flet/Android has a reported quirk where replaying the exact same Audio
control's source doesn't reliably restart playback (flet-dev Discussion
#2925) — alternating instances sidesteps needing to seek(0)+replay the same
control twice in one session (bell_start plays once, bell_end plays once,
but a future "interval bell" feature would hit this directly).
"""

from __future__ import annotations

import flet as ft
from flet_audio import Audio


class BellPlayer:
    def __init__(self, page: ft.Page, start_asset: str, end_asset: str) -> None:
        self._start_players = [Audio(src=start_asset), Audio(src=start_asset)]
        self._end_players = [Audio(src=end_asset), Audio(src=end_asset)]
        self._start_idx = 0
        self._end_idx = 0
        page.overlay.extend(self._start_players + self._end_players)
        page.update()

    def play_start(self) -> None:
        player = self._start_players[self._start_idx % len(self._start_players)]
        self._start_idx += 1
        player.play()

    def play_end(self) -> None:
        player = self._end_players[self._end_idx % len(self._end_players)]
        self._end_idx += 1
        player.play()
