"""
Human Player Module

This module contains a human player implementation that allows users to participate
in tournaments by making their own decisions during gameplay.
"""

import axelrod as axl

class HumanPlayer(axl.Player):
    """
    A player that gets moves from the user.

    This is a player that is controlled by a human. The `strategy` method will
    be passed the opponent.

    Names:

    - Human: [Axelrod2012]_
    """

    name = "Human"
    classifier = {
        "memory_depth": float("inf"),  # Long memory
        "stochastic": True,
        "makes_use_of": set(),
        "long_run_time": True,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self, name: str = "You") -> None:
        """
        Initialize the player.
        """
        super().__init__()
        self.name = name
        self._move = axl.Action.C  # Default move

    def set_move(self, move: axl.Action):
        """Sets the next move for the player."""
        self._move = move

    def strategy(self, opponent: axl.Player) -> axl.Action:
        """Returns the move set by the UI."""
        return self._move
