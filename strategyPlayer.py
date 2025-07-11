#!/usr/bin/env python3
"""
StrategyPlayer class for Axelrod game theory strategies.
"""

import axelrod as axl
from typing import List, Tuple, Optional, Union


class StrategyPlayer:
    """
    A class that wraps Axelrod strategies for easy use in games.
    
    Attributes:
        strategyClass: The Axelrod strategy class
        strategyName: Name of the strategy
        player: The actual Axelrod player instance
        history: List of moves made by this player
        opponentHistory: List of moves made by the opponent
    """
    
    def __init__(self, strategyIdentifier: Union[int, str]):
        """
        Initialize a StrategyPlayer with a strategy.
        
        Args:
            strategyIdentifier (Union[int, str]): Either the strategy number (int) or name (str) from the Axelrod package
        
        Returns:
            None
        
        Raises:
            ValueError: If strategy number is out of range or strategy name not found
            TypeError: If strategyIdentifier is not int or str
        """
        self.strategyClass = self._getStrategyClass(strategyIdentifier)
        self.strategyName = self.strategyClass.__name__
        self.player = self.strategyClass()
        self.history = []
        self.opponentHistory = []
    

    def _getStrategyClass(self, identifier: Union[int, str]):
        """
        Get the strategy class from identifier.
        
        Args:
            identifier (Union[int, str]): Strategy number (int) or name (str)
        
        Returns:
            type: The Axelrod strategy class
        
        Raises:
            ValueError: If strategy number is out of range or strategy name not found
            TypeError: If identifier is not int or str
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(axl.strategies):
                return axl.strategies[identifier]
            else:
                raise ValueError(f"Strategy number {identifier} is out of range. "
                              f"Available strategies: 0-{len(axl.strategies)-1}")
        elif isinstance(identifier, str):
            # Find strategy by name
            for strategyClass in axl.strategies:
                if strategyClass.__name__.lower() == identifier.lower():
                    return strategyClass
            # If exact match not found, try partial match
            for strategyClass in axl.strategies:
                if identifier.lower() in strategyClass.__name__.lower():
                    return strategyClass
            raise ValueError(f"Strategy '{identifier}' not found")
        else:
            raise TypeError("Strategy identifier must be int or str")
    

    def makeDecision(self, opponentHistory: Optional[List[str]] = None) -> str:
        """
        Make a decision based on the opponent's history.
        
        Args:
            opponentHistory (Optional[List[str]], optional): List of opponent's previous moves (C/D). If None or empty, this is the first move. Defaults to None.
        
        Returns:
            str: The strategy's decision: 'C' for Cooperate, 'D' for Defect
        """
        # Update opponent history
        if opponentHistory is not None:
            self.opponentHistory = opponentHistory.copy()
        else:
            self.opponentHistory = []
        
        # Create a mock opponent player with the history
        opponentPlayer = axl.Player()
        opponentPlayer.history = self.opponentHistory
        
        # Get the strategy's decision
        decision = self.player.strategy(self.player, opponentPlayer)
        
        # Update this player's history
        self.history.append(decision)
        
        return decision
    

    def reset(self):
        """
        Reset the player's history and opponent history.
        
        Args:
            None
        
        Returns:
            None
        """
        self.history = []
        self.opponentHistory = []
        # Recreate the player to reset internal state
        self.player = self.strategyClass()
    

    def getScore(self, opponentHistory: List[str]) -> int:
        """
        Calculate the score for this player based on game history.
        
        Args:
            opponentHistory (List[str]): List of opponent's moves (C/D)
        
        Returns:
            int: Total score for this player based on Prisoner's Dilemma scoring
        
        Raises:
            ValueError: If history lengths don't match between player and opponent
        """
        if len(self.history) != len(opponentHistory):
            raise ValueError("History lengths don't match")
        
        score = 0
        for myMove, opponentMove in zip(self.history, opponentHistory):
            if myMove == 'C' and opponentMove == 'C':
                score += 3
            elif myMove == 'C' and opponentMove == 'D':
                score += 0
            elif myMove == 'D' and opponentMove == 'C':
                score += 5
            else:  # D vs D
                score += 1
        
        return score
    

    def __str__(self) -> str:
        """
        String representation of the StrategyPlayer.
        
        Args:
            None
        
        Returns:
            str: String representation showing the strategy name
        """
        return f"StrategyPlayer({self.strategyName})"
    
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the StrategyPlayer.
        
        Args:
            None
        
        Returns:
            str: Detailed string representation showing the strategy name
        """
        return f"StrategyPlayer('{self.strategyName}')"


def getAvailableStrategies() -> List[Tuple[int, str]]:
    """
    Get a list of available strategies with their indices.
    
    Args:
        None
    
    Returns:
        List[Tuple[int, str]]: List of tuples containing (strategy_index, strategy_name)
    """
    strategies = []

    for i, strategyClass in enumerate(axl.strategies):
        strategyName = strategyClass.__name__
        strategies.append((i, strategyName))
    
    return strategies


def displayStrategies(strategies: List[Tuple[int, str]]) -> None:
    """
    Display available strategies in a numbered list.
    
    Args:
        strategies (List[Tuple[int, str]]): List of tuples containing (strategy_index, strategy_name)
    
    Returns:
        None
    """
    print("\nAvailable strategies:")
    print("-" * 50)
    for i, (idx, name) in enumerate(strategies, 1):
        print(f"{i:2d}. {name}")
    print("-" * 50)


def getUserChoice(strategies: List[Tuple[int, str]]) -> int:
    """
    Get user's choice of strategy from the displayed list.
    
    Args:
        strategies (List[Tuple[int, str]]): List of tuples containing (strategy_index, strategy_name)
    
    Returns:
        int: The selected strategy index from the original Axelrod strategies list
    
    Note:
        This function handles input validation and will keep prompting until a valid choice is made.
    """
    while True:
        try:
            choice = input(f"\nEnter strategy number (1-{len(strategies)}): ")
            choiceNum = int(choice)
            if 1 <= choiceNum <= len(strategies):
                return strategies[choiceNum - 1][0]  # Return the actual strategy index
            else:
                print(f"Please enter a number between 1 and {len(strategies)}")
        except ValueError:
            print("Please enter a valid number")