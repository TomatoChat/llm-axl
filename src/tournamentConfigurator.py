"""
Tournament Configurator Module

This module handles all user interactions and configuration setup for the tournament.
"""

from typing import List, Tuple
from axelrod.strategies import all_strategies
from .tournamentConfiguration import TournamentConfiguration


class TournamentConfigurator:
    """
    Handles all user interactions and configuration setup for the tournament.
    """
    
    def __init__(self):
        """Initialize the configurator with default values."""
        self.defaultPayoffMatrix = {
            'R': 3,  # Reward for mutual cooperation
            'P': 1,  # Punishment for mutual defection
            'S': 0,  # Sucker's payoff (cooperate while opponent defects)
            'T': 5   # Temptation to defect (defect while opponent cooperates)
        }
    
    def getAvailableStrategies(self) -> List[Tuple[int, str]]:
        """
        Get a list of available strategies from the axelrod library.
        
        Returns:
            List[Tuple[int, str]]: List of (index, strategy_name) tuples
        """
        strategies = []
        for i, strategy in enumerate(all_strategies, 1):
            strategies.append((i, strategy.name))
        return strategies
    
    def displayStrategies(self, strategies: List[Tuple[int, str]]) -> None:
        """
        Display available strategies in a numbered list.
        
        Args:
            strategies (List[Tuple[int, str]]): List of (index, strategy_name) tuples
        """
        print("\nAvailable Strategies:")
        for index, name in strategies:
            print(f"{index}: {name}")
    
    def getNumPlayers(self) -> int:
        """
        Prompt user for number of players.
        
        Returns:
            int: Number of players for the tournament
        """
        while True:
            try:
                numPlayers = int(input("\nHow many players will participate? "))
                if numPlayers > 0:
                    return numPlayers
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    def getStrategySelection(self, playerNum: int, strategies: List[Tuple[int, str]]) -> str:
        """
        Get strategy selection for a specific player.
        
        Args:
            playerNum (int): Player number (1-based)
            strategies (List[Tuple[int, str]]): Available strategies
            
        Returns:
            str: Selected strategy name
        """
        while True:
            try:
                print(f"\n--- Player {playerNum} Configuration ---")
                self.displayStrategies(strategies)
                selection = int(input(f"Please select a strategy for Player {playerNum}: "))
                
                if 1 <= selection <= len(strategies):
                    return strategies[selection - 1][1]
                else:
                    print(f"Please enter a number between 1 and {len(strategies)}.")
            except ValueError:
                print("Please enter a valid number.")
    
    def getGameParameters(self) -> Tuple[int, float]:
        """
        Get game parameters (turns and noise) from user.
        
        Returns:
            Tuple[int, float]: (turns, noise) tuple
        """
        # Get number of turns
        while True:
            try:
                turnsInput = input("\nEnter number of turns per match [default: 200]: ").strip()
                if turnsInput == "":
                    turns = 200
                else:
                    turns = int(turnsInput)
                    if turns > 0:
                        break
                    else:
                        print("Please enter a positive number.")
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Get noise level
        while True:
            try:
                noiseInput = input("Enter noise level (0.0 to 1.0) [default: 0.0]: ").strip()
                if noiseInput == "":
                    noise = 0.0
                else:
                    noise = float(noiseInput)
                    if 0.0 <= noise <= 1.0:
                        break
                    else:
                        print("Please enter a number between 0.0 and 1.0.")
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        return turns, noise
    
    def configureTournament(self) -> TournamentConfiguration:
        """
        Configure the entire tournament through user interaction.
        
        Returns:
            TournamentConfiguration: Complete tournament configuration
        """
        print("Welcome to the Axelrod Tournament Simulator!")
        print("\n--- Tournament Setup ---")
        
        # Get number of players
        numPlayers = self.getNumPlayers()
        
        # Get available strategies
        strategies = self.getAvailableStrategies()
        
        # Get strategy selection for each player
        selectedStrategies = []
        for i in range(1, numPlayers + 1):
            strategy = self.getStrategySelection(i, strategies)
            selectedStrategies.append(strategy)
        
        # Get game parameters
        print("\n--- Game Parameters ---")
        turns, noise = self.getGameParameters()
        
        return TournamentConfiguration(
            numPlayers=numPlayers,
            strategies=selectedStrategies,
            turns=turns,
            noise=noise,
            payoffMatrix=self.defaultPayoffMatrix
        ) 