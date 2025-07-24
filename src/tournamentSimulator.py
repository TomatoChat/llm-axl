"""
Tournament Simulator Module

This module contains the main orchestrator class for the tournament simulation process.
"""

import sys
import axelrod
from .tournamentConfigurator import TournamentConfigurator
from .tournamentRunner import TournamentRunner


class TournamentSimulator:
    """
    Main class to orchestrate the entire tournament simulation process.
    """
    
    def __init__(self):
        """Initialize the tournament simulator."""
        self.configurator = TournamentConfigurator()
        self.runner = None
        self.results = None
    
    def displayResults(self, results: axelrod.ResultSet) -> None:
        """
        Display tournament results in a formatted table.
        
        Args:
            results (axelrod.ResultSet): Tournament results
        """
        print("\n--- Tournament Results ---")
        
        # Get scores and rankings
        scores = results.scores
        meanScores = results.mean_scores
        
        # Create a list of (player_index, total_score, mean_score) tuples
        playerResults = []
        for i in range(len(self.runner.players)):
            totalScore = scores[i][0]  # First (and only) repetition
            meanScore = meanScores[i]
            playerResults.append((i, totalScore, meanScore))
        
        # Sort by total score (descending)
        playerResults.sort(key=lambda x: x[1], reverse=True)
        
        # Create results table
        print("Rank | Strategy    | Score | Score Per Turn")
        print("-" * 45)
        
        for rank, (playerIndex, totalScore, meanScore) in enumerate(playerResults, 1):
            strategyName = self.runner.players[playerIndex].name
            
            # Format the output
            strategyDisplay = strategyName[:12].ljust(12)  # Truncate and pad
            scoreDisplay = f"{totalScore:.1f}"
            meanScoreDisplay = f"{meanScore:.2f}"
            
            print(f"{rank:<4} | {strategyDisplay} | {scoreDisplay:<5} | {meanScoreDisplay}")
    
    def run(self) -> None:
        """
        Run the complete tournament simulation process.
        """
        try:
            # Configure tournament
            config = self.configurator.configureTournament()
            
            # Run tournament
            self.runner = TournamentRunner(config)
            self.results = self.runner.runTournament()
            
            # Display results
            self.displayResults(self.results)
            
        except KeyboardInterrupt:
            print("\n\nTournament simulation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError during tournament simulation: {e}")
            sys.exit(1) 