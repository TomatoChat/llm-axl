"""
Tournament Configuration Module

This module contains the data class for storing tournament configuration parameters.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TournamentConfiguration:
    """
    Data class to store tournament configuration parameters.
    
    Attributes:
        numPlayers (int): Number of participants in the tournament
        strategies (List[str]): List of strategy names for each player
        turns (int): Number of turns per match (or max turns if using variable length)
        noise (float): Noise level (probability of move being flipped)
        probEnd (float): Probability of match ending each turn (None for fixed length)
        distributionType (str): Type of distribution for variable length ('geometric', 'normal', 'uniform')
        distributionParams (Dict): Parameters for the distribution (mean, std for normal, etc.)
        payoffMatrix (Dict[str, int]): Game payoff matrix (R, P, S, T)
    """
    numPlayers: int
    strategies: List[str]
    turns: int
    noise: float
    probEnd: float = None
    distributionType: str = None
    distributionParams: Dict = None
    payoffMatrix: Dict[str, int] = None 