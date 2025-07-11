#!/usr/bin/env python3
"""
Axelrod Game Theory Player
A program to play against strategies from the Axelrod package.
"""

import sys
from typing import List, Tuple
from strategyPlayer import StrategyPlayer, getAvailableStrategies, displayStrategies, getUserChoice


def getUserMove() -> str:
    """
    Get user's move (C for Cooperate, D for Defect).
    
    Args:
        None
    
    Returns:
        str: User's move as 'C' for Cooperate or 'D' for Defect
    
    Note:
        This function handles input validation and will keep prompting until a valid move is entered.
    """
    while True:
        move = input("Your move (C/D): ").upper().strip()
        if move in ['C', 'D']:
            return move
        else:
            print("Please enter 'C' for Cooperate or 'D' for Defect")


def playRound(opponent: StrategyPlayer, userHistory: List[str]) -> Tuple[str, str]:
    """
    Play one round of the game.
    
    Args:
        opponent (StrategyPlayer): The strategy player to play against
        userHistory (List[str]): List of user's previous moves (C/D)
    
    Returns:
        Tuple[str, str]: Tuple containing (user_move, opponent_move) where each move is 'C' or 'D'
    """
    # Get user's move
    userMove = getUserMove()
    
    # Get opponent's move based on the strategy
    opponentMove = opponent.makeDecision(userHistory)
    
    return userMove, opponentMove


def displayRoundResult(userMove: str, opponentMove: str, roundNum: int) -> None:
    """
    Display the result of a round.
    
    Args:
        userMove (str): User's move ('C' or 'D')
        opponentMove (str): Opponent's move ('C' or 'D')
        roundNum (int): Current round number
    
    Returns:
        None
    
    Note:
        This function prints the round result and explains the scoring based on Prisoner's Dilemma rules.
    """
    print(f"\nRound {roundNum}:")
    print(f"You played: {userMove}")
    print(f"Opponent played: {opponentMove}")
    
    # Determine outcome
    if userMove == 'C' and opponentMove == 'C':
        print("Both cooperated - You both get 3 points")
    elif userMove == 'C' and opponentMove == 'D':
        print("You cooperated, opponent defected - You get 0, opponent gets 5")
    elif userMove == 'D' and opponentMove == 'C':
        print("You defected, opponent cooperated - You get 5, opponent gets 0")
    else:  # D vs D
        print("Both defected - You both get 1 point")


def calculateScores(userHistory: List[str], opponentHistory: List[str]) -> Tuple[int, int]:
    """
    Calculate final scores based on game history.
    
    Args:
        userHistory (List[str]): List of user's moves (C/D)
        opponentHistory (List[str]): List of opponent's moves (C/D)
    
    Returns:
        Tuple[int, int]: Tuple containing (user_score, opponent_score) based on Prisoner's Dilemma scoring
        
    Note:
        Scoring: CC=3,3; CD=0,5; DC=5,0; DD=1,1
    """
    userScore = 0
    opponentScore = 0
    
    for userMove, opponentMove in zip(userHistory, opponentHistory):
        if userMove == 'C' and opponentMove == 'C':
            userScore += 3
            opponentScore += 3
        elif userMove == 'C' and opponentMove == 'D':
            userScore += 0
            opponentScore += 5
        elif userMove == 'D' and opponentMove == 'C':
            userScore += 5
            opponentScore += 0
        else:  # D vs D
            userScore += 1
            opponentScore += 1
    
    return userScore, opponentScore


def main():
    """
    Main game loop.
    
    Args:
        None
    
    Returns:
        None
    
    Note:
        This function handles the complete game flow including:
        - Displaying available strategies
        - Getting user input for strategy selection and number of rounds
        - Playing the game rounds
        - Displaying final results and game history
    """
    print("Welcome to Axelrod Game Theory Player!")
    print("You will play against strategies from the Axelrod package.")
    print("In each round, you can choose to Cooperate (C) or Defect (D).")
    print("Scoring: CC=3,3; CD=0,5; DC=5,0; DD=1,1")
    
    # Get available strategies
    strategies = getAvailableStrategies()
    displayStrategies(strategies)
    
    # Get user's choice
    strategyIdx = getUserChoice(strategies)
    
    # Create opponent using StrategyPlayer
    opponent = StrategyPlayer(strategyIdx)
    
    print(f"\nYou chose to play against: {opponent.strategyName}")
    
    # Initialize histories
    userHistory = []
    opponentHistory = []
    
    # Get number of rounds
    while True:
        try:
            rounds = int(input("\nHow many rounds would you like to play? "))
            if rounds > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nStarting game against {opponent.strategyName} for {rounds} rounds...")
    print("=" * 50)
    
    # Play rounds
    for roundNum in range(1, rounds + 1):
        userMove, opponentMove = playRound(opponent, userHistory)
        displayRoundResult(userMove, opponentMove, roundNum)
        
        # Update histories
        userHistory.append(userMove)
        opponentHistory.append(opponentMove)
    
    # Calculate and display final scores
    userScore, opponentScore = calculateScores(userHistory, opponentHistory)
    
    print("\n" + "=" * 50)
    print("GAME OVER")
    print("=" * 50)
    print(f"Final Scores:")
    print(f"You: {userScore} points")
    print(f"{opponent.strategyName}: {opponentScore} points")
    
    if userScore > opponentScore:
        print("Congratulations! You won!")
    elif userScore < opponentScore:
        print(f"You lost to {opponent.strategyName}!")
    else:
        print("It's a tie!")
    
    print(f"\nGame history:")
    for i, (userMove, opponentMove) in enumerate(zip(userHistory, opponentHistory), 1):
        print(f"Round {i}: You={userMove}, {opponent.strategyName}={opponentMove}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1) 