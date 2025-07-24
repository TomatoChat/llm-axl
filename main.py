#!/usr/bin/env python3
"""
Axelrod Tournament Simulator - Main Entry Point

A flexible and interactive command-line application that simulates the Iterated
Prisoner's Dilemma using the axelrod library. The application allows users to
configure, run, and view the results of a complete tournament through a series
of clear and intuitive prompts.

This is the main entry point for the application.
"""

from src.tournamentSimulator import TournamentSimulator


def main():
    """
    Main entry point for the application.
    """
    simulator = TournamentSimulator()
    simulator.run()


if __name__ == "__main__":
    main() 