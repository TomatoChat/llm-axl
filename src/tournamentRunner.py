"""
Tournament Runner Module

This module handles the execution of the tournament simulation.
"""

import axelrod
import numpy as np
from axelrod.strategies import all_strategies
from .tournamentConfiguration import TournamentConfiguration
from .llmStrategy import LLMStrategy, LLMStrategyWithMemory, tournament_config_to_game_config


class TournamentRunner:
    """
    Handles the execution of the tournament simulation.
    """
    
    def __init__(self, config: TournamentConfiguration):
        """
        Initialize the tournament runner with configuration.
        
        Args:
            config (TournamentConfiguration): Tournament configuration
        """
        self.config = config
        self.players = []
        self.tournament = None
        self.results = None
    
    def createPlayers(self) -> None:
        """
        Create player instances from the selected strategies.
        """
        self.players = []
        for strategyName in self.config.strategies:
            # Handle LLM strategies specially
            if strategyName.startswith("LLMStrategy"):
                player = self._createLLMPlayer(strategyName)
            else:
                # Find the strategy class by name
                strategyClass = None
                for strategy in all_strategies:
                    if strategy.name == strategyName:
                        strategyClass = strategy
                        break
                
                if strategyClass:
                    player = strategyClass()
                else:
                    raise ValueError(f"Strategy '{strategyName}' not found")
            
            self.players.append(player)
    
    def _createLLMPlayer(self, strategyName: str):
        """
        Create an LLM strategy player with the configured settings.
        
        Args:
            strategyName (str): Name of the LLM strategy (e.g., "LLMStrategy (OpenAI - GPT-3.5 Turbo)")
            
        Returns:
            LLMStrategy or LLMStrategyWithMemory: Configured LLM player
        """
        # Parse strategy name to extract provider and model
        import re
        
        # Extract provider and model from strategy name
        # Format: "LLMStrategy (Provider - Model)" or "LLMStrategyWithMemory (Provider - Model)"
        match = re.match(r'LLMStrategy(WithMemory)?\s*\(([^-]+)\s*-\s*(.+)\)', strategyName)
        
        if match:
            strategy_type = "LLMStrategyWithMemory" if match.group(1) else "LLMStrategy"
            provider_display = match.group(2).strip()
            model_display = match.group(3).strip()
            
            # Map display names to actual provider/model values
            provider_map = {
                "Openai": "openai",
                "Gemini": "gemini", 
                "Anthropic": "anthropic"
            }
            
            # For dynamic models, we'll extract the model ID directly from the display name
            # The format is "Provider - Model Display Name"
            # We need to map the display name back to the actual model ID
            model_map = {
                # OpenAI models
                "GPT-4o (Latest)": "gpt-4o",
                "GPT-4o Mini (Fast)": "gpt-4o-mini",
                "GPT-4 Turbo": "gpt-4-turbo",
                "GPT-4": "gpt-4",
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
                "GPT-3.5 Turbo 16K": "gpt-3.5-turbo-16k",
                
                # Gemini models
                "Gemini 1.5 Pro": "gemini-1.5-pro",
                "Gemini 1.5 Flash": "gemini-1.5-flash",
                "Gemini 1.5 Pro Latest": "gemini-1.5-pro-latest",
                "Gemini 1.5 Flash Latest": "gemini-1.5-flash-latest",
                "Gemini Pro": "gemini-pro",
                "Gemini Pro Vision": "gemini-pro-vision",
                
                # Anthropic models
                "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
                "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
                "Claude 3 Opus": "claude-3-opus-20240229",
                "Claude 3 Sonnet": "claude-3-sonnet-20240229",
                "Claude 3 Haiku": "claude-3-haiku-20240307",
                "Claude 2.1": "claude-2.1",
                "Claude 2.0": "claude-2.0",
                "Claude Instant 1.2": "claude-instant-1.2"
            }
            
            provider = provider_map.get(provider_display, "openai")
            model = model_map.get(model_display, "gpt-3.5-turbo")
            
            # Create with parsed settings
            game_config = tournament_config_to_game_config(self.config)
            
            if strategy_type == "LLMStrategy":
                return LLMStrategy(
                    provider=provider,
                    model=model,
                    api_key=None,  # Will use environment variable
                    game_config=game_config
                )
            else:  # LLMStrategyWithMemory
                return LLMStrategyWithMemory(
                    provider=provider,
                    model=model,
                    api_key=None,  # Will use environment variable
                    game_config=game_config
                )
        else:
            # Fallback for old format or if parsing fails
            print(f"Warning: Could not parse LLM strategy name '{strategyName}'. Using fallback settings.")
            game_config = tournament_config_to_game_config(self.config)
            if "WithMemory" in strategyName:
                return LLMStrategyWithMemory(game_config=game_config)
            else:
                return LLMStrategy(game_config=game_config)
    
    def runTournament(self, progress_callback=None) -> axelrod.ResultSet:
        """
        Run the tournament simulation.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            axelrod.ResultSet: Tournament results
        """
        print("\nRunning simulation...")
        
        # Create players
        self.createPlayers()
        
        # Create custom game object if payoff matrix is provided
        game = None
        if self.config.payoffMatrix:
            game = axelrod.Game(
                r=self.config.payoffMatrix['R'],
                s=self.config.payoffMatrix['S'],
                t=self.config.payoffMatrix['T'],
                p=self.config.payoffMatrix['P']
            )
        
        # Handle different distribution types
        if self.config.distributionType == 'geometric':
            # Use Axelrod's built-in prob_end for geometric distribution
            tournament_kwargs = {
                'players': self.players,
                'noise': self.config.noise,
                'repetitions': 1,
                'prob_end': self.config.probEnd
            }
            if self.config.turns:
                tournament_kwargs['turns'] = self.config.turns
            if game:
                tournament_kwargs['game'] = game
                
            self.tournament = axelrod.Tournament(**tournament_kwargs)
            self.results = self.tournament.play()
            
        elif self.config.distributionType in ['normal', 'uniform']:
            # Custom implementation for other distributions
            self.results = self._runCustomDistributionTournament(progress_callback)
            
        else:
            # Fixed length tournament
            tournament_kwargs = {
                'players': self.players,
                'noise': self.config.noise,
                'repetitions': 1,
                'turns': self.config.turns
            }
            if game:
                tournament_kwargs['game'] = game
            self.tournament = axelrod.Tournament(**tournament_kwargs)
            self.results = self.tournament.play()
        
        return self.results
    
    def _runCustomDistributionTournament(self, progress_callback=None) -> axelrod.ResultSet:
        """
        Run tournament with custom distribution for game lengths.
        
        Args:
            progress_callback: Optional callback function for progress updates
            
        Returns:
            axelrod.ResultSet: Tournament results
        """
        # Generate game lengths for each match
        num_players = len(self.players)
        num_matches = num_players * (num_players - 1) // 2  # Round robin
        
        if self.config.distributionType == 'normal':
            # Normal distribution
            mean = self.config.distributionParams.get('mean', self.config.turns)
            std = self.config.distributionParams.get('std', mean * 0.2)
            game_lengths = np.random.normal(mean, std, num_matches)
        elif self.config.distributionType == 'uniform':
            # Uniform distribution
            min_turns = self.config.distributionParams.get('min', 10)
            max_turns = self.config.distributionParams.get('max', self.config.turns)
            game_lengths = np.random.uniform(min_turns, max_turns, num_matches)
        
        # Ensure all lengths are positive integers within bounds
        game_lengths = np.clip(game_lengths, 1, self.config.turns)
        game_lengths = game_lengths.astype(int)
        
        # Create a custom tournament by running individual matches
        scores = [[0] for _ in range(num_players)]
        
        match_index = 0
        for i in range(num_players):
            for j in range(i + 1, num_players):
                # Get game length for this match
                turns = game_lengths[match_index]
                
                # Create match with custom game if provided
                match_kwargs = {
                    'players': (self.players[i], self.players[j]),
                    'turns': turns,
                    'noise': self.config.noise
                }
                
                # Add custom game if available
                if self.config.payoffMatrix:
                    game = axelrod.Game(
                        r=self.config.payoffMatrix['R'],
                        s=self.config.payoffMatrix['S'],
                        t=self.config.payoffMatrix['T'],
                        p=self.config.payoffMatrix['P']
                    )
                    match_kwargs['game'] = game
                
                match = axelrod.Match(**match_kwargs)
                
                # Play match
                match.play()
                
                # Update scores
                scores[i][0] += match.final_score()[0]
                scores[j][0] += match.final_score()[1]
                
                # Update progress
                if progress_callback:
                    progress = (match_index + 1) / num_matches
                    progress_callback(progress, f"Match {match_index + 1}/{num_matches}: {self.players[i].name} vs {self.players[j].name}")
                
                match_index += 1
        
        # Create a ResultSet-like object
        class CustomResultSet:
            def __init__(self, scores):
                self.scores = scores
                self.mean_scores = [score[0] / len(scores) for score in scores]
        
        return CustomResultSet(scores) 