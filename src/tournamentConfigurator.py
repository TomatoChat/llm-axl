"""
Tournament Configurator Module

This module handles all user interactions and configuration setup for the tournament.
"""

from typing import List, Tuple, Optional
from axelrod.strategies import all_strategies
from .tournamentConfiguration import TournamentConfiguration, LLMStrategyConfig


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
    
    def getLLMConfiguration(self) -> Optional[LLMStrategyConfig]:
        """
        Get LLM configuration from user if LLM strategies are selected.
        
        Returns:
            Optional[LLMStrategyConfig]: LLM configuration or None if not needed
        """
        print("\n--- LLM Strategy Configuration ---")
        
        # Check if any LLM strategies were selected
        llm_strategies = ["LLMStrategy", "LLMStrategyWithMemory"]
        has_llm_strategies = any(strategy in self.selected_strategies for strategy in llm_strategies)
        
        if not has_llm_strategies:
            print("No LLM strategies selected. Skipping LLM configuration.")
            return None
        
        print("LLM strategies detected! Please configure the LLM provider and model.")
        
        # Get provider selection
        providers = [
            ("openai", "OpenAI (GPT-3.5, GPT-4)"),
            ("gemini", "Google Gemini (Gemini Pro)"),
            ("anthropic", "Anthropic Claude (Claude Sonnet, Haiku, Opus)")
        ]
        
        print("\nAvailable LLM Providers:")
        for i, (provider, description) in enumerate(providers, 1):
            print(f"{i}: {description}")
        
        while True:
            try:
                provider_choice = int(input("\nSelect LLM provider (1-3): "))
                if 1 <= provider_choice <= 3:
                    provider = providers[provider_choice - 1][0]
                    break
                else:
                    print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get model selection based on provider
        if provider == "openai":
            models = [
                ("gpt-3.5-turbo", "GPT-3.5 Turbo (Fast, Cost-effective)"),
                ("gpt-4", "GPT-4 (More capable, Higher cost)"),
                ("gpt-4-turbo-preview", "GPT-4 Turbo (Latest, Best performance)")
            ]
        elif provider == "gemini":
            models = [
                ("gemini-pro", "Gemini Pro (Recommended)"),
                ("gemini-pro-vision", "Gemini Pro Vision (With image support)")
            ]
        elif provider == "anthropic":
            models = [
                ("claude-3-sonnet-20240229", "Claude 3 Sonnet (Balanced)"),
                ("claude-3-haiku-20240307", "Claude 3 Haiku (Fast, Cost-effective)"),
                ("claude-3-opus-20240229", "Claude 3 Opus (Most capable)")
            ]
        
        print(f"\nAvailable {provider.upper()} Models:")
        for i, (model, description) in enumerate(models, 1):
            print(f"{i}: {description}")
        
        while True:
            try:
                model_choice = int(input(f"\nSelect {provider.upper()} model (1-{len(models)}): "))
                if 1 <= model_choice <= len(models):
                    model = models[model_choice - 1][0]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get API key (optional)
        print(f"\nAPI Key Configuration for {provider.upper()}:")
        print("You can provide an API key now or set it as an environment variable.")
        
        api_key_input = input(f"Enter {provider.upper()} API key (or press Enter to use environment variable): ").strip()
        api_key = api_key_input if api_key_input else None
        
        if not api_key:
            env_var_name = f"{provider.upper()}_API_KEY"
            print(f"Using environment variable: {env_var_name}")
            print(f"Make sure to set: export {env_var_name}='your-api-key'")
        
        return LLMStrategyConfig(
            provider=provider,
            model=model,
            api_key=api_key
        )
    
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
        
        # Store selected strategies for LLM configuration check
        self.selected_strategies = selectedStrategies
        
        # Get game parameters
        print("\n--- Game Parameters ---")
        turns, noise = self.getGameParameters()
        
        # Get LLM configuration if needed
        llm_config = self.getLLMConfiguration()
        
        return TournamentConfiguration(
            numPlayers=numPlayers,
            strategies=selectedStrategies,
            turns=turns,
            noise=noise,
            payoffMatrix=self.defaultPayoffMatrix,
            llmConfig=llm_config
        ) 