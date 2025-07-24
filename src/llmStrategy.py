"""
LLM Strategy Module

This module contains an LLM-based strategy for the Iterated Prisoner's Dilemma.
The strategy uses a context prompt to understand the game rules and make decisions.
"""

import axelrod
from typing import List, Optional, Dict, Any, Literal
import os
import yaml
from pathlib import Path

# Optional imports for different providers
try:
    import openai
except ImportError:
    openai = None

# Optional imports for different providers
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


def tournament_config_to_game_config(tournament_config) -> Dict[str, Any]:
    """
    Convert tournament configuration to game config format for LLM strategies.
    
    Args:
        tournament_config: TournamentConfiguration object or dict
        
    Returns:
        Dict[str, Any]: Game configuration dictionary
    """
    if hasattr(tournament_config, '__dict__'):
        # Convert object to dict
        config_dict = tournament_config.__dict__
    else:
        config_dict = tournament_config
    
    return {
        'turns': config_dict.get('turns', 200),
        'noise': config_dict.get('noise', 0.0),
        'probEnd': config_dict.get('probEnd'),
        'distributionType': config_dict.get('distributionType'),
        'distributionParams': config_dict.get('distributionParams'),
        'payoffMatrix': config_dict.get('payoffMatrix', {
            'R': 3, 'P': 1, 'S': 0, 'T': 5
        })
    }


class LLMStrategy(axelrod.Player):
    """
    An LLM-based strategy that uses OpenAI's API to make decisions in the Iterated Prisoner's Dilemma.
    
    This strategy provides the LLM with the game context and history, then uses the LLM's
    response to determine whether to cooperate or defect.
    """
    
    name = "LLMStrategy"
    classifier = {
        'memory_depth': float('inf'),  # Remembers entire history
        'stochastic': True,  # LLM responses may be stochastic
        'makes_use_of': ['game', 'length', 'opponent_name'],
        'long_run_time': True,  # API calls take time
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    
    def __init__(self, 
                 provider: Literal["openai", "gemini", "anthropic"] = "openai",
                 api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 game_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM strategy.
        
        Args:
            provider (str): LLM provider ("openai", "gemini", or "anthropic")
            api_key (Optional[str]): API key for the provider. If None, will try to get from environment.
            model (str): Model name to use for decisions.
            game_config (Optional[Dict]): Game configuration including payoff matrix, turns, noise, etc.
        """
        super().__init__()
        self.provider = provider
        self.model = model
        self.client = None
        
        # Store game configuration
        self.game_config = game_config or {}
        
        # Set up API key based on provider
        if provider == "openai":
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if self.api_key and openai:
                try:
                    self.client = openai.OpenAI(api_key=self.api_key)
                except Exception as e:
                    print(f"Warning: Could not initialize OpenAI client: {e}")
                    self.client = None
            elif not openai:
                print("Warning: OpenAI module not available. Please install it with: pip install openai")
                self.client = None
                    
        elif provider == "gemini":
            self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
            if self.api_key and genai:
                try:
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(model)
                except Exception as e:
                    print(f"Warning: Could not initialize Gemini client: {e}")
                    self.client = None
                    
        elif provider == "anthropic":
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if self.api_key and Anthropic:
                try:
                    self.client = Anthropic(api_key=self.api_key)
                except Exception as e:
                    print(f"Warning: Could not initialize Anthropic client: {e}")
                    self.client = None
        
        if not self.client:
            print(f"Warning: No client initialized for provider '{provider}'. Using fallback strategy.")
    
    def _loadPromptTemplate(self, prompt_file: str) -> str:
        """
        Load a prompt template from an external file.
        
        Args:
            prompt_file (str): Name of the prompt file in the prompts directory
            
        Returns:
            str: The loaded prompt template
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        # Get the path to the prompts directory
        current_dir = Path(__file__).parent
        prompts_dir = current_dir / "prompts"
        prompt_path = prompts_dir / prompt_file
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # Load the prompt template
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _getGameContextPrompt(self) -> str:
        """
        Get the context prompt that explains the game rules to the LLM.
        
        Returns:
            str: The context prompt explaining the game rules
        """
        try:
            # Load the prompt template from external file
            prompt_template = self._loadPromptTemplate("gameRulesPrompt.txt")
        except FileNotFoundError:
            # Fallback to hardcoded prompt if file not found
            print("Warning: Could not load gameRulesPrompt.txt, using fallback prompt")
            return self._getFallbackGameContextPrompt()
        
        # Get payoff matrix from game config or use defaults
        payoff_matrix = self.game_config.get('payoffMatrix', {
            'R': 3,  # Reward for mutual cooperation
            'P': 1,  # Punishment for mutual defection
            'S': 0,  # Sucker's payoff (cooperate while opponent defects)
            'T': 5   # Temptation to defect (defect while opponent cooperates)
        })
        
        # Get game length information
        turns = self.game_config.get('turns', 200)
        prob_end = self.game_config.get('probEnd')
        distribution_type = self.game_config.get('distributionType')
        
        # Get noise level
        noise = self.game_config.get('noise', 0.0)
        
        # Build game length information
        if prob_end and distribution_type:
            if distribution_type == "geometric":
                expected_length = 1 / prob_end
                game_length_info = f"""
- Game length is variable with geometric distribution
- Each turn has a {prob_end:.3f} probability of ending the game
- Expected game length is approximately {expected_length:.1f} turns
- Maximum possible turns: {turns}"""
            elif distribution_type == "normal":
                mean = self.game_config.get('distributionParams', {}).get('mean', turns)
                std = self.game_config.get('distributionParams', {}).get('std', mean * 0.2)
                game_length_info = f"""
- Game length is variable with normal distribution
- Mean game length: {mean} turns
- Standard deviation: {std:.1f} turns
- Maximum possible turns: {turns}"""
            elif distribution_type == "uniform":
                min_turns = self.game_config.get('distributionParams', {}).get('min', 10)
                max_turns = self.game_config.get('distributionParams', {}).get('max', turns)
                game_length_info = f"""
- Game length is variable with uniform distribution
- Game length ranges from {min_turns} to {max_turns} turns
- Maximum possible turns: {turns}"""
        else:
            game_length_info = f"""
- Game length is fixed at {turns} turns"""
        
        # Build noise information
        if noise > 0:
            noise_info = f"""
- There is {noise:.3f} probability that moves will be randomly flipped (noise)"""
        else:
            noise_info = ""
        
        # Build noise considerations
        if noise > 0:
            noise_considerations = f"""
- Due to noise ({noise:.3f}), some moves may be misinterpreted
- Consider that your opponent's moves might be affected by noise
- Be more forgiving of occasional defections as they might be noise-related"""
        else:
            noise_considerations = ""
        
        # Format the prompt template with dynamic values
        formatted_prompt = prompt_template.format(
            R=payoff_matrix['R'],
            P=payoff_matrix['P'],
            S=payoff_matrix['S'],
            T=payoff_matrix['T'],
            game_length_info=game_length_info,
            noise_info=noise_info,
            noise_considerations=noise_considerations
        )
        
        return formatted_prompt
    
    def _getFallbackGameContextPrompt(self) -> str:
        """
        Fallback game context prompt when external file is not available.
        
        Returns:
            str: Hardcoded fallback prompt
        """
        # Get payoff matrix from game config or use defaults
        payoff_matrix = self.game_config.get('payoffMatrix', {
            'R': 3, 'P': 1, 'S': 0, 'T': 5
        })
        
        return f"""You are playing the Iterated Prisoner's Dilemma, a classic game theory problem.

GAME RULES:
- You and your opponent play multiple rounds of the same game
- In each round, both players simultaneously choose to either COOPERATE (C) or DEFECT (D)
- Your goal is to maximize your total score across all rounds
- You can see the complete history of both players' moves

PAYOFF MATRIX (Your Score, Opponent's Score):
- If you COOPERATE and opponent COOPERATES: You get {payoff_matrix['R']} points, opponent gets {payoff_matrix['R']} points
- If you COOPERATE and opponent DEFECTS: You get {payoff_matrix['S']} points, opponent gets {payoff_matrix['T']} points  
- If you DEFECT and opponent COOPERATES: You get {payoff_matrix['T']} points, opponent gets {payoff_matrix['S']} points
- If you DEFECT and opponent DEFECTS: You get {payoff_matrix['P']} points, opponent gets {payoff_matrix['P']} points

STRATEGIC CONSIDERATIONS:
- The highest individual payoff ({payoff_matrix['T']} points) comes from defecting when opponent cooperates
- The lowest individual payoff ({payoff_matrix['S']} points) comes from cooperating when opponent defects
- Mutual cooperation ({payoff_matrix['R']} points each) is better than mutual defection ({payoff_matrix['P']} points each)
- Your opponent is also trying to maximize their score
- You can use the history of moves to understand your opponent's strategy

DECISION MAKING:
- Analyze the opponent's previous moves to understand their strategy
- Consider whether they are cooperative, retaliatory, or exploitative
- Balance short-term gains with long-term relationship building
- Remember that your opponent can see your history too

RESPONSE FORMAT:
Respond with a YAML object containing your decision and reasoning:

```yaml
decision: "C"  # or "D"
reasoning: "Brief explanation of your strategic thinking"
confidence: 0.85  # Confidence level from 0.0 to 1.0
strategy_notes: "Any additional strategic observations"
```

Example response:
```yaml
decision: "C"
reasoning: "Opponent has been cooperative, maintaining mutual cooperation is optimal"
confidence: 0.9
strategy_notes: "Detected Tit-for-Tat pattern, continuing cooperation"
```"""
    
    def _getMoveHistoryPrompt(self, opponent_history: List[str], my_history: List[str]) -> str:
        """
        Create a prompt that includes the move history for decision making.
        
        Args:
            opponent_history (List[str]): List of opponent's previous moves ('C' or 'D')
            my_history (List[str]): List of your previous moves ('C' or 'D')
            
        Returns:
            str: Prompt with move history
        """
        if not opponent_history:
            return "This is the first round. Make your first move."
        
        # Create a conversation-style history
        history_lines = []
        for i, (my_move, opp_move) in enumerate(zip(my_history, opponent_history)):
            round_num = i + 1
            history_lines.append(f"Round {round_num}: You chose {my_move}, opponent chose {opp_move}")
        
        history_text = "\n".join(history_lines)
        current_round = len(opponent_history) + 1
        
        return f"""GAME CONVERSATION:
{history_text}

Round {current_round}: It's your turn to make a move. Based on the conversation so far, what do you choose? Respond with only 'C' or 'D'."""
    
    def _callLLM(self, prompt: str) -> Dict[str, Any]:
        """
        Call the LLM API to get a decision.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            Dict[str, Any]: Parsed YAML response with decision and metadata
        """
        if not self.client:
            # Fallback to a simple strategy if no API access
            print(f"Warning: No {self.provider} client available. Using fallback strategy.")
            fallback_decision = self._fallbackStrategy()
            return {
                'decision': fallback_decision,
                'reasoning': f'Fallback strategy used (no {self.provider} API access)',
                'confidence': 0.5,
                'strategy_notes': 'Tit-for-Tat fallback'
            }
        
        try:
            if self.provider == "openai":
                response_text = self._callOpenAI(prompt)
            elif self.provider == "gemini":
                response_text = self._callGemini(prompt)
            elif self.provider == "anthropic":
                response_text = self._callAnthropic(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse YAML response
            parsed_response = self._parseYAMLResponse(response_text)
            return parsed_response
                
        except Exception as e:
            print(f"Error: {self.provider} API call failed: {e}")
            # For LLM strategies, we should throw an error if API calls fail
            # since the whole point is to use the LLM
            raise RuntimeError(f"LLM strategy failed to make API call to {self.provider}: {e}")
    
    def _callOpenAI(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not openai:
            raise ImportError("OpenAI module not available. Please install it with: pip install openai")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._getGameContextPrompt()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    
    def _callGemini(self, prompt: str) -> str:
        """Call Gemini API."""
        full_prompt = f"{self._getGameContextPrompt()}\n\n{prompt}"
        response = self.client.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.1
            )
        )
        return response.text.strip()
    
    def _callAnthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        full_prompt = f"{self._getGameContextPrompt()}\n\n{prompt}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0.1,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.content[0].text.strip()
    
    def _parseYAMLResponse(self, response_text: str) -> Dict[str, Any]:
        """Parse YAML response from LLM."""
        try:
            # Extract YAML from the response (handle cases where it's wrapped in code blocks)
            yaml_text = response_text
            if '```yaml' in response_text:
                yaml_text = response_text.split('```yaml')[1].split('```')[0].strip()
            elif '```' in response_text:
                yaml_text = response_text.split('```')[1].strip()
            
            parsed_response = yaml.safe_load(yaml_text)
            
            # Validate the parsed response
            if isinstance(parsed_response, dict) and 'decision' in parsed_response:
                decision = parsed_response['decision'].upper()
                if decision in ['C', 'D']:
                    return {
                        'decision': decision,
                        'reasoning': parsed_response.get('reasoning', 'No reasoning provided'),
                        'confidence': parsed_response.get('confidence', 0.5),
                        'strategy_notes': parsed_response.get('strategy_notes', 'No notes provided')
                    }
            
            # If YAML parsing failed or invalid, try to extract just the decision
            if 'decision:' in response_text:
                decision_line = [line for line in response_text.split('\n') if 'decision:' in line][0]
                decision = decision_line.split('decision:')[1].strip().strip('"').upper()
                if decision in ['C', 'D']:
                    return {
                        'decision': decision,
                        'reasoning': 'Extracted from malformed YAML',
                        'confidence': 0.3,
                        'strategy_notes': 'YAML parsing failed, extracted decision only'
                    }
            
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML response: {e}")
            print(f"Raw response: {response_text}")
        
        # If all parsing fails, use fallback
        return self._fallbackStrategyDict()
    
    def _fallbackStrategyDict(self) -> Dict[str, Any]:
        """
        Fallback strategy when LLM is not available, returns structured data.
        
        Returns:
            Dict[str, Any]: Fallback decision with metadata
        """
        fallback_decision = self._fallbackStrategy()
        return {
            'decision': fallback_decision,
            'reasoning': 'Fallback strategy used (API unavailable)',
            'confidence': 0.5,
            'strategy_notes': 'Tit-for-Tat fallback strategy'
        }
    
    def _fallbackStrategy(self) -> str:
        """
        Fallback strategy when LLM is not available.
        Uses a simple Tit-for-Tat strategy.
        
        Returns:
            str: 'C' or 'D'
        """
        if not self.history:
            return 'C'  # Start by cooperating
        else:
            # Return opponent's last move (Tit-for-Tat)
            return self.history[-1]
    
    def strategy(self, opponent: axelrod.Player) -> str:
        """
        Determine the next move based on the game history.
        
        Args:
            opponent (axelrod.Player): The opponent player
            
        Returns:
            str: 'C' for cooperate, 'D' for defect
        """
        # Get move histories from Axelrod's tournament history
        # Axelrod manages the history automatically, we just read from it
        my_history = [move for move in self.history]
        opponent_history = [move for move in opponent.history]
        

        
        # Create the decision prompt
        decision_prompt = self._getMoveHistoryPrompt(opponent_history, my_history)
        
        # Get decision from LLM
        llm_response = self._callLLM(decision_prompt)
        
        # Store the full response for analysis
        self.last_response = llm_response
        
        decision = llm_response['decision']
        
        # Return just the decision
        return decision


class LLMStrategyWithMemory(LLMStrategy):
    """
    An enhanced LLM strategy that maintains additional context about the game.
    """
    
    name = "LLMStrategyWithMemory"
    
    def __init__(self, 
                 provider: Literal["openai", "gemini", "anthropic"] = "openai",
                 api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 game_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced LLM strategy.
        
        Args:
            provider (str): LLM provider ("openai", "gemini", or "anthropic")
            api_key (Optional[str]): API key for the provider
            model (str): Model name to use
            game_config (Optional[Dict]): Game configuration including payoff matrix, turns, noise, etc.
        """
        super().__init__(provider, api_key, model, game_config)
        self.game_context = {}
    
    def _getEnhancedContextPrompt(self) -> str:
        """
        Get an enhanced context prompt with additional strategic information.
        
        Returns:
            str: Enhanced context prompt
        """
        try:
            # Load the enhanced prompt template from external file
            prompt_template = self._loadPromptTemplate("enhancedGameRulesPrompt.txt")
        except FileNotFoundError:
            # Fallback to hardcoded prompt if file not found
            print("Warning: Could not load enhancedGameRulesPrompt.txt, using fallback prompt")
            return self._getFallbackEnhancedContextPrompt()
        
        # Get payoff matrix from game config or use defaults
        payoff_matrix = self.game_config.get('payoffMatrix', {
            'R': 3,  # Reward for mutual cooperation
            'P': 1,  # Punishment for mutual defection
            'S': 0,  # Sucker's payoff (cooperate while opponent defects)
            'T': 5   # Temptation to defect (defect while opponent cooperates)
        })
        
        # Get game length information
        turns = self.game_config.get('turns', 200)
        prob_end = self.game_config.get('probEnd')
        distribution_type = self.game_config.get('distributionType')
        
        # Get noise level
        noise = self.game_config.get('noise', 0.0)
        
        # Build game length information
        if prob_end and distribution_type:
            if distribution_type == "geometric":
                expected_length = 1 / prob_end
                game_length_info = f"""
- Game length is variable with geometric distribution
- Each turn has a {prob_end:.3f} probability of ending the game
- Expected game length is approximately {expected_length:.1f} turns
- Maximum possible turns: {turns}"""
            elif distribution_type == "normal":
                mean = self.game_config.get('distributionParams', {}).get('mean', turns)
                std = self.game_config.get('distributionParams', {}).get('std', mean * 0.2)
                game_length_info = f"""
- Game length is variable with normal distribution
- Mean game length: {mean} turns
- Standard deviation: {std:.1f} turns
- Maximum possible turns: {turns}"""
            elif distribution_type == "uniform":
                min_turns = self.game_config.get('distributionParams', {}).get('min', 10)
                max_turns = self.game_config.get('distributionParams', {}).get('max', turns)
                game_length_info = f"""
- Game length is variable with uniform distribution
- Game length ranges from {min_turns} to {max_turns} turns
- Maximum possible turns: {turns}"""
        else:
            game_length_info = f"""
- Game length is fixed at {turns} turns"""
        
        # Build noise information
        if noise > 0:
            noise_info = f"""
- There is {noise:.3f} probability that moves will be randomly flipped (noise)"""
        else:
            noise_info = ""
        
        # Build noise considerations
        if noise > 0:
            noise_considerations = f"""
- Due to noise ({noise:.3f}), some moves may be misinterpreted
- Consider that your opponent's moves might be affected by noise
- Be more forgiving of occasional defections as they might be noise-related"""
        else:
            noise_considerations = ""
        
        # Format the prompt template with dynamic values
        formatted_prompt = prompt_template.format(
            R=payoff_matrix['R'],
            P=payoff_matrix['P'],
            S=payoff_matrix['S'],
            T=payoff_matrix['T'],
            game_length_info=game_length_info,
            noise_info=noise_info,
            noise_considerations=noise_considerations
        )
        
        return formatted_prompt
    
    def _getFallbackEnhancedContextPrompt(self) -> str:
        """
        Fallback enhanced context prompt when external file is not available.
        
        Returns:
            str: Hardcoded fallback enhanced prompt
        """
        # Get payoff matrix from game config or use defaults
        payoff_matrix = self.game_config.get('payoffMatrix', {
            'R': 3, 'P': 1, 'S': 0, 'T': 5
        })
        
        return f"""You are playing the Iterated Prisoner's Dilemma, a classic game theory problem.

GAME RULES:
- You and your opponent play multiple rounds of the same game
- In each round, both players simultaneously choose to either COOPERATE (C) or DEFECT (D)
- Your goal is to maximize your total score across all rounds
- You can see the complete history of both players' moves

PAYOFF MATRIX (Your Score, Opponent's Score):
- If you COOPERATE and opponent COOPERATES: You get {payoff_matrix['R']} points, opponent gets {payoff_matrix['R']} points
- If you COOPERATE and opponent DEFECTS: You get {payoff_matrix['S']} points, opponent gets {payoff_matrix['T']} points  
- If you DEFECT and opponent COOPERATES: You get {payoff_matrix['T']} points, opponent gets {payoff_matrix['S']} points
- If you DEFECT and opponent DEFECTS: You get {payoff_matrix['P']} points, opponent gets {payoff_matrix['P']} points

STRATEGIC CONSIDERATIONS:
- The highest individual payoff ({payoff_matrix['T']} points) comes from defecting when opponent cooperates
- The lowest individual payoff ({payoff_matrix['S']} points) comes from cooperating when opponent defects
- Mutual cooperation ({payoff_matrix['R']} points each) is better than mutual defection ({payoff_matrix['P']} points each)
- Your opponent is also trying to maximize their score
- You can use the history of moves to understand your opponent's strategy

COMMON OPPONENT STRATEGIES TO RECOGNIZE:
- Always Cooperate: Opponent always plays C
- Always Defect: Opponent always plays D  
- Tit-for-Tat: Opponent starts with C, then copies your previous move
- Grudger: Opponent cooperates until you defect, then always defects
- Random: Opponent makes random moves
- Pavlov: Opponent repeats their last move if you cooperated, switches if you defected

DECISION MAKING:
- Analyze the opponent's previous moves to understand their strategy
- Consider whether they are cooperative, retaliatory, or exploitative
- Balance short-term gains with long-term relationship building
- Remember that your opponent can see your history too
- Adapt your strategy based on what you learn about the opponent

RESPONSE FORMAT:
Respond with a YAML object containing your decision and reasoning:

```yaml
decision: "C"  # or "D"
reasoning: "Brief explanation of your strategic thinking"
confidence: 0.85  # Confidence level from 0.0 to 1.0
strategy_notes: "Any additional strategic observations"
```

Example response:
```yaml
decision: "C"
reasoning: "Opponent has been cooperative, maintaining mutual cooperation is optimal"
confidence: 0.9
strategy_notes: "Detected Tit-for-Tat pattern, continuing cooperation"
```"""
    
    def strategy(self, opponent: axelrod.Player) -> str:
        """
        Enhanced strategy that includes opponent analysis.
        
        Args:
            opponent (axelrod.Player): The opponent player
            
        Returns:
            str: 'C' for cooperate, 'D' for defect
        """
        # Get move histories from Axelrod's tournament history
        # Axelrod manages the history automatically, we just read from it
        my_history = [move for move in self.history]
        opponent_history = [move for move in opponent.history]
        
        # Analyze opponent's strategy
        opponent_analysis = self._analyzeOpponent(opponent_history)
        
        # Create enhanced decision prompt
        decision_prompt = self._getEnhancedMoveHistoryPrompt(
            opponent_history, my_history, opponent_analysis
        )
        
        # Get decision from LLM
        llm_response = self._callLLM(decision_prompt)
        
        # Store the full response for analysis
        self.last_response = llm_response
        
        # Return just the decision
        return llm_response['decision']
    
    def _analyzeOpponent(self, opponent_history: List[str]) -> str:
        """
        Analyze the opponent's strategy based on their history.
        
        Args:
            opponent_history (List[str]): Opponent's move history
            
        Returns:
            str: Analysis of opponent's strategy
        """
        if not opponent_history:
            return "No history available yet."
        
        # Count moves
        cooperations = opponent_history.count('C')
        defections = opponent_history.count('D')
        total_moves = len(opponent_history)
        
        # Calculate cooperation rate
        coop_rate = cooperations / total_moves if total_moves > 0 else 0
        
        # Analyze patterns
        if total_moves == 0:
            return "First move - no pattern yet."
        elif total_moves == 1:
            return f"First move was {opponent_history[0]}."
        else:
            # Look for patterns
            if all(move == 'C' for move in opponent_history):
                return "Always Cooperate strategy detected."
            elif all(move == 'D' for move in opponent_history):
                return "Always Defect strategy detected."
            elif opponent_history[0] == 'C' and len(opponent_history) > 1:
                # Check for Tit-for-Tat
                if len(opponent_history) >= 2:
                    return f"Started with C, cooperation rate: {coop_rate:.2f}"
            else:
                return f"Mixed strategy, cooperation rate: {coop_rate:.2f}"
    
    def _getEnhancedMoveHistoryPrompt(self, opponent_history: List[str], 
                                     my_history: List[str], 
                                     opponent_analysis: str) -> str:
        """
        Create an enhanced prompt with opponent analysis.
        
        Args:
            opponent_history (List[str]): Opponent's move history
            my_history (List[str]): Your move history
            opponent_analysis (str): Analysis of opponent's strategy
            
        Returns:
            str: Enhanced prompt
        """
        if not opponent_history:
            return "This is the first round. Make your first move."
        
        # Create a conversation-style history
        history_lines = []
        for i, (my_move, opp_move) in enumerate(zip(my_history, opponent_history)):
            round_num = i + 1
            history_lines.append(f"Round {round_num}: You chose {my_move}, opponent chose {opp_move}")
        
        history_text = "\n".join(history_lines)
        current_round = len(opponent_history) + 1
        
        return f"""GAME CONVERSATION:
{history_text}

OPPONENT ANALYSIS:
{opponent_analysis}

Round {current_round}: It's your turn to make a move. Based on the conversation and analysis so far, what do you choose? Respond with only 'C' or 'D'.""" 