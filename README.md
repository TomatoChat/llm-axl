# Axelrod Tournament Simulator

A flexible and interactive command-line application that simulates the Iterated Prisoner's Dilemma using the `axelrod` library. The application allows users to configure, run, and view the results of a complete tournament through a series of clear and intuitive prompts.

## Features

- **Interactive Tournament Setup**: Guide users through configuration via command-line interface
- **Player Configuration**: Select from a comprehensive list of available strategies from the axelrod library
- **Customizable Parameters**: Configure game parameters including turns, noise, and payoff matrix
- **Clear Results Display**: View ranked results with scores and performance metrics
- **Object-Oriented Design**: Clean, modular code structure with proper separation of concerns

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd llm-axl
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
Run the tournament simulator:
```bash
python main.py
```

### Web Interface (Recommended)
Run the Streamlit web application for a user-friendly interface:
```bash
streamlit run streamlitApp.py
```

The web interface provides:
- Interactive tournament configuration
- Visual strategy selection
- Real-time tournament execution
- Interactive results visualization with charts
- Download results as CSV
- Educational information about game theory

The application will guide you through the following steps:

1. **Number of Players**: Enter how many strategies will participate in the tournament
2. **Strategy Selection**: For each player, choose from the available strategies by entering the corresponding number
3. **Game Parameters**: 
   - Number of turns per match (default: 200)
   - Noise level (0.0 to 1.0, default: 0.0)
4. **Simulation**: The tournament runs automatically
5. **Results**: View the ranked results table

## Example Session

```
Welcome to the Axelrod Tournament Simulator!

--- Tournament Setup ---
How many players will participate? 3

--- Player 1 Configuration ---
Available Strategies:
1: Cooperator
2: Defector
3: TitForTat
4: Grudger
...
Please select a strategy for Player 1: 3

--- Player 2 Configuration ---
Available Strategies:
1: Cooperator
2: Defector
3: TitForTat
4: Grudger
...
Please select a strategy for Player 2: 2

--- Player 3 Configuration ---
Available Strategies:
1: Cooperator
2: Defector
3: TitForTat
4: Grudger
...
Please select a strategy for Player 3: 1

--- Game Parameters ---
Enter number of turns per match [default: 200]:
Enter noise level (0.0 to 1.0) [default: 0.0]: 0.05

Running simulation...

--- Tournament Results ---
Rank | Strategy    | Score | Score Per Turn
---------------------------------------------
1    | Defector    | 550.5 | 2.75
2    | TitForTat   | 490.2 | 2.45
3    | Cooperator  | 420.8 | 2.10
```

## Project Structure

The application follows a clean object-oriented design with modular file structure:

### Core Files:
- **`main.py`**: Main entry point for the application (in root directory)
- **`streamlitApp.py`**: Web-based user interface using Streamlit
- **`src/`**: Source code directory containing all modules
  - **`src/tournamentSimulator.py`**: Main orchestrator class that manages the entire process
  - **`src/tournamentConfigurator.py`**: Handles all user interactions and configuration setup
  - **`src/tournamentRunner.py`**: Manages tournament execution and simulation
  - **`src/tournamentConfiguration.py`**: Data class for storing tournament configuration parameters
  - **`src/__init__.py`**: Package initialization file

### Classes:
- **`TournamentSimulator`**: Main orchestrator class that manages the entire process
- **`TournamentConfigurator`**: Handles all user interactions and configuration setup
- **`TournamentRunner`**: Manages tournament execution and simulation
- **`TournamentConfiguration`**: Data class for storing configuration parameters

## Game Parameters

The application uses the standard Axelrod payoff matrix:
- **R (Reward)**: 3 points for mutual cooperation
- **P (Punishment)**: 1 point for mutual defection  
- **S (Sucker's payoff)**: 0 points for cooperating while opponent defects
- **T (Temptation)**: 5 points for defecting while opponent cooperates

## Available Strategies

The application provides access to all strategies available in the axelrod library, including:
- **Cooperator**: Always cooperates
- **Defector**: Always defects
- **TitForTat**: Cooperates on first move, then copies opponent's previous move
- **Grudger**: Cooperates until opponent defects, then always defects
- And many more...

## Requirements

- Python 3.7+
- numpy<2.0
- axelrod==4.14.0
- streamlit>=1.28.0
- plotly>=5.0.0
- pandas>=1.5.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 