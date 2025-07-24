#!/usr/bin/env python3
"""
Axelrod Tournament Simulator - Streamlit Web Application

A web-based interface for the Axelrod Tournament Simulator that allows users to
configure, run, and view tournament results through an intuitive web interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import sys
import os
import random

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.tournamentConfiguration import TournamentConfiguration
from src.tournamentConfigurator import TournamentConfigurator
from src.tournamentRunner import TournamentRunner
from axelrod.strategies import all_strategies


def getAvailableStrategies() -> List[Dict[str, Any]]:
    """
    Get a list of available strategies with additional information.
    
    Returns:
        List[Dict[str, Any]]: List of strategy dictionaries
    """
    strategies = []
    for i, strategy in enumerate(all_strategies):
        strategies.append({
            'index': i + 1,
            'name': strategy.name,
            'class': strategy
        })
    return strategies


def createTournamentConfiguration() -> TournamentConfiguration:
    """
    Create tournament configuration using Streamlit widgets.
    Configuration is automatically saved on every change.
    
    Returns:
        TournamentConfiguration: Tournament configuration object
    """
    # Number of players
    numPlayers = st.number_input(
        "Number of Players",
        min_value=2,
        value=4,
        help="Select the number of players for the tournament. Note: More players create more matches (n*(n-1)/2). Large tournaments may take longer to run.",
        key="num_players"
    )
    
    # Get available strategies
    strategies = getAvailableStrategies()
    
    # Strategy selection for each player
    st.subheader("🎯 Player Strategy Selection")
    selectedStrategies = []
    
    for i in range(numPlayers):
        strategyNames = [f"{s['index']}: {s['name']}" for s in strategies]
        
        # Generate a random default index for each player to ensure different strategies
        random.seed(i)  # Use player index as seed for consistent but different defaults
        defaultIndex = random.randint(0, len(strategies) - 1)
        
        selectedIndex = st.selectbox(
            f"Player {i + 1} Strategy",
            options=range(len(strategies)),
            index=defaultIndex,
            format_func=lambda x: strategyNames[x],
            key=f"player_{i}"
        )
        selectedStrategies.append(strategies[selectedIndex]['name'])
    
    # Game parameters
    st.subheader("⚙️ Game Parameters")
    
    # Game length type selection
    gameLengthType = st.radio(
        "Game Length Type",
        ["Fixed Length", "Variable Length (Geometric)", "Variable Length (Normal)", "Variable Length (Uniform)"],
        help="Choose between fixed length or different probability distributions for variable game lengths",
        key="game_length_type"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if gameLengthType == "Fixed Length":
            turns = st.number_input(
                "Number of Turns per Match",
                min_value=10,
                max_value=1000,
                value=200,
                help="Number of iterations for each match",
                key="turns"
            )
            probEnd = None
            distributionType = None
            distributionParams = None
            
        elif gameLengthType == "Variable Length (Geometric)":
            probEnd = st.slider(
                "Probability of Ending Each Turn",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                help="Probability that each turn will be the last (geometric distribution). Based on Axelrod's prob_end parameter.",
                key="prob_end_geo"
            )
            
            # Show expected game length
            expectedLength = 1 / probEnd
            st.info(f"📊 Expected game length: {expectedLength:.1f} turns")
            
            # Set a reasonable default max turns for geometric
            turns = 1000
            distributionType = "geometric"
            distributionParams = None
            
        elif gameLengthType == "Variable Length (Normal)":
            # Normal distribution parameters
            mean = st.number_input(
                "Mean Game Length",
                min_value=10,
                max_value=1000,
                value=100,
                help="Average number of turns per game",
                key="normal_mean"
            )
            
            std = st.slider(
                "Standard Deviation",
                min_value=1.0,
                max_value=mean * 0.5,
                value=mean * 0.2,
                step=1.0,
                help="Standard deviation of game length (controls spread of bell curve)",
                key="normal_std"
            )
            
            st.info(f"📊 Bell curve centered at {mean} turns with std dev {std:.1f}")
            
            # Set max turns based on mean + 3*std for normal distribution
            turns = min(1000, int(mean + 3 * std))
            probEnd = None
            distributionType = "normal"
            distributionParams = {"mean": mean, "std": std}
            
        elif gameLengthType == "Variable Length (Uniform)":
            # Uniform distribution parameters
            min_turns = st.number_input(
                "Minimum Turns",
                min_value=10,
                max_value=500,
                value=50,
                help="Minimum number of turns per game",
                key="uniform_min"
            )
            
            max_turns = st.number_input(
                "Maximum Turns",
                min_value=min_turns,
                max_value=1000,
                value=200,
                help="Maximum number of turns per game",
                key="uniform_max"
            )
            
            expectedLength = (min_turns + max_turns) / 2
            st.info(f"📊 Uniform distribution from {min_turns} to {max_turns} turns (avg: {expectedLength:.1f})")
            
            # Set turns to max_turns for uniform distribution
            turns = max_turns
            probEnd = None
            distributionType = "uniform"
            distributionParams = {"min": min_turns, "max": max_turns}
    
    with col2:
        noise = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            help="Probability of a player's move being flipped (0.0 = no noise, 1.0 = completely random)",
            key="noise"
        )
    
    # Payoff matrix
    st.subheader("💰 Payoff Matrix")
    
    # Payoff matrix customization
    payoffType = st.radio(
        "Payoff Matrix Type",
        ["Standard Axelrod", "Custom Values"],
        help="Choose between standard Axelrod values or customize the payoff matrix",
        key="payoff_type"
    )
    
    if payoffType == "Standard Axelrod":
        st.info("Standard Axelrod payoff matrix: R=3, P=1, S=0, T=5")
        payoffMatrix = {
            'R': 3,  # Reward for mutual cooperation
            'P': 1,  # Punishment for mutual defection
            'S': 0,  # Sucker's payoff (cooperate while opponent defects)
            'T': 5   # Temptation to defect (defect while opponent cooperates)
        }
    else:
        st.info("Customize the payoff matrix values. Ensure T > R > P > S for a proper Prisoner's Dilemma.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            R = st.number_input(
                "R (Reward for mutual cooperation)",
                min_value=1,
                max_value=10,
                value=3,
                help="Points when both players cooperate",
                key="payoff_R"
            )
            
            P = st.number_input(
                "P (Punishment for mutual defection)",
                min_value=0,
                max_value=5,
                value=1,
                help="Points when both players defect",
                key="payoff_P"
            )
        
        with col2:
            S = st.number_input(
                "S (Sucker's payoff)",
                min_value=0,
                max_value=3,
                value=0,
                help="Points when you cooperate but opponent defects",
                key="payoff_S"
            )
            
            T = st.number_input(
                "T (Temptation to defect)",
                min_value=3,
                max_value=15,
                value=5,
                help="Points when you defect but opponent cooperates",
                key="payoff_T"
            )
        
        payoffMatrix = {'R': R, 'P': P, 'S': S, 'T': T}
        
        # Validate Prisoner's Dilemma conditions
        if T <= R or R <= P or P <= S:
            st.warning("⚠️ Warning: These values don't satisfy T > R > P > S. This may not be a proper Prisoner's Dilemma.")
        else:
            st.success("✅ Valid Prisoner's Dilemma payoff matrix!")
    
    # Display payoff matrix
    st.markdown("**Payoff Matrix:**")
    payoff_df = pd.DataFrame({
        'Player 2': ['Cooperate', 'Defect'],
        'Player 1 Cooperate': [f"({payoffMatrix['R']}, {payoffMatrix['R']})", f"({payoffMatrix['S']}, {payoffMatrix['T']})"],
        'Player 1 Defect': [f"({payoffMatrix['T']}, {payoffMatrix['S']})", f"({payoffMatrix['P']}, {payoffMatrix['P']})"]
    })
    st.table(payoff_df)
    
    return TournamentConfiguration(
        numPlayers=numPlayers,
        strategies=selectedStrategies,
        turns=turns,
        noise=noise,
        probEnd=probEnd,
        distributionType=distributionType,
        distributionParams=distributionParams,
        payoffMatrix=payoffMatrix
    )


def runTournament(config: TournamentConfiguration) -> Any:
    """
    Run the tournament and return results.
    
    Args:
        config (TournamentConfiguration): Tournament configuration
        
    Returns:
        Any: Tournament results
    """
    try:
        # Calculate total number of matches for progress bar
        num_matches = config.numPlayers * (config.numPlayers - 1) // 2
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update status
        status_text.text(f"Running tournament with {config.numPlayers} players ({num_matches} matches)...")
        
        # Progress callback function
        def update_progress(progress, status):
            progress_bar.progress(progress)
            status_text.text(status)
        
        # Run tournament
        runner = TournamentRunner(config)
        results = runner.runTournament(progress_callback=update_progress)
        
        # Complete progress bar
        progress_bar.progress(100)
        status_text.text("Tournament completed!")
        
        return results, runner
    except Exception as e:
        st.error(f"Error running tournament: {e}")
        return None, None


def displayResults(results: Any, runner: Any, config: TournamentConfiguration) -> None:
    """
    Display tournament results using Streamlit and Plotly.
    
    Args:
        results: Tournament results
        runner: Tournament runner object
        config: Tournament configuration
    """
    if results is None or runner is None:
        return
    
    # Get scores and calculate mean scores
    scores = results.scores
    
    # Try to get ranked names if available (Axelrod's built-in ranking)
    rankedNames = None
    try:
        if hasattr(results, 'ranked_names'):
            rankedNames = results.ranked_names
    except:
        pass
    
    # Create results data
    resultsData = []
    for i in range(len(runner.players)):
        totalScore = scores[i][0]  # First (and only) repetition
        
        # Calculate mean score - handle variable game lengths
        if config.distributionType == "geometric":
            # For geometric distribution, use expected length for mean calculation
            expectedLength = 1 / config.probEnd
            meanScore = totalScore / expectedLength
        elif config.distributionType == "normal":
            # For normal distribution, use mean parameter
            expectedLength = config.distributionParams["mean"]
            meanScore = totalScore / expectedLength
        elif config.distributionType == "uniform":
            # For uniform distribution, use average of min and max
            min_turns = config.distributionParams["min"]
            max_turns = config.distributionParams["max"]
            expectedLength = (min_turns + max_turns) / 2
            meanScore = totalScore / expectedLength
        else:
            # For fixed length games, use actual turns
            meanScore = totalScore / config.turns
            
        strategyName = runner.players[i].name
        resultsData.append({
            'Strategy': strategyName,
            'Total Score': totalScore,
            'Mean Score': meanScore,
            'Player Index': i
        })
    
    # Sort by total score (descending)
    resultsData.sort(key=lambda x: x['Total Score'], reverse=True)
    
    # Add ranking
    for i, data in enumerate(resultsData):
        data['Rank'] = i + 1
    
    # Create DataFrame
    df = pd.DataFrame(resultsData)
    
    # Display results table
    st.subheader("🏅 Final Rankings")
    
    # Show Axelrod's built-in ranking if available
    if rankedNames:
        st.info(f"📊 Axelrod's built-in ranking: {', '.join(rankedNames)}")
    
    # Show game length information for variable length games
    if config.distributionType == "geometric":
        expectedLength = 1 / config.probEnd
        st.info(f"🎲 Probabilistic endings: Each turn has {config.probEnd:.3f} chance of ending the match")
        st.info(f"📏 Expected game length: {expectedLength:.1f} turns")
    elif config.distributionType in ["normal", "uniform"]:
        st.info(f"🎲 Variable game lengths: Each match has different length based on {config.distributionType} distribution")
    
    # Format the display table
    display_df = df[['Rank', 'Strategy', 'Total Score', 'Mean Score']].copy()
    display_df['Total Score'] = display_df['Total Score'].round(1)
    display_df['Mean Score'] = display_df['Mean Score'].round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Create visualizations
    st.subheader("📈 Results Visualization")
    
    # Bar chart of total scores
    fig1 = px.bar(
        df, 
        x='Strategy', 
        y='Total Score',
        title='Total Scores by Strategy',
        color='Total Score',
        color_continuous_scale='viridis'
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Bar chart of mean scores
    fig2 = px.bar(
        df, 
        x='Strategy', 
        y='Mean Score',
        title='Mean Scores per Turn by Strategy',
        color='Mean Score',
        color_continuous_scale='plasma'
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Scatter plot of total vs mean scores
    fig3 = px.scatter(
        df,
        x='Total Score',
        y='Mean Score',
        text='Strategy',
        title='Total Score vs Mean Score per Turn',
        size='Total Score',
        color='Rank',
        color_continuous_scale='RdYlBu_r'
    )
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="axelrod_tournament_results.csv",
        mime="text/csv"
    )


def showStrategiesPage():
    """
    Display the strategies information page with descriptions and search.
    """
    st.title("📚 Strategy Library")
    st.markdown("Explore all available strategies in the Axelrod library with detailed descriptions.")
    
    # Search bar
    searchTerm = st.text_input(
        "🔍 Search Strategies",
        placeholder="Type strategy name or keywords...",
        help="Search for strategies by name or description"
    )
    
    # Get all strategies
    strategies = getAvailableStrategies()
    
    # Filter strategies based on search
    if searchTerm:
        filteredStrategies = []
        searchLower = searchTerm.lower()
        for strategy in strategies:
            if (searchLower in strategy['name'].lower() or 
                searchLower in strategy['class'].__doc__.lower() if strategy['class'].__doc__ else False):
                filteredStrategies.append(strategy)
        strategies = filteredStrategies
    
    # Display strategies
    st.subheader(f"📋 Available Strategies ({len(strategies)} found)")
    
    for strategy in strategies:
        with st.expander(f"**{strategy['name']}**", expanded=False):
            # Get strategy description
            description = strategy['class'].__doc__ or "No description available."
            
            # Clean up the description
            description = description.strip()
            if description.startswith(strategy['name']):
                description = description[len(strategy['name']):].strip()
            
            st.markdown(f"**Description:** {description}")
            
            # Add some common strategy characteristics
            strategyName = strategy['name'].lower()
            if 'cooperat' in strategyName:
                st.info("🤝 This strategy tends to cooperate")
            elif 'defect' in strategyName:
                st.warning("💔 This strategy tends to defect")
            elif 'tit' in strategyName and 'tat' in strategyName:
                st.success("🔄 This strategy uses tit-for-tat logic")
            elif 'random' in strategyName:
                st.info("🎲 This strategy uses random decisions")


def showTournamentPage():
    """
    Display the main tournament page.
    """
    # Header with tournament button in top right
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("🏆 Axelrod Tournament Simulator")
        st.markdown("""
        Welcome to the Axelrod Tournament Simulator! This application allows you to:
        - Configure tournament parameters
        - Select strategies for each player
        - Run tournaments with customizable settings
        - View detailed results and visualizations
        """)
    
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("🚀 Run Tournament", type="primary", use_container_width=True):
            if 'tournament_config' in st.session_state:
                results, runner = runTournament(st.session_state.tournament_config)
                if results is not None:
                    st.session_state.tournament_results = results
                    st.session_state.tournament_runner = runner
                    st.success("Tournament completed successfully!")
                    st.balloons()
            else:
                st.error("Please configure the tournament first!")
    
    # Main content with collapsible sections
    st.markdown("---")
    
    # Tournament Configuration Section
    with st.expander("🎮 Tournament Configuration", expanded=True):
        st.info("💡 Configuration is automatically saved on every change. No need to manually save!")
        
        config = createTournamentConfiguration()
        
        # Automatically save configuration to session state
        st.session_state.tournament_config = config
    
    # Current Configuration Section
    if 'tournament_config' in st.session_state:
        with st.expander("📋 Current Configuration", expanded=False):
            config = st.session_state.tournament_config
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Players:** {config.numPlayers}")
                if config.distributionType == "geometric":
                    st.write(f"**Game length:** Variable - Geometric (max {config.turns}, prob_end: {config.probEnd:.3f})")
                    st.write(f"**Expected length:** {1/config.probEnd:.1f} turns")
                elif config.distributionType == "normal":
                    mean = config.distributionParams["mean"]
                    std = config.distributionParams["std"]
                    st.write(f"**Game length:** Variable - Normal (max {config.turns}, μ={mean}, σ={std:.1f})")
                    st.write(f"**Expected length:** {mean} turns")
                elif config.distributionType == "uniform":
                    min_turns = config.distributionParams["min"]
                    max_turns = config.distributionParams["max"]
                    st.write(f"**Game length:** Variable - Uniform (max {config.turns}, range: {min_turns}-{max_turns})")
                    st.write(f"**Expected length:** {(min_turns + max_turns) / 2:.1f} turns")
                else:
                    st.write(f"**Game length:** Fixed ({config.turns} turns)")
                st.write(f"**Noise level:** {config.noise}")
            
            with col2:
                st.write(f"**Strategies:** {', '.join(config.strategies)}")
                if config.payoffMatrix:
                    st.write(f"**Payoff Matrix:** R={config.payoffMatrix['R']}, P={config.payoffMatrix['P']}, S={config.payoffMatrix['S']}, T={config.payoffMatrix['T']}")
    
    # Results Section
    if 'tournament_results' in st.session_state:
        with st.expander("📊 Tournament Results", expanded=True):
            displayResults(
                st.session_state.tournament_results,
                st.session_state.tournament_runner,
                st.session_state.tournament_config
            )
    else:
        with st.expander("📊 Tournament Results", expanded=False):
            st.info("No tournament results available. Please run a tournament first.")


def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="Axelrod Tournament Simulator",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "tournament"
    
    # Navigation buttons with active state indication
    tournament_active = st.session_state.current_page == "tournament"
    strategies_active = st.session_state.current_page == "strategies"
    
    # Tournament button
    if st.sidebar.button(
        "🏆 Tournament",
        type="primary" if tournament_active else "secondary",
        use_container_width=True
    ):
        st.session_state.current_page = "tournament"
        st.rerun()
    
    # Strategies button
    if st.sidebar.button(
        "📚 Strategies",
        type="primary" if strategies_active else "secondary",
        use_container_width=True
    ):
        st.session_state.current_page = "strategies"
        st.rerun()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **The Iterated Prisoner's Dilemma** is a classic game theory problem where two players repeatedly choose to either cooperate or defect.
    
    **Strategies** determine how players make their decisions based on the history of play.
    
    **Tournaments** pit multiple strategies against each other to see which performs best over many rounds.
    """)
    
    st.sidebar.header("📚 Learn More")
    st.sidebar.markdown("""
    - [Axelrod's Tournament](https://en.wikipedia.org/wiki/Axelrod%27s_tournament)
    - [Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma)
    - [Game Theory](https://en.wikipedia.org/wiki/Game_theory)
    """)
    
    # Display current page
    if st.session_state.current_page == "tournament":
        showTournamentPage()
    elif st.session_state.current_page == "strategies":
        showStrategiesPage()


if __name__ == "__main__":
    main() 