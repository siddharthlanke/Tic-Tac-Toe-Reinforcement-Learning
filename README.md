# Tic-Tac-Toe-Reinforcement-Learning
This repository hosts a Python project that utilizes reinforcement learning to build an intelligent Tic-Tac-Toe game-playing agent. The project involves training an AI agent to play Tic-Tac-Toe through a series of reinforcement learning tasks, enabling it to learn the optimal strategies for the game. The project also includes a human-vs-AI game mode for interactive play.

## Project Details

- The game environment is created using Python and NumPy, and the AI agent is trained through reinforcement learning methods.
- The project includes a human-vs-AI game mode, where you can test your skills against the trained AI agent.
- The reinforcement learning agent is saved and loaded using pickle to preserve learning progress.

## How to Play

1. **Training the Agent**: The AI agent can be trained by running the `play` method in the provided code. You can adjust the number of training rounds for learning.
   ```python
   st.play(10000)  # Train the AI agent for 10,000 rounds (you can adjust this number)
   ```

2. **Save Trained Policy**: After training, save the trained policy using the `savePolicy` method.
   ```python
   p1.savePolicy()  # Save the trained policy
   ```

3. **Load Trained Policy**: To load the trained policy for the AI agent, use the `loadPolicy` method.
   ```python
   p1.loadPolicy("policy_p1")  # Load the trained policy
   ```

4. **Play with the AI**: You can play against the trained AI using the `play2` method. The AI agent will apply its learned strategies.
   ```python
   st.play2()  # Play against the trained AI
   ```

5. **Interactive Gameplay**: Enjoy a game of Tic-Tac-Toe and test your skills against the AI agent.

## Requirements

- Python (3.x)
- NumPy (for game logic and AI)
- Pickle (for saving and loading trained policies)

## Repository Structure

- `tictactoe_reinforcement.py`: The Python code for the Tic-Tac-Toe game and the reinforcement learning agent.
- `policy_p1`: A sample trained policy file (you can replace it with your own trained policy).

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.

Enjoy playing and exploring the world of reinforcement learning with Tic-Tac-Toe!
