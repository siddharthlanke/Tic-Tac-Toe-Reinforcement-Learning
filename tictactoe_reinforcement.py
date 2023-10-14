import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=10000):
        for i in range(rounds):
            if i % 1000 == 0:
                print(f"Rounds {i}")
            self.reset()
            while not self.isEnd:
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    break
                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    win = self.winner()
                    if win is not None:
                        self.giveReward()

    def play2(self):
        while not self.isEnd:
            self.showBoard()
            positions = self.availablePositions()
            if not positions:
                print("Tie!")
                break

            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            win = self.winner()
            if win is not None:
                self.showBoard()
                if win == 1:
                    print(f"{self.p1.name} wins!")
                else:
                    print("Tie!")
                break
            else:
                self.showBoard()
                positions = self.availablePositions()
                if not positions:
                    print("Tie!")
                    break

                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                win = self.winner()
                if win is not None:
                    self.showBoard()
                    if win == -1:
                        print(f"{self.p2.name} wins!")
                    else:
                        print("Tie!")
                    break

    def showBoard(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        with open(f'policy_{self.name}', 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def loadPolicy(self, file):
        with open(file, 'rb') as fr:
            self.states_value = pickle.load(fr)

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board, symbol):
        while True:
            try:
                row = int(input("Input your action row: "))
                col = int(input("Input your action col: "))
                action = (row, col)
                if action in positions:
                    return action
                else:
                    print("Invalid move. Please try again.")
            except ValueError:
                print("Invalid input. Please enter row and column as integers.")

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("Training...")
    st.play(1000)
    p1.savePolicy()  
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")  
    p2 = HumanPlayer("human")

    st = State(p1, p2)
    print("Let's play!")
    st.play2()

