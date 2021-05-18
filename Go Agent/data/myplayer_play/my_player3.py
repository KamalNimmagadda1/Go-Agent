from copy import deepcopy
from collections import deque
from math import inf
from os import path
import queue


class Go:

    def __init__(self, n, par, now, piece, move_count):
        self.len = n
        self.piece = piece
        self.par_board = par
        self.board = now
        self.moves = move_count
        self.move_max = n * n - 1
        self.komi = n / 2
        self.direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def valid_board(self, i, j):
        return 0 <= i < self.len and 0 <= j < self.len

    def group_allies(self, i, j, board):
        allies = []
        for d in self.direction:
            neighbor = (i + d[0], j + d[1])
            if self.valid_board(neighbor[0], neighbor[1]) and board[neighbor[0]][neighbor[1]] == board[i][j]:
                allies.append(neighbor)
        return allies

    def get_ally_members(self, i, j, board):
        q = deque()
        q.append((i, j))
        ally_mem = []
        while q:
            piece1 = q.popleft()
            ally_mem.append(piece1)
            ally_neighbors = self.group_allies(piece1[0], piece1[1], board)
            for ally in ally_neighbors:
                if ally not in ally_mem:
                    q.append(ally)
        return ally_mem

    def no_liberty(self, i, j, board):
        for d in self.direction:
            neighbor = (i + d[0], j + d[1])
            if self.valid_board(neighbor[0], neighbor[1]) and board[neighbor[0]][neighbor[1]] == 0:
                return True
        return False

    def liberty(self, i, j, board):
        allies = self.get_ally_members(i, j, board)
        for p in allies:
            if self.no_liberty(p[0], p[1], board):
                return True
        return False

    def del_dead(self, board):
        rival = 3 - self.piece
        rem = False
        for i in range(self.len):
            for j in range(self.len):
                if board[i][j] == rival and not self.liberty(i, j, board):
                    allies = self.get_ally_members(i, j, board)
                    for ally in allies:
                        board[ally[0]][ally[1]] = 0
                    rem = True
        return rem

    def same_board(self, b1, b2):
        for i in range(self.len):
            for j in range(self.len):
                if b1[i][j] != b2[i][j]:
                    return False
        return True

    def move(self, i, j, action):
        board = deepcopy(self.board)
        if action == "PASS":
            return True, board
        if not self.valid_board(i, j) or not self.board[i][j] == 0:
            return False, board

        board[i][j] = self.piece
        if self.liberty(i, j, board):
            self.del_dead(board)
            return True, board
        else:
            self.del_dead(board)
            if not self.liberty(i, j, board) or self.same_board(self.par_board, board):
                return False, board

        return True, board

    def get_point_list(self):
        point_list = []
        for i in range(self.len):
            for j in range(self.len):
                if self.board[i][j] == 0:
                    point_list.append((i, j))
        return point_list

    def get_points(self):
        points = 0
        for i in range(self.len):
            for j in range(self.len):
                if self.board[i][j] == 0:
                    points += 1
        return points

    def remove_pieces(self, pos):
        board = self.board
        for piece in pos:
            board[piece[0]][piece[1]] = 0

    def score(self, piece):
        board = self.board
        scr = 0
        for i in range(self.len):
            for j in range(self.len):
                if board[i][j] == piece:
                    scr += 1
        return scr

    def judge_winner(self):
        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi:
            return 1
        elif cnt_1 < cnt_2 + self.komi:
            return 2
        else:
            return 0

    def update_board(self, new_board):
        self.board = new_board


class Player:

    def __init__(self, n, limit):
        self.len = n
        self.limit = limit
        vsl = read_input(input_file)
        self.piece = vsl[0]
        self.par = vsl[1]
        self.board = vsl[2]
        move_count = get_move_count(self.len, self.piece, self.par)
        self.state = Go(self.len, self.par, self.board, self.piece, move_count)
        self.direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def value_estimate(self, value):
        return value * (value / 81)

    def step_value(self, step):
        return (24 - step) / 24.0

    def score(self):
        return [[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]]

    def get_step(self):
        bound = 1
        step = 0
        point_count = self.state.get_points()
        while step + self.state.moves < self.state.move_max and bound < self.limit:
            bound *= point_count
            if bound >= self.limit or 2 * step >= self.state.move_max:
                break
            step += 1
            point_count -= 1
        return step

    def get_next_state(self, state, board):
        pre_state = deepcopy(state.board)
        cur_state = deepcopy(board)
        next_state = Go(state.len, pre_state, cur_state, 3 - state.piece, state.moves + 1)
        return next_state

    def get_past_state(self, state, board):
        pre_state = deepcopy(state.board)
        cur_state = deepcopy(board)
        past_state = Go(state.len, pre_state, cur_state, 3 - state.piece, state.moves - 1)
        return past_state

    def get_future_state(self, state, board):
        pre_state = deepcopy(state.board)
        st = deepcopy(board)
        next_state = self.get_next_state(state, st)
        cur_state = deepcopy(next_state.board)
        fut_state = Go(state.len, pre_state, cur_state, 3 - state.piece, state.moves + 1)
        return fut_state

    def max(self, alpha, beta, state, pass1, step, piece):
        if step == 0 or state.moves == state.move_max:
            value = self.eval(state, step, pass1, piece)
            return value, "Terminal", (-1, -1)

        rem_point_list = state.get_point_list()
        next_pos = (-1, -1)
        action = "MOVE"
        value = -inf
        next_val = -1

        for pnt in rem_point_list:
            valid, board = state.move(pnt[0], pnt[1], "MOVE")
            if not valid:
                continue
            next_state = self.get_next_state(state, board)
            next_val, _, _ = self.min(alpha, beta, next_state, False, step - 1, piece)

            if next_val > value:
                value = next_val
                next_pos = pnt
                action = "MOVE"

            if value > beta:
                return value, action, next_pos

            if value > alpha:
                alpha = next_val

        next_state = self.get_next_state(state, state.board)
        if state.same_board(state.board, state.par_board) and state.same_board(next_state.board, next_state.par_board):
            next_val = self.eval(next_state, step, True, piece)

        else:
            next_val, _, _ = self.min(alpha, beta, next_state, True, step - 1, piece)

        if next_val > value:
            value = next_val
            action = "PASS"

        if next_val > beta:
            return value, action, next_pos

        if value > alpha:
            alpha = next_val

        return value, action, next_pos

    def min(self, alpha, beta, state, pass1, step, piece):
        if step == 0 or state.moves == state.move_max:
            value = self.eval(state, step, pass1, piece)
            return value, "Terminal", (-1, -1)

        rem_point_list = state.get_point_list()
        next_pos = (-1, -1)
        action = "MOVE"
        value = inf
        next_val = -1

        for pnt in rem_point_list:
            valid, board = state.move(pnt[0], pnt[1], "MOVE")
            if not valid:
                continue
            next_state = self.get_next_state(state, board)
            next_val, _, _ = self.max(alpha, beta, next_state, False, step - 1, piece)

            if next_val < value:
                value = next_val
                next_pos = pnt
                action = "MOVE"

            if value <= alpha:
                return value, action, next_pos

            if value < beta:
                beta = value

        next_state = self.get_next_state(state, state.board)
        if state.same_board(state.board, state.par_board) and state.same_board(next_state.board, next_state.par_board):
            next_val = self.eval(next_state, step, True, piece)

        else:
            next_val, _, _ = self.max(alpha, beta, next_state, True, step - 1, piece)

        if next_val < value:
            value = next_val
            action = "PASS"

        if next_val < alpha:
            return value, action, next_pos

        if value < beta:
            beta = value

        return value, action, next_pos

    def eval(self, state, step, pass1, piece):
        value = 0
        direction = self.direction
        scores = self.score()

        for i in range(state.len):
            for j in range(state.len):
                if state.board[i][j] != 0:
                    if piece == 1:
                        if state.board[i][j] == piece:
                            value += self.value_estimate(scores[i][j]) / float(0.5 + self.step_value(step))
                        else:
                            value -= self.value_estimate(scores[i][j]) / float(0.2 + self.step_value(step))
                    else:
                        if state.board[i][j] == piece:
                            value += self.value_estimate(scores[i][j]) / float(0.2 + self.step_value(step))
                        else:
                            value -= self.value_estimate(scores[i][j]) / float(0.5 + self.step_value(step))
                    continue

                for d in direction:
                    me = False
                    rival = False
                    neighbor = (i + d[0], j + d[1])
                    if state.valid_board(neighbor[0], neighbor[1]):
                        if not me:
                            if state.board[neighbor[0]][neighbor[1]] == piece:
                                value += self.value_estimate(scores[i][j]) * self.step_value(step)
                                me = True
                        if not rival:
                            if state.board[neighbor[0]][neighbor[1]] == 3 - piece:
                                value -= self.value_estimate(scores[i][j]) * self.step_value(step)
                                rival = True
                        if me and rival:
                            break

        return value + (2.5 * (3 - piece)) * 5

    def minimax(self):
        state = deepcopy(self.state)
        if state.moves == 0 and state.piece == 1:
            return "MOVE", 2, 2
        if state.moves == 1 and state.piece == 2 and state.board[2][2] == 0:
            return "MOVE", 2, 2
        if state.moves == 1 and state.piece == 2 and state.board[2][2] == 1:
            return "MOVE", 1, 2
        alpha = -inf
        beta = inf
        d = self.get_step()
        pass1 = state.same_board(state.par_board, state.board)
        _, action, pos = self.max(alpha, beta, state, pass1, d, self.piece)
        return action, pos[0], pos[1]


def print_output(action, x, y, path="output.txt"):
    with open(path, "w") as o:
        if action == "PASS":
            o.write("PASS")
        else:
            o.write(str(x) + "," + str(y))
        o.close()


def is_start(n, board):
    for i in range(n):
        for j in range(n):
            if board[i][j] != 0:
                return True
    return False


def get_move_count(n, piece, par):
    if not is_start(n, par):
        with open(move_file, "w") as m:
            move_count = piece - 1
            m.write(str(move_count + 2))
            m.close()
    else:
        with open(move_file, "r") as m:
            c = m.readlines()
            move_count = int(c[0])
        with open(move_file, "w") as m:
            m.write(str(move_count + 2))
            m.close()
    return move_count



def read_input(path="input.txt"):
    with open(path, 'r') as f:
        ips = f.readlines()
        p = int(ips[0])
        par_board = [[int(x) for x in ip.rstrip('\n')] for ip in ips[1: n + 1]]
        bd = [[int(x) for x in ip.rstrip('\n')] for ip in ips[n + 1: 2 * n + 1]]
    return (p, par_board, bd)


if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    move_file = "move.txt"
    n = 5
    limit = 20000
    agent = Player(n, limit)
    action, x, y = agent.minimax()
    print_output(action, x, y, output_file)


