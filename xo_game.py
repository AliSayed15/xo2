# ============================================================
#   XO Game - vs AI using Search Algorithms
#   Algorithms: BFS / UCS / DFS
#   pip install pygame
# ============================================================

import pygame
import sys
import random
import math
from collections import deque
import heapq
from enum import Enum, auto
from typing import Optional, List, Tuple


# ── Constants ────────────────────────────────────────────────
WINDOW_SIZE  = 540
GRID_SIZE    = 3
CELL_SIZE    = WINDOW_SIZE // GRID_SIZE
LINE_WIDTH   = 6
PIECE_WIDTH  = 10
RADIUS       = CELL_SIZE // 3
AI_THINK_MS  = 600

BG_COLOR     = ( 15,  15,  20)
LINE_COLOR   = ( 50,  50,  65)
X_COLOR      = (231,  76,  60)
O_COLOR      = ( 52, 152, 219)
WIN_COLOR    = (241, 196,  15)
TEXT_COLOR   = (236, 240, 241)
MUTED_COLOR  = (127, 140, 141)
PANEL_COLOR  = ( 25,  25,  35)
BTN_COLOR    = ( 44,  62,  80)
BTN_HOVER    = ( 52,  73,  94)
BFS_COLOR    = ( 39, 174,  96)   # أخضر
UCS_COLOR    = (230, 126,  34)   # برتقالي
DFS_COLOR    = (192,  57,  43)   # أحمر


class Algorithm(Enum):
    BFS = auto()
    UCS = auto()
    DFS = auto()


class GameState(Enum):
    MENU      = auto()
    PLAYING   = auto()
    AI_THINK  = auto()
    GAME_OVER = auto()


# ── Board Logic ──────────────────────────────────────────────

class Board:
    EMPTY = 0
    HUMAN = 1    # X
    AI    = -1   # O

    def __init__(self, cells=None):
        if cells is None:
            self.cells: List[List[int]] = [[self.EMPTY]*3 for _ in range(3)]
        else:
            self.cells = [row[:] for row in cells]

    def clone(self) -> "Board":
        return Board(self.cells)

    def reset(self):
        self.cells = [[self.EMPTY]*3 for _ in range(3)]

    def make_move(self, row: int, col: int, player: int) -> bool:
        if self.cells[row][col] == self.EMPTY:
            self.cells[row][col] = player
            return True
        return False

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(3) for c in range(3)
                if self.cells[r][c] == self.EMPTY]

    def check_winner(self) -> Optional[int]:
        b = self.cells
        lines = [
            [b[0][0], b[0][1], b[0][2]],
            [b[1][0], b[1][1], b[1][2]],
            [b[2][0], b[2][1], b[2][2]],
            [b[0][0], b[1][0], b[2][0]],
            [b[0][1], b[1][1], b[2][1]],
            [b[0][2], b[1][2], b[2][2]],
            [b[0][0], b[1][1], b[2][2]],
            [b[0][2], b[1][1], b[2][0]],
        ]
        for line in lines:
            if line[0] != self.EMPTY and line[0] == line[1] == line[2]:
                return line[0]
        return None

    def get_winning_line(self) -> Optional[Tuple[Tuple, Tuple]]:
        b = self.cells
        checks = [
            ((0,0),(0,1),(0,2)), ((1,0),(1,1),(1,2)), ((2,0),(2,1),(2,2)),
            ((0,0),(1,0),(2,0)), ((0,1),(1,1),(2,1)), ((0,2),(1,2),(2,2)),
            ((0,0),(1,1),(2,2)), ((0,2),(1,1),(2,0)),
        ]
        for (r1,c1),(r2,c2),(r3,c3) in checks:
            if (b[r1][c1] != self.EMPTY and
                    b[r1][c1] == b[r2][c2] == b[r3][c3]):
                start = (c1*CELL_SIZE + CELL_SIZE//2, r1*CELL_SIZE + CELL_SIZE//2)
                end   = (c3*CELL_SIZE + CELL_SIZE//2, r3*CELL_SIZE + CELL_SIZE//2)
                return (start, end)
        return None

    def is_full(self) -> bool:
        return all(self.cells[r][c] != self.EMPTY
                   for r in range(3) for c in range(3))

    def to_tuple(self) -> tuple:
        """تحويل اللوحة لـ tuple عشان نقدر نحطها في set (للـ visited)."""
        return tuple(self.cells[r][c] for r in range(3) for c in range(3))

    def evaluate(self) -> int:
        """
        دالة تقييم بسيطة للحالة الحالية للوحة.
        بترجع:
          +10  لو AI فايز
          -10  لو HUMAN فايز
           0   تعادل أو ما انتهتش
        مع bonus للمركز والزوايا عشان تحسن الاختيار.
        """
        winner = self.check_winner()
        if winner == self.AI:    return 10
        if winner == self.HUMAN: return -10

        score = 0
        # المركز مهم
        if self.cells[1][1] == self.AI:    score += 3
        if self.cells[1][1] == self.HUMAN: score -= 3
        # الزوايا
        corners = [(0,0),(0,2),(2,0),(2,2)]
        for r, c in corners:
            if self.cells[r][c] == self.AI:    score += 1
            if self.cells[r][c] == self.HUMAN: score -= 1
        return score


# ── Search Algorithms ─────────────────────────────────────────
#
#  كل خوارزمية بتاخد اللوحة الحالية وبترجع أحسن حركة للـ AI
#
#  State = (board_tuple, move_that_led_here, depth, whose_turn)
#
# ─────────────────────────────────────────────────────────────

class SearchAI:
    def __init__(self, algorithm: Algorithm):
        self.algorithm = algorithm

    def get_move(self, board: Board) -> Tuple[int, int]:
        empty = board.get_empty_cells()
        if not empty:
            return (-1, -1)

        if self.algorithm == Algorithm.BFS:
            return self._bfs(board)
        elif self.algorithm == Algorithm.UCS:
            return self._ucs(board)
        else:
            return self._dfs(board)

    # ── BFS ──────────────────────────────────────────────────
    # بيستكشف كل الحالات مستوى مستوى (layer by layer).
    # بيختار الحركة اللي تودي لأسرع فوز أو تمنع أسرع خسارة.
    def _bfs(self, board: Board) -> Tuple[int, int]:
        """
        Queue: (board_tuple, first_move, current_player)
        بنحتفظ بـ first_move عشان نعرف أول حركة اتخدت من الجذر.
        """
        best_move   = board.get_empty_cells()[0]
        best_score  = -math.inf
        visited     = set()

        # كل عنصر: (لوحة كـ tuple, أول حركة, اللاعب الحالي, عمق)
        queue = deque()
        for (r, c) in board.get_empty_cells():
            new_board = board.clone()
            new_board.make_move(r, c, Board.AI)
            queue.append((new_board, (r, c), Board.HUMAN, 1))

        while queue:
            cur_board, first_move, player, depth = queue.popleft()
            state_key = (cur_board.to_tuple(), player)

            if state_key in visited:
                continue
            visited.add(state_key)

            winner = cur_board.check_winner()
            if winner or cur_board.is_full():
                score = cur_board.evaluate() - depth  # أسرع فوز أفضل
                if score > best_score:
                    best_score = score
                    best_move  = first_move
                continue

            # توليد الحالات التالية
            next_player = Board.HUMAN if player == Board.AI else Board.AI
            for (r, c) in cur_board.get_empty_cells():
                nb = cur_board.clone()
                nb.make_move(r, c, player)
                queue.append((nb, first_move, next_player, depth + 1))

        return best_move

    # ── UCS ──────────────────────────────────────────────────
    # Uniform-Cost Search: بيعطي كل حركة تكلفة وبيختار المسار الأقل تكلفة.
    # التكلفة هنا = عمق الحركة (depth) + عقوبة على الحالات السيئة.
    def _ucs(self, board: Board) -> Tuple[int, int]:
        """
        Priority Queue: (cost, counter, board_tuple, first_move, player)
        التكلفة = depth - evaluate()  (بنحول maximize لـ minimize)
        """
        best_move  = board.get_empty_cells()[0]
        best_score = -math.inf
        visited    = set()
        counter    = 0  # عشان nbreak ties في الـ heap

        heap = []
        for (r, c) in board.get_empty_cells():
            new_board = board.clone()
            new_board.make_move(r, c, Board.AI)
            cost = 1 - new_board.evaluate()  # تكلفة = عمق - تقييم
            heapq.heappush(heap, (cost, counter, new_board, (r, c), Board.HUMAN, 1))
            counter += 1

        while heap:
            cost, _, cur_board, first_move, player, depth = heapq.heappop(heap)
            state_key = (cur_board.to_tuple(), player)

            if state_key in visited:
                continue
            visited.add(state_key)

            winner = cur_board.check_winner()
            if winner or cur_board.is_full():
                score = cur_board.evaluate()
                if score > best_score:
                    best_score = score
                    best_move  = first_move
                continue

            next_player = Board.HUMAN if player == Board.AI else Board.AI
            for (r, c) in cur_board.get_empty_cells():
                nb = cur_board.clone()
                nb.make_move(r, c, player)
                new_cost = depth + 1 - nb.evaluate()
                heapq.heappush(heap, (new_cost, counter, nb, first_move,
                                      next_player, depth + 1))
                counter += 1

        return best_move

    # ── DFS ──────────────────────────────────────────────────
    # Depth-First Search: بيغوص لأعمق نقطة أول قبل ما يرجع.
    # بيستخدم Stack وبيقيّم الحالات النهائية ويختار أحسنها.
    def _dfs(self, board: Board) -> Tuple[int, int]:
        """
        Stack: (board, first_move, player, depth)
        بيغوص لـ depth=9 (كل الحالات) وبيختار أحسن تقييم.
        """
        best_move  = board.get_empty_cells()[0]
        best_score = -math.inf
        visited    = set()

        stack = []
        for (r, c) in board.get_empty_cells():
            new_board = board.clone()
            new_board.make_move(r, c, Board.AI)
            stack.append((new_board, (r, c), Board.HUMAN, 1))

        while stack:
            cur_board, first_move, player, depth = stack.pop()  # LIFO
            state_key = (cur_board.to_tuple(), player)

            if state_key in visited:
                continue
            visited.add(state_key)

            winner = cur_board.check_winner()
            if winner or cur_board.is_full() or depth >= 9:
                score = cur_board.evaluate()
                if score > best_score:
                    best_score = score
                    best_move  = first_move
                continue

            next_player = Board.HUMAN if player == Board.AI else Board.AI
            for (r, c) in cur_board.get_empty_cells():
                nb = cur_board.clone()
                nb.make_move(r, c, player)
                stack.append((nb, first_move, next_player, depth + 1))

        return best_move


# ── Renderer ─────────────────────────────────────────────────

class Renderer:
    PANEL_H = 160

    def __init__(self, screen: pygame.Surface):
        self.screen  = screen
        self.font_xl = pygame.font.SysFont("Arial", 52, bold=True)
        self.font_lg = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 21)
        self.font_sm = pygame.font.SysFont("Arial", 16)

    def _cx(self, surf: pygame.Surface) -> int:
        return (WINDOW_SIZE - surf.get_width()) // 2

    # ── Menu ─────────────────────────────────────────────────

    def draw_menu(self, hovered: Optional[str]):
        self.screen.fill(BG_COLOR)

        title = self.font_xl.render("X  O", True, TEXT_COLOR)
        self.screen.blit(title, (self._cx(title), 40))

        sub = self.font_md.render("Choose a Search Algorithm for the AI", True, MUTED_COLOR)
        self.screen.blit(sub, (self._cx(sub), 108))

        algos = [
            ("BFS  -  Breadth-First Search", "bfs", BFS_COLOR, 185),
            ("UCS  -  Uniform-Cost Search",  "ucs", UCS_COLOR, 275),
            ("DFS  -  Depth-First Search",   "dfs", DFS_COLOR, 365),
        ]
        for label, key, color, y in algos:
            rect = pygame.Rect(WINDOW_SIZE//2 - 185, y, 370, 60)
            hover = hovered == key
            pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN_COLOR,
                             rect, border_radius=12)
            pygame.draw.rect(self.screen, color, rect, width=3, border_radius=12)
            txt = self.font_md.render(label, True, color)
            self.screen.blit(txt, (rect.centerx - txt.get_width()//2,
                                   rect.centery - txt.get_height()//2))

        hint = self.font_sm.render("You are X   |   AI is O", True, MUTED_COLOR)
        self.screen.blit(hint, (self._cx(hint), 460))

        hint2 = self.font_sm.render("Click an algorithm to start", True, MUTED_COLOR)
        self.screen.blit(hint2, (self._cx(hint2), 484))

    def get_menu_btn(self, pos) -> Optional[str]:
        for key, y in [("bfs", 185), ("ucs", 275), ("dfs", 365)]:
            if pygame.Rect(WINDOW_SIZE//2 - 185, y, 370, 60).collidepoint(pos):
                return key
        return None

    # ── Game ─────────────────────────────────────────────────

    def draw_game(self, board: Board, state: GameState,
                  winner: Optional[int], algo: Algorithm,
                  score: dict, thinking: bool):

        self.screen.fill(BG_COLOR, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         (0, WINDOW_SIZE, WINDOW_SIZE, self.PANEL_H))
        pygame.draw.line(self.screen, LINE_COLOR,
                         (0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE), 2)

        self._draw_grid()
        self._draw_pieces(board)

        if winner is not None:
            line = board.get_winning_line()
            if line:
                pygame.draw.line(self.screen, WIN_COLOR,
                                 line[0], line[1], LINE_WIDTH + 4)

        self._draw_panel(algo, score, state, winner, thinking)

    def _draw_grid(self):
        for i in range(1, GRID_SIZE):
            pygame.draw.line(self.screen, LINE_COLOR,
                             (i*CELL_SIZE, 20), (i*CELL_SIZE, WINDOW_SIZE-20), LINE_WIDTH)
            pygame.draw.line(self.screen, LINE_COLOR,
                             (20, i*CELL_SIZE), (WINDOW_SIZE-20, i*CELL_SIZE), LINE_WIDTH)

    def _draw_pieces(self, board: Board):
        for r in range(3):
            for c in range(3):
                val = board.cells[r][c]
                cx  = c * CELL_SIZE + CELL_SIZE // 2
                cy  = r * CELL_SIZE + CELL_SIZE // 2
                if val == Board.HUMAN:
                    off = RADIUS - 15
                    pygame.draw.line(self.screen, X_COLOR,
                                     (cx-off, cy-off), (cx+off, cy+off), PIECE_WIDTH)
                    pygame.draw.line(self.screen, X_COLOR,
                                     (cx+off, cy-off), (cx-off, cy+off), PIECE_WIDTH)
                elif val == Board.AI:
                    pygame.draw.circle(self.screen, O_COLOR,
                                       (cx, cy), RADIUS, PIECE_WIDTH)

    def _draw_panel(self, algo: Algorithm, score: dict,
                    state: GameState, winner: Optional[int], thinking: bool):
        Y = WINDOW_SIZE

        algo_info = {
            Algorithm.BFS: ("BFS", BFS_COLOR),
            Algorithm.UCS: ("UCS", UCS_COLOR),
            Algorithm.DFS: ("DFS", DFS_COLOR),
        }
        algo_name, algo_color = algo_info[algo]

        # Algorithm badge
        badge = self.font_sm.render(f"AI Algorithm: {algo_name}", True, algo_color)
        self.screen.blit(badge, (20, Y + 14))

        # Score
        sc = self.font_sm.render(
            f"You: {score['human']}   Draw: {score['draw']}   AI: {score['ai']}",
            True, MUTED_COLOR)
        self.screen.blit(sc, (20, Y + 35))

        # Main message
        if thinking:
            msg   = f"AI ({algo_name}) is searching...  [*]"
            color = algo_color
        elif state == GameState.GAME_OVER:
            if winner == Board.HUMAN:
                msg, color = "You Win!   \\(^o^)/", BFS_COLOR
            elif winner == Board.AI:
                msg, color = f"AI ({algo_name}) Wins!   (>_<)", DFS_COLOR
            else:
                msg, color = "Draw!   (-_-)", UCS_COLOR
        else:
            msg, color = "Your turn!  Click a cell.", TEXT_COLOR

        msg_surf = self.font_lg.render(msg, True, color)
        self.screen.blit(msg_surf, (self._cx(msg_surf), Y + 62))

        self._draw_bottom_btns()

    def _draw_bottom_btns(self):
        mouse = pygame.mouse.get_pos()
        Y = WINDOW_SIZE
        btn_r = pygame.Rect(WINDOW_SIZE//2 + 10,  Y + 108, 155, 36)
        btn_m = pygame.Rect(WINDOW_SIZE//2 - 170, Y + 108, 155, 36)
        for btn, label in [(btn_r, "New Game  [R]"), (btn_m, "[M]  Menu")]:
            hover = btn.collidepoint(mouse)
            pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN_COLOR,
                             btn, border_radius=8)
            pygame.draw.rect(self.screen, LINE_COLOR, btn, width=1, border_radius=8)
            t = self.font_sm.render(label, True, TEXT_COLOR)
            self.screen.blit(t, (btn.centerx - t.get_width()//2,
                                 btn.centery - t.get_height()//2))

    def get_game_btn(self, pos) -> Optional[str]:
        Y = WINDOW_SIZE
        if pygame.Rect(WINDOW_SIZE//2 + 10,  Y + 108, 155, 36).collidepoint(pos): return "restart"
        if pygame.Rect(WINDOW_SIZE//2 - 170, Y + 108, 155, 36).collidepoint(pos): return "menu"
        return None


# ── Main Game Controller ──────────────────────────────────────

class XOGame:
    PANEL_H = 160

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("XO  -  Search Algorithms AI")
        self.screen    = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + self.PANEL_H))
        self.clock     = pygame.time.Clock()
        self.renderer  = Renderer(self.screen)
        self.board     = Board()
        self.ai        = SearchAI(Algorithm.BFS)
        self.algorithm = Algorithm.BFS
        self.state     = GameState.MENU
        self.winner: Optional[int] = None
        self.score          = {"human": 0, "ai": 0, "draw": 0}
        self.ai_think_timer = 0
        self.hovered_btn: Optional[str] = None

    def _start_game(self, algorithm: Algorithm):
        self.algorithm = algorithm
        self.ai        = SearchAI(algorithm)
        self.board.reset()
        self.winner = None
        self.state  = GameState.PLAYING

    def _restart(self):
        self.board.reset()
        self.winner = None
        self.state  = GameState.PLAYING

    def _handle_menu_events(self):
        mouse = pygame.mouse.get_pos()
        self.hovered_btn = self.renderer.get_menu_btn(mouse)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                btn = self.renderer.get_menu_btn(event.pos)
                if btn == "bfs": self._start_game(Algorithm.BFS)
                elif btn == "ucs": self._start_game(Algorithm.UCS)
                elif btn == "dfs": self._start_game(Algorithm.DFS)

    def _handle_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.state == GameState.GAME_OVER:
                    self._restart(); return
                if event.key == pygame.K_m:
                    self.state = GameState.MENU; return
            if event.type == pygame.MOUSEBUTTONDOWN:
                btn = self.renderer.get_game_btn(event.pos)
                if btn == "restart": self._restart(); return
                if btn == "menu":    self.state = GameState.MENU; return
                if self.state == GameState.PLAYING:
                    mx, my = event.pos
                    if my < WINDOW_SIZE:
                        col = mx // CELL_SIZE
                        row = my // CELL_SIZE
                        if self.board.make_move(row, col, Board.HUMAN):
                            winner = self.board.check_winner()
                            if winner or self.board.is_full():
                                self._end_game(winner)
                            else:
                                self.state = GameState.AI_THINK
                                self.ai_think_timer = pygame.time.get_ticks()

    def _update_ai(self):
        if self.state != GameState.AI_THINK:
            return
        if pygame.time.get_ticks() - self.ai_think_timer >= AI_THINK_MS:
            move = self.ai.get_move(self.board)
            if move != (-1, -1):
                self.board.make_move(move[0], move[1], Board.AI)
            winner = self.board.check_winner()
            if winner or self.board.is_full():
                self._end_game(winner)
            else:
                self.state = GameState.PLAYING

    def _end_game(self, winner: Optional[int]):
        self.winner = winner
        self.state  = GameState.GAME_OVER
        if winner == Board.HUMAN: self.score["human"] += 1
        elif winner == Board.AI:  self.score["ai"]    += 1
        else:                     self.score["draw"]  += 1

    def run(self):
        while True:
            if self.state == GameState.MENU:
                self._handle_menu_events()
                self.renderer.draw_menu(self.hovered_btn)
            else:
                self._handle_game_events()
                self._update_ai()
                thinking = (self.state == GameState.AI_THINK)
                self.renderer.draw_game(
                    self.board, self.state, self.winner,
                    self.algorithm, self.score, thinking
                )
            pygame.display.flip()
            self.clock.tick(60)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    game = XOGame()
    game.run()