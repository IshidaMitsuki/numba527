# gui.py – pygame Tetris (Single and Duel Modes)
from __future__ import annotations
import sys, random, json, pathlib, pygame as pg, numpy as np
from core import (MATRIX_W, MATRIX_H, COLORS, PIECE_SHAPES,
                  Piece, Board, HeuristicAI, DEFAULT_W)
import pathlib, json
from gui import SCREEN_W, SCREEN_H

# ── 定数定義 ─────────────────────────────────
BLOCK, FPS = 30, 60
LOCK_DELAY_MS, MOVE_DELAY_MS, MOVE_REPEAT_MS = 500, 120, 25
SOFT_DROP_REPEAT_MS = 14  # ソフトドロップ専用のリピート間隔（小さいほど速い）
GRAVITY = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2]

# Mini ブロックサイズ（通常の約2/3）
MINI_BLOCK = BLOCK * 2 // 3

# 画面サイズ単位
# ── レイアウト用定数 ──
SCREEN_W = BLOCK * (MATRIX_W + 12)
SCREEN_H = BLOCK * (MATRIX_H + 2)
BOARD_W  = MATRIX_W * BLOCK
BOARD_H  = MATRIX_H * BLOCK
BOARD_X  = (SCREEN_W - BOARD_W) // 2
BOARD_Y  = (SCREEN_H - BOARD_H) // 2

HOLD_W = 6 * BLOCK
HOLD_H = 6 * BLOCK
HOLD_X = BOARD_X - HOLD_W - BLOCK
HOLD_Y = BOARD_Y

NEXT_W      = 6 * BLOCK
NEXT_N      = 5
NEXT_BOX_H  = 3 * BLOCK
NEXT_GAP    = BLOCK // 2
NEXT_X      = BOARD_X + BOARD_W + BLOCK
NEXT_Y      = BOARD_Y

ATTACK_TABLE = {
    1: 0,   # シングル
    2: 1,   # ダブル
    3: 2,   # トリプル
    4: 4,   # テトリス
}

p = pathlib.Path('best_weights.json')
weights = json.loads(p.read_text()) if p.exists() else DEFAULT_W
AI = HeuristicAI(weights, use_bb=True)


# ── 単一プレイヤー用クラス ─────────────────────────────────
class Game:
    def __init__(self, ai_mode=True, use_mcp=False):
        self.board = Board()
        self.bag = []
        self.future_bag = self._new_bag()
        self.current = None
        self.hold_piece = None; self.can_hold = True
        self.lock_timer = None; self.game_over = False; self.tick = 0
        self.move_state = {k: {'held': False, 'ts': 0}
                           for k in (pg.K_LEFT, pg.K_RIGHT, pg.K_DOWN)}
        self.ai_mode = ai_mode; self.target_rot = 0; self.target_x = 4
        self._prev_was_tetris = False
        self.last_cleared = 0  # 直前に clear() で消した行数
        self.combo        = 0  # 連鎖カウンタ
        self.garbage_buffer = 0  # 受信したおじゃまのバッファ
        self.spawn()
        
    def _new_bag(self):
        pieces = list('IOTSZJL')
        random.shuffle(pieces)
        return pieces
        
    def next_queue(self,n): 
        return (self.bag[::-1]+self.future_bag[::-1])[:n]
    
    def _next(self):
        if not self.bag:
            self.bag, self.future_bag = self.future_bag, self._new_bag()
        return self.bag.pop()
    def spawn(self):
        self.current = Piece(self._next())
        if not self.board.valid(self.current,0): self.game_over=True; return
        self.can_hold, self.lock_timer = True, None
        if self.ai_mode:
            next4=self.next_queue(4)
            move=AI.best_move(self.board,self.current.kind,next4)
            self.target_rot,self.target_x = move if move else (0,self.current.x)
    def side(self,dx):
        if self.board.valid(self.current,self.current.rot,dx,0): self.current.x += dx
    def soft(self):
        if self.board.valid(self.current,self.current.rot,0,-1): self.current.y -= 1
    def hard(self):
        while self.board.valid(self.current,self.current.rot,0,-1): self.current.y -= 1
        self._lock()
    def hold(self):
        if not self.can_hold: return
        # hold 一度使用: spawn() によるリセット後も hold 無効を維持
        self.can_hold = False
        if self.hold_piece is None:
            self.hold_piece = Piece(self.current.kind)
            # スポーン後も hold 無効
            self.spawn()
            self.can_hold = False
        else:
            self.current.kind, self.hold_piece.kind = self.hold_piece.kind, self.current.kind
            self.current.x, self.current.y, self.current.rot = 4, 21, 0
            if not self.board.valid(self.current, 0):
                self.game_over = True
    def is_tspin(self):
        """Tスピン判定: Tミノで、回転後、3つ以上の角が埋まっている"""
        if self.current.kind != 'T':
            return False
        x, y = self.current.x, self.current.y
        # 盤面上のTミノ中心座標
        corners = [
            (x, y), (x+2, y), (x, y-2), (x+2, y-2)
        ]
        filled = 0
        for cx, cy in corners:
            # 盤外 or ブロックがあれば埋まっているとみなす
            if (cx < 0 or cx >= MATRIX_W or cy < 0 or cy >= MATRIX_H or self.board.grid[cy][cx]):
                filled += 1
        return filled >= 3

    def tspin_type(self, cleared):
        """Tスピンの種類を判定（シングル/ダブル/トリプル）"""
        if not self.is_tspin() or cleared == 0:
            return None
        if cleared == 1:
            return "single"
        elif cleared == 2:
            return "double"
        elif cleared == 3:
            return "triple"
        return None

    def _lock(self):
        tspin = self.is_tspin()  # Tスピン判定
        self.board.lock(self.current)
        cleared = self.board.clear()
        tspin_type = self.tspin_type(cleared)
        if cleared > 0:
            is_tetris = (cleared == 4)
            is_b2b = (is_tetris or tspin)
            b2b = (is_b2b and self._prev_was_tetris)
            # tag = " B2B" if b2b else "  nonB2B"
            # tspin_tag = f" T-SPIN-{tspin_type.upper()}" if tspin_type else (" T-SPIN" if tspin else "")
            # print(f"[_lock] cleared {cleared} lines{tag}{tspin_tag}")
            self._prev_was_tetris = is_b2b  # テトリスまたはTスピンでB2B継続
            self.combo += 1
        else:
            self.combo = 0
        self.last_cleared = cleared
        self.last_tspin_type = tspin_type  # 追加: 直前のTスピン種別

        # --- ここで溜まったおじゃまを盤面に追加 ---
        if self.garbage_buffer > 0:
            self._apply_garbage(self.garbage_buffer)
            self.garbage_buffer = 0

        if any(y >= MATRIX_H for _, y in self.current.cells()):
            self.game_over = True
        else:
            self.spawn()
    def gravity(self):
        if self.board.valid(self.current,self.current.rot,0,-1):
            self.current.y-=1; self.lock_timer=None
        else:
            if self.lock_timer is None: self.lock_timer=pg.time.get_ticks()
            elif pg.time.get_ticks()-self.lock_timer>=LOCK_DELAY_MS: self._lock()
    def handle(self,e):
        # ← 追加：キー以外のイベントは無視
        if e.type not in (pg.KEYDOWN, pg.KEYUP):
            return
        # ESC でゲームオーバー
        if e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
            self.game_over = True
            return
        # 左右＆↓ の移動キー
        if e.key in self.move_state:
            st = self.move_state[e.key]
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_DOWN:
                    self.soft()
                elif e.key in (pg.K_LEFT, pg.K_RIGHT):
                    self.side(-1 if e.key==pg.K_LEFT else 1)
                st['held'] = True
                st['ts'] = pg.time.get_ticks()
            else:
                st['held'] = False
            return
        # 回転・ハードドロップ・ホールド（KEYDOWNのみ）
        if e.type == pg.KEYDOWN:
            if e.key in (pg.K_z, pg.K_q, pg.K_LCTRL, pg.K_RCTRL):
                self.current.rotate(-1, self.board)
            elif e.key in (pg.K_x, pg.K_w, pg.K_UP):
                self.current.rotate(1, self.board)
            elif e.key == pg.K_SPACE:
                self.hard()
            elif e.key == pg.K_c:
                self.hold()
    def auto_repeat(self):
        now = pg.time.get_ticks()
        for k, s in self.move_state.items():
            if s['held'] and now - s['ts'] >= MOVE_DELAY_MS:
                if k == pg.K_LEFT:
                    self.side(-1)
                    s['ts'] += MOVE_REPEAT_MS
                elif k == pg.K_RIGHT:
                    self.side(1)
                    s['ts'] += MOVE_REPEAT_MS
                elif k == pg.K_DOWN:
                    self.soft()
                    s['ts'] += SOFT_DROP_REPEAT_MS  # ソフトドロップだけ速く
    def ai_step(self):
        if not self.ai_mode: return
        if self.current.rot!=self.target_rot:
            self.current.rotate(1,self.board); return
        if self.current.x<self.target_x: self.side(1); return
        if self.current.x>self.target_x: self.side(-1); return
        self.hard()
    def update(self):
        self.tick+=1
        if self.tick%GRAVITY[0]==0: self.gravity()
        
    def receive_garbage(self, lines: int):
        """おじゃまラインをバッファに溜める"""
        self.garbage_buffer += lines

    def _apply_garbage(self, lines: int):
        """下から lines 本のおじゃまラインを追加（穴ランダム）"""
        import random
        for _ in range(lines):
            hole = random.randrange(MATRIX_W)
            # 上端を切り上げ
            self.board.grid.pop()
            # 下におじゃまを挿入（0＝空穴, 'X'＝ガベージ）
            garbage = [(0 if x == hole else 'X') for x in range(MATRIX_W)]
            self.board.grid.insert(0, garbage)
            
            
class Renderer:
    def __init__(self,g):
        self.g=g; pg.display.set_caption('Tetris')
        self.sc=pg.display.set_mode((SCREEN_W,SCREEN_H))
        self.font=pg.font.SysFont('consolas',18)
    def _cell(self,x,y,c,a=255):
        if 0<=y<MATRIX_H:
            px=BOARD_X+x*BLOCK
            py=BOARD_Y+(MATRIX_H-1-y)*BLOCK
            r=pg.Rect(px,py,BLOCK,BLOCK)
            s=pg.Surface((BLOCK-1,BLOCK-1)); s.fill(c); s.set_alpha(a)
            self.sc.blit(s,(r.x+1,r.y+1))
    def _mini(self,k,ox,oy):
        for dx,dy in PIECE_SHAPES[k][0]:
            disp_dy=-dy
            px=ox+dx*MINI_BLOCK
            py=oy+disp_dy*MINI_BLOCK
            pg.draw.rect(self.sc,COLORS[k],(px,py,MINI_BLOCK,MINI_BLOCK))
    def draw(self):
        g=self.g; self.sc.fill((0,0,0))
        pg.draw.rect(self.sc,(40,40,40),(BOARD_X,BOARD_Y,BOARD_W,BOARD_H))
        for y,row in enumerate(g.board.grid):
            for x,c in enumerate(row):
                if c: self._cell(x,y,COLORS[c])
        gy=g.current.y
        while g.board.valid(g.current,g.current.rot,0,gy-g.current.y-1): gy-=1
        for x,y in g.current.cells(): self._cell(x,gy-(g.current.y-y),COLORS['G'],80)
        for x,y in g.current.cells(): self._cell(x,y,COLORS[g.current.kind])
        # Hold
        pg.draw.rect(self.sc,(100,100,100),(HOLD_X,HOLD_Y,HOLD_W,HOLD_H),2)
        self.sc.blit(self.font.render('HOLD',True,(200,200,200)),(HOLD_X,HOLD_Y-BLOCK//2))
        if g.hold_piece:
            hold_ox=HOLD_X+(HOLD_W-4*MINI_BLOCK)//2
            hold_oy=HOLD_Y+(HOLD_H-4*MINI_BLOCK)//2
            self._mini(g.hold_piece.kind,hold_ox,hold_oy)
        # Next
        self.sc.blit(self.font.render('NEXT',True,(200,200,200)),(NEXT_X,NEXT_Y-BLOCK//2))
        for i,k in enumerate(g.next_queue(NEXT_N)):
            box_y=NEXT_Y+i*(NEXT_BOX_H+NEXT_GAP)
            pg.draw.rect(self.sc,(100,100,100),(NEXT_X,box_y,NEXT_W,NEXT_BOX_H),2)
            next_ox=NEXT_X+(NEXT_W-4*MINI_BLOCK)//2
            next_oy=box_y+(NEXT_BOX_H-4*MINI_BLOCK)//2+MINI_BLOCK
            self._mini(k,next_ox,next_oy)
        pg.display.flip()

# ── 対戦（2P Duel）拡張 ─────────────────────────────────
class DualGame: 
    def __init__(self):
        # 1P: human, 2P: AI
        self.player = Game(ai_mode=False)
        self.ai = Game(ai_mode=True)
        # AI 用 HeuristicAI インスタンス
        self.ai_ai = HeuristicAI(DEFAULT_W)
        self.ai_last_kind = None
    def update(self):
        # AIのミノが変わったときだけbest_moveを呼ぶ
        if self.ai.current.kind != self.ai_last_kind:
            next4 = self.ai.next_queue(4)
            move = self.ai_ai.best_move(self.ai.board, self.ai.current.kind, next4)
            if move:
                self.ai.target_rot, self.ai.target_x = move
            self.ai_last_kind = self.ai.current.kind
        self.ai.ai_step()
        self.ai.update()

        # ② Player → AI へのガベージ送出
        pc = self.player.last_cleared
        tspin_type = getattr(self.player, "last_tspin_type", None)
        garb = 0
        if pc > 0:
            if tspin_type == "single":
                garb = 2
            elif tspin_type == "double":
                garb = 4
            elif tspin_type == "triple":
                garb = 6
            else:
                garb = ATTACK_TABLE.get(pc, 0)
                # テトリスの B2B ボーナス
                if pc == 4 and self.player._prev_was_tetris:
                    garb += 1
            # コンボボーナス（2連鎖目以降、(combo-1)//2）
            if self.player.combo > 1:
                garb += (self.player.combo - 1) // 2
            self.ai.receive_garbage(garb)

        # ③ AI → Player へのガベージ送出
        ac = self.ai.last_cleared
        ai_tspin_type = getattr(self.ai, "last_tspin_type", None)
        garb = 0
        if ac > 0:
            if ai_tspin_type == "single":
                garb = 2
            elif ai_tspin_type == "double":
                garb = 4
            elif ai_tspin_type == "triple":
                garb = 6
            else:
                garb = ATTACK_TABLE.get(ac, 0)
                if ac == 4 and self.ai._prev_was_tetris:
                    garb += 1
            if self.ai.combo > 1:
                garb += (self.ai.combo - 1) // 2
            self.player.receive_garbage(garb)

        # ④ ステートリセット
        self.player.last_cleared = 0
        self.ai.last_cleared     = 0
        self.player.last_tspin_type = None
        self.ai.last_tspin_type = None

class DuelRenderer:
    def __init__(self, dual: DualGame):
        pg.init()
        width  = SCREEN_W*2 + BLOCK*2
        # ── ① 上に2行分（BLOCK*2）の余白を確保 ──
        #     (スポーンエリアを2行見せる)
        self.TOP_MARGIN = BLOCK * 2
        # 画面高さは「上余白 + ボード本体 + 下余白(BLOCK)」
        height = self.TOP_MARGIN + BOARD_H + BLOCK
        self.sc = pg.display.set_mode((width, height))
        pg.display.set_caption('Tetris Duel')
        self.font = pg.font.SysFont('consolas', 16)
        self.dual = dual

    def _draw_cell(self, x, y, c, ox, alpha=255 ):
        px = ox + x * BLOCK
        # ── ④ Y に TOP_MARGIN を加算 ──
        py = self.TOP_MARGIN + (MATRIX_H-1-y) * BLOCK
        surf = pg.Surface((BLOCK-1, BLOCK-1), pg.SRCALPHA)
        surf.set_alpha(alpha)
        surf.fill(c)
        self.sc.blit(surf, (px+1, py+1))

    def draw_board(self, g: Game, ox: int):

        # ── 背景クリア ──
        # 上余白も含めて塗りつぶし
        pg.draw.rect(
            self.sc, (40,40,40),
            (ox, 0, BOARD_W, self.TOP_MARGIN + BOARD_H + BLOCK)
        )
        # ── ゴースト用クリア ──
        gy = g.current.y
        while g.board.valid(g.current, g.current.rot, 0, gy - g.current.y - 1):
            gy -= 1
        for x, y in g.current.cells():
            # ゴーストのセル位置を背景色でクリア
            px = ox + x * BLOCK
            py = BLOCK + (MATRIX_H - 1 - (gy - (g.current.y - y))) * BLOCK
            self.sc.fill((40,40,40), (px, py, BLOCK, BLOCK))

        # ── 現在ミノ用クリア ──
        for x, y in g.current.cells():
            # 現在ミノのセル位置を背景色でクリア
            px = ox + x * BLOCK
            py = BLOCK + (MATRIX_H - 1 - y) * BLOCK
            self.sc.fill((40,40,40), (px, py, BLOCK, BLOCK))
        # 固定ブロック
        for y,row in enumerate(g.board.grid):
            for x,c in enumerate(row):
                if c: self._draw_cell(x, y, COLORS[c], ox)

        # ゴースト
        gy = g.current.y
        while g.board.valid(g.current, g.current.rot, 0, gy-g.current.y-1): gy-=1
        for x,y in g.current.cells():
            self._draw_cell(x, gy-(g.current.y-y), COLORS['G'], ox)
            
        # 現在ミノ
        for x,y in g.current.cells():
            self._draw_cell(x, y, COLORS[g.current.kind], ox)
            
                
        # ── 枠線を最後に描画 ──
        pg.draw.rect(
            self.sc,
            (100,100,100),
            (ox, self.TOP_MARGIN, BOARD_W, BOARD_H),
            2   # 線の太さ
       )
            
        # HOLD / NEXT / その他は省略、必要に応じて各自追加
    def _mini(self, kind, ox, oy):
        """
        ミニミノを描画
        """
        for dx, dy in PIECE_SHAPES[kind][0]:
            # gui と同様に上下反転
            disp_dy = -dy
            x = ox + dx * (BLOCK//2)
            y = oy + disp_dy * (BLOCK//2)
            pg.draw.rect(self.sc, COLORS[kind], (x, y, BLOCK//2, BLOCK//2))
            
    def draw(self):
        # 左: player
        left_x = BLOCK *6
        self.draw_board(self.dual.player, left_x)
        # ── Player の Hold/Next を描画 ──
        # Hold
        hold_x = left_x - BLOCK*5   # ボード左端から 5セル分左
        hold_y = BLOCK*2             # 上から 2セル分下

        # ① ホールド領域（枠＋内側）を丸ごと黒でクリア
        clear_rect = (
            hold_x - 2,            # 枠線太さ分だけ左にはみ出し
            hold_y - 2,            # 上も同様
            BLOCK*4 + 4,           # 幅 + 左右の枠線分
            BLOCK*4 + 4            # 高さ + 上下の枠線分
        )
        self.sc.fill((0,0,0), clear_rect)
        # ② ミニピースを描画
        if self.dual.player.hold_piece:
            self._mini(self.dual.player.hold_piece.kind,
                    hold_x + BLOCK, hold_y + BLOCK)
        # ③ 枠線を最後に描画
        pg.draw.rect(self.sc, (100,100,100),
                    (hold_x, hold_y, BLOCK*4, BLOCK*4), 2)
        # おじゃまバッファ表示（Playerのホールドの下）
        ojama_bg_rect = (hold_x, hold_y + BLOCK*4 + 10, BLOCK*4, 20)  # 20は高さ、必要に応じて調整
        self.sc.fill((0,0,0), ojama_bg_rect)
        ojama_text1 = self.font.render(f"OJAMA: {self.dual.player.garbage_buffer}", True, (255, 80, 80))
        self.sc.blit(ojama_text1, (hold_x, hold_y + BLOCK*4 + 10))

        #  ── Player の Next-5 ──
        #Next-5 はボード右側、余白 BLOCK 分
        next_x = left_x + BOARD_W + BLOCK
        next_y = BLOCK*6
        # ① 古い Next 表示をクリア
        self.sc.fill((0,0,0), (next_x, next_y, BLOCK*4, BLOCK*2*5))
        # ② 枠を再描画
        pg.draw.rect(self.sc, (100,100,100),
                     (next_x, next_y, BLOCK*4, BLOCK*2*5), 2)
        # ③ 新しい Next ミニピースを描画
        for i, kind in enumerate(self.dual.player.next_queue(5)):
            oy = next_y + BLOCK + i * (BLOCK*2)
            self._mini(kind, next_x + BLOCK, oy)


        # 右: AI
        right_x = left_x + BOARD_W + BLOCK*12  # 左側 off + ボード幅 + マージン
        self.draw_board(self.dual.ai, right_x)
        
        # ── AI の Hold ──
        hold2_x = right_x - BLOCK*5
        hold2_y = BLOCK*2
        pg.draw.rect(self.sc, (100,100,100),
                     (hold2_x, hold2_y, BLOCK*4, BLOCK*4), 2)
        if self.dual.ai.hold_piece:
            self._mini(self.dual.ai.hold_piece.kind,
                       hold2_x + BLOCK, hold2_y + BLOCK)
        # おじゃまバッファ表示（AIのホールドの下）
        ojama_bg_rect2 = (hold2_x, hold2_y + BLOCK*4 + 10, BLOCK*4, 20)
        self.sc.fill((0,0,0), ojama_bg_rect2)
        ojama_text2 = self.font.render(f"OJAMA: {self.dual.ai.garbage_buffer}", True, (255, 80, 80))
        self.sc.blit(ojama_text2, (hold2_x, hold2_y + BLOCK*4 + 10))
        # ── AI の Next-5 ──
        next2_x = right_x + BOARD_W + BLOCK
        next2_y = BLOCK*6
        # ① 古い AI Next 表示をクリア
        self.sc.fill((0,0,0), (next2_x, next2_y, BLOCK*4, BLOCK*2*5))
        # ② 枠を再描画
        pg.draw.rect(self.sc, (100,100,100),
                     (next2_x, next2_y, BLOCK*4, BLOCK*2*5), 2)
        # ③ AI の Next ミニピースを描画
        for i, kind in enumerate(self.dual.ai.next_queue(5)):
            oy2 = next2_y + BLOCK + i * (BLOCK*2)
            self._mini(kind, next2_x + BLOCK, oy2)
        
        pg.display.flip()

# ── メインループ ─────────────────────────────────
def gameover_menu():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    font = pg.font.SysFont('consolas', 36)
    small_font = pg.font.SysFont('consolas', 24)
    clock = pg.time.Clock()
    selected = 0
    options = ["restart", "title", "finish"]
    while True:
        screen.fill((0, 0, 0))
        title = font.render("GAME OVER", True, (255, 100, 100))
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 80))
        for i, opt in enumerate(options):
            color = (255, 255, 0) if i == selected else (200, 200, 200)
            text = small_font.render(opt, True, color)
            screen.blit(text, (SCREEN_W//2 - text.get_width()//2, 200 + i*60))
        info = small_font.render("", True, (180,180,180))
        screen.blit(info, (SCREEN_W//2 - info.get_width()//2, 350))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif e.type == pg.KEYDOWN:
                if e.key in (pg.K_UP, pg.K_w):
                    selected = (selected - 1) % len(options)
                elif e.key in (pg.K_DOWN, pg.K_s):
                    selected = (selected + 1) % len(options)
                elif e.key in (pg.K_RETURN, pg.K_SPACE):
                    return selected
        clock.tick(30)

def main_single():
    while True:
        pg.init(); clock=pg.time.Clock()
        game = Game(ai_mode=False, use_mcp=True)
        rndr = Renderer(game)
        while not game.game_over:
            for e in pg.event.get():
                if e.type==pg.QUIT: return
                game.handle(e)
            game.auto_repeat(); game.ai_step(); game.update(); rndr.draw()
            clock.tick(FPS)
        choice = gameover_menu()
        if choice == 0:
            continue  # リスタート
        elif choice == 1:
            return    # タイトルに戻る
        else:
            pg.quit()
            sys.exit()

def main_duel():
    while True:
        duel = DualGame()
        rndr = DuelRenderer(duel)
        clock = pg.time.Clock()
        while not duel.player.game_over and not duel.ai.game_over:
            for e in pg.event.get():
                if e.type==pg.QUIT: return
                duel.player.handle(e)
            duel.player.auto_repeat()
            duel.update()
            rndr.draw()
            clock.tick(FPS)
        choice = gameover_menu()
        if choice == 0:
            continue  # リスタート
        elif choice == 1:
            return    # タイトルに戻る
        else:
            pg.quit()
            sys.exit()

def main_menu():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    font = pg.font.SysFont('consolas', 36)
    small_font = pg.font.SysFont('consolas', 24)
    clock = pg.time.Clock()
    selected = 0
    options = ["solomode", "duelmode"]
    while True:
        screen.fill((0, 0, 0))
        title = font.render("TETRIS", True, (200, 200, 255))
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 80))
        for i, opt in enumerate(options):
            color = (255, 255, 0) if i == selected else (200, 200, 200)
            text = small_font.render(opt, True, color)
            screen.blit(text, (SCREEN_W//2 - text.get_width()//2, 200 + i*60))
        info = small_font.render("", True, (180,180,180))
        screen.blit(info, (SCREEN_W//2 - info.get_width()//2, 350))
        pg.display.flip()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif e.type == pg.KEYDOWN:
                if e.key in (pg.K_UP, pg.K_w):
                    selected = (selected - 1) % len(options)
                elif e.key in (pg.K_DOWN, pg.K_s):
                    selected = (selected + 1) % len(options)
                elif e.key in (pg.K_RETURN, pg.K_SPACE):
                    return selected
        clock.tick(30)

if __name__=='__main__':
    while True:
        mode = main_menu()
        if mode == 1:
            main_duel()
        else:
            main_single()
