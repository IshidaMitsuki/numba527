# gui.py – pygame Tetris (Next-5 表示)
from __future__ import annotations
import random, json, pathlib, pygame as pg, numpy as np

from core import (MATRIX_W, MATRIX_H, COLORS, PIECE_SHAPES,
                  Piece, Board, HeuristicAI, DEFAULT_W)

# ── 定数定義 ─────────────────────────────────
BLOCK, FPS = 30, 60
LOCK_DELAY_MS, MOVE_DELAY_MS, MOVE_REPEAT_MS = 500, 190, 16
GRAVITY = [48, 43, 38, 33, 28, 23, 18, 13, 8, 6, 5, 5, 5, 4, 4, 4, 3, 3, 3, 2]

# Mini ブロックサイズ（通常の約2/3）
MINI_BLOCK = BLOCK * 2 // 3

# 画面サイズ／ボード位置／Hold/Next の配置
SCREEN_W = BLOCK * (MATRIX_W + 12)
SCREEN_H = BLOCK * (MATRIX_H + 2)
BOARD_W = MATRIX_W * BLOCK
BOARD_H = MATRIX_H * BLOCK
BOARD_X = (SCREEN_W - BOARD_W) // 2
BOARD_Y = (SCREEN_H - BOARD_H) // 2

# Hold 表示サイズ（6x6 ブロック）
HOLD_W = 6 * BLOCK
HOLD_H = 6 * BLOCK
HOLD_X = BOARD_X - HOLD_W - BLOCK
HOLD_Y = BOARD_Y

# Next-5 表示サイズ（幅6ブロック, 高さ3ブロックずつ）
NEXT_W = 6 * BLOCK
NEXT_N = 5
NEXT_BOX_H = 3 * BLOCK
NEXT_GAP = BLOCK // 2
NEXT_X = BOARD_X + BOARD_W + BLOCK
NEXT_Y = BOARD_Y

# ── AI 初期化 ─────────────────────────────────
p = pathlib.Path('best_weights.json')
weights = json.loads(p.read_text()) if p.exists() else DEFAULT_W
AI = HeuristicAI(weights, use_bb=True)

# ── Game クラス ─────────────────────────────────
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
        # ── garbage 用ステート ──
        self.last_cleared = 0    # 直前に clear() で消した行数
        self.combo        = 0    # 連鎖カウンタ
        self.spawn()
        
    def _new_bag(self): b=list('IOTSZJL'); random.shuffle(b); return b
    def next_queue(self,n): return (self.bag[::-1]+self.future_bag[::-1])[:n]
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
    def _lock(self):
        self.board.lock(self.current)
        cleared=self.board.clear()      
        if cleared>0:
            is_tetris=(cleared==4); b2b=(is_tetris and self._prev_was_tetris)
            tag = " B2B" if b2b else "  nonB2B"
            print(f"[_lock] cleared {cleared} lines{tag}")
            self._prev_was_tetris=is_tetris
            self.combo += 1
        else:
            self.combo = 0
        self.last_cleared = cleared
        if any(y>=MATRIX_H for _,y in self.current.cells()): self.game_over=True
        else: self.spawn()
    def gravity(self):
        if self.board.valid(self.current, self.current.rot, 0, -1):
            self.current.y -= 1
            self.lock_timer = None
        else:
            # フレーム数でカウントする
            if self.lock_timer is None:
                self.lock_timer = self.tick
            elif self.tick - self.lock_timer >= (LOCK_DELAY_MS * FPS // 1000):
                self._lock()
    def handle(self,e):
        if e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE: self.game_over=True
        if e.key in self.move_state:
            st=self.move_state[e.key]
            if e.type==pg.KEYDOWN:
                if e.key==pg.K_DOWN: self.soft()
                else: self.side(-1 if e.key==pg.K_LEFT else 1)
                st['held']=True; st['ts']=pg.time.get_ticks()
            else: st['held']=False
        elif e.type==pg.KEYDOWN:
            if e.key in (pg.K_z,pg.K_q,pg.K_LCTRL,pg.K_RCTRL): self.current.rotate(-1,self.board)
            elif e.key in (pg.K_x,pg.K_w,pg.K_UP): self.current.rotate(1,self.board)
            elif e.key==pg.K_SPACE: self.hard()
            elif e.key==pg.K_c: self.hold()
    def auto_repeat(self):
        now=pg.time.get_ticks()
        for k,s in self.move_state.items():
            if s['held'] and now-s['ts']>=MOVE_DELAY_MS:
                if k==pg.K_LEFT: self.side(-1)
                elif k==pg.K_RIGHT: self.side(1)
                elif k==pg.K_DOWN: self.soft()
                s['ts']+=MOVE_REPEAT_MS
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
        """下から lines 本のおじゃまラインを追加（穴ランダム）"""
        import random
        for _ in range(lines):
            hole = random.randrange(MATRIX_W)
            # 上端を一行切り上げ
            self.board.grid.pop()
            # 下におじゃまを挿入
            garbage = [(0 if x == hole else 'G') for x in range(MATRIX_W)]
            self.board.grid.insert(0, garbage)

# ── Renderer ───────────────────────────────────
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
# ── main ───────────────────────────────────────
def main():
    pg.init(); clock=pg.time.Clock()
    game=Game(ai_mode=True,use_mcp=True); rndr=Renderer(game)
    while not game.game_over:
        for e in pg.event.get():
            if e.type==pg.QUIT: return
            if e.type in (pg.KEYDOWN,pg.KEYUP): game.handle(e)
        game.auto_repeat(); game.ai_step(); game.update(); rndr.draw(); clock.tick(FPS)
    pg.quit()

if __name__=='__main__':
    main()
