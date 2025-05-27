        ojama_bg_rect2 = (hold2_x, hold2_y + BLOCK*4 + 10, BLOCK*4, 20)
        self.sc.fill((0,0,0), ojama_bg_rect2)
        ojama_text2 = self.font.render(f"OJAMA: {self.dual.ai.garbage_buffer}", True, (255, 80, 80))
        self.sc.blit(ojama_text2, (hold2_x, hold2_y + BLOCK*4 + 10))