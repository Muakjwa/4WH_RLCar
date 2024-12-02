import pygame
import math
import numpy as np

screen_width = 1000
screen_height = 600
mode_height = 120

car_width = 27
car_height = 20
check_point = ((670, 550), (150, 530), (100, 250), (115, 100), (500, 70), (950, 70))

class Car:
    def __init__(self, car_file, map_file, pos):
        self.surface = pygame.image.load(car_file)
        self.map = pygame.image.load(map_file)
        self.surface = pygame.transform.scale(self.surface, (car_width, car_height))
        self.rotate_surface = self.surface
        self.pos = pos
        self.move = 0
        self.mode = 'G'
        self.angle = np.random.randint(-15, 16)
        self.speed = 0
        self.center = [self.pos[0] + car_width/2, self.pos[1] + car_height/2]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.goal = False
        self.check_flag = False
        self.distance = 0
        self.time_spent = 0
        for d in range(90, 300, 90):
            self.check_radar(d)

        for d in range(90, 300, 90):
            self.check_radar_for_draw(d)

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)

    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 1)

    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self):
        self.is_alive = True
        for p in self.four_points:
            if p[0] < 0 or p[0] > screen_width or p[1] < 0 or p[1] > screen_height:
                self.is_alive = False
                return
        for p in self.four_points:
            if self.map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 70:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


    def check_radar_for_draw(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 70:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])

    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 90:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True
            else:
                self.goal = False

        self.cur_distance = dist

    def update(self):
        self.rotate_surface = rot_center(self.surface, self.angle)
        if self.move != 0:
            if self.mode == 'G':
                self.pos[0] += math.cos(math.radians(180 - self.angle)) * self.move
                if self.pos[0] < car_width/2:
                    self.pos[0] = car_width/2
                elif self.pos[0] > screen_width - car_width/2:
                    self.pos[0] = screen_width - car_width/2

                self.distance += self.move
                self.time_spent += 1

                self.pos[1] += math.sin(math.radians(180 - self.angle)) * self.move
                if self.pos[1] < car_height/2:
                    self.pos[1] = car_height/2
                elif self.pos[1] > screen_height - car_height/2:
                    self.pos[1] = screen_height - car_height/2

            elif self.mode == 'DR':
                self.pos[0] += math.cos(math.radians(180 - self.angle - 15)) * self.move
                if self.pos[0] < car_width/2:
                    self.pos[0] = car_width/2
                elif self.pos[0] > screen_width - car_width/2:
                    self.pos[0] = screen_width - car_width/2

                self.distance += self.move
                self.time_spent += 1

                self.pos[1] += math.sin(math.radians(180 - self.angle - 15)) * self.move
                if self.pos[1] < car_height/2:
                    self.pos[1] = car_height/2
                elif self.pos[1] > screen_height - car_height/2:
                    self.pos[1] = screen_height - car_height/2

            elif self.mode == 'DL':
                self.pos[0] += math.cos(math.radians(180 - self.angle + 15)) * self.move
                if self.pos[0] < car_width/2:
                    self.pos[0] = car_width/2
                elif self.pos[0] > screen_width - car_width/2:
                    self.pos[0] = screen_width - car_width/2

                self.distance += self.move
                self.time_spent += 1

                self.pos[1] += math.sin(math.radians(180 - self.angle + 15)) * self.move
                if self.pos[1] < car_height/2:
                    self.pos[1] = car_height/2
                elif self.pos[1] > screen_height - car_height/2:
                    self.pos[1] = screen_height - car_height/2

            elif self.mode == 'TR':
                self.angle -= 30
                self.pos[0] += math.cos(math.radians(180 - self.angle)) * self.move
                if self.pos[0] < car_width/2:
                    self.pos[0] = car_width/2
                elif self.pos[0] > screen_width - car_width/2:
                    self.pos[0] = screen_width - car_width/2

                self.distance += self.move
                self.time_spent += 1

                self.pos[1] += math.sin(math.radians(180 - self.angle)) * self.move
                if self.pos[1] < car_height/2:
                    self.pos[1] = car_height/2
                elif self.pos[1] > screen_height - car_height/2:
                    self.pos[1] = screen_height - car_height/2
            elif self.mode == 'TL':
                self.angle += 30
                self.pos[0] += math.cos(math.radians(180 - self.angle)) * self.move
                if self.pos[0] < car_width/2:
                    self.pos[0] = car_width/2
                elif self.pos[0] > screen_width - car_width/2:
                    self.pos[0] = screen_width - car_width/2

                self.distance += self.move
                self.time_spent += 1

                self.pos[1] += math.sin(math.radians(180 - self.angle)) * self.move
                if self.pos[1] < car_height/2:
                    self.pos[1] = car_height/2
                elif self.pos[1] > screen_height - car_height/2:
                    self.pos[1] = screen_height - car_height/2
            
            self.move = 0

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + car_width/2, int(self.pos[1]) + car_height/2]
        len = 10
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height + mode_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.car = Car('car.png', 'map.png', [960, 500])
        self.game_speed = 60
        self.mode = 0

    def action(self, action):
        if action == 0:
            self.car.mode = 'G'
            self.car.move = 9 # 1/2바퀴
        if action == 1:
            self.car.mode = 'DR'
            self.car.move = 9 # 1/2바퀴
        if action == 2:
            self.car.mode = 'DL'
            self.car.move = 9 # 1/2바퀴
        if action == 3:
            self.car.mode = 'TR'
            self.car.move = 9 # 1/2바퀴
        if action == 4:
            self.car.mode = 'TL'
            self.car.move = 9 # 1/2바퀴
        if action == 5:
            self.car.move = 9 # 1/2바퀴
        if action == 6:
            self.car.move = 18 # 1바퀴
        if action == 7:
            self.car.move = 36 # 2바퀴

        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(90, 300, 90):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        """
        if self.car.check_flag:
            self.car.check_flag = False
            reward = 2000 - self.car.time_spent
            self.car.time_spent = 0
        """
        if not self.car.is_alive:
            reward = -10000 + self.car.distance + 500 * self.car.current_check

        if self.car.goal:
            reward = 10000
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # return state
        radars = self.car.radars
        ret = [0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 7)

        if  self.car.mode == 'G':
            ret[-1] = 0
        if  self.car.mode == 'DR':
            ret[-1] = 1
        if  self.car.mode == 'DL':
            ret[-1] = 2
        if  self.car.mode == 'TR':
            ret[-1] = 3
        if  self.car.mode == 'TL':
            ret[-1] = 4


        return tuple(ret)

    def view(self):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.car.map, (0, 0))


        if self.mode == 1:
            self.screen.fill((0, 0, 0))

        self.car.radars_for_draw.clear()
        for d in range(90, 300, 90):
            self.car.check_radar_for_draw(d)

        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 30, 1)
        self.car.draw_collision(self.screen)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)
        
        # 모드 박스 그리기
        modes = ['G', 'DR', 'DL', 'TR', 'TL']
        colors_light = {
            'G': (180, 180, 180),
            'DR': (180, 150, 150),
            'DL': (150, 180, 150),
            'TR': (150, 150, 180),
            'TL': (180, 180, 150)
        }
        colors_dark = {
            'G': (100, 100, 100),
            'DR': (255, 0, 0),
            'DL': (0, 255, 0),
            'TR': (0, 0, 255),
            'TL': (255, 255, 0)
        }
        mode_name = {
            'G' : 'Go Straight',
            'DR' : 'Diagonal Right',
            'DL' : 'Diagonal Left',
            'TR' : 'Turn Right',
            'TL' : 'Turn Left'
        }

        font_regular = pygame.font.SysFont("Arial", 30)
        font_bold = pygame.font.SysFont("Arial", 30, bold=True)
        
        for i, mode in enumerate(modes):
            if mode == self.car.mode:
                color = colors_dark[mode]
                font = font_bold
            else:
                color = colors_light[mode]
                font = font_regular
            pygame.draw.rect(self.screen, color, pygame.Rect(i * 200, screen_height, 200, mode_height))
            # 모드 이름 표시
            text = font.render(mode_name[mode], True, (0, 0, 0))
            text_rect = text.get_rect(center=(i * 200 + 100, screen_height + mode_height//2))
            self.screen.blit(text, text_rect)

        # for i in range(5):
        #     pygame.draw.rect(self.screen, colors[i], pygame.Rect(i * 200, screen_height, 200, 120))


        pygame.display.flip()
        self.clock.tick(self.game_speed)
        
    
    def get_screen(self):
        # Capture the screen as an RGB array
        image_data = pygame.surfarray.array3d(self.screen)
        # Transpose the array to match (Height, Width, Channels)
        return np.transpose(image_data, (1, 0, 2))


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    try:
        rot_image = rot_image.subsurface(rot_rect).copy()
    except:
        pass
    return rot_image
