import time
import math
import pygame
from .movement import Car
from model import *


pygame.init()

screen = pygame.display.set_mode((1380, 820))
pygame.display.set_caption("Need for Speed")
clock = pygame.time.Clock()


def run():
    running = True
    car = Car(200, 100)
    track_image = pygame.image.load("game/maps/map1.png").convert()
    wall_mask = pygame.mask.from_threshold(track_image, (255, 127, 39), (5, 5, 5, 255))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            car.turn_left()
        if keys[pygame.K_d]:
            car.turn_right()
        if keys[pygame.K_w]:
            car.forward()
        elif keys[pygame.K_s]:
            car.backward()
        else:
            car.stop()
        car.move()

        screen.blit(track_image, (0, 0))

        if check_collision(wall_mask, car):
            break

        checkpoint(screen, car)

        car.score -= 0.1

        get_distance(car, wall_mask, 0)
        get_distance(car, wall_mask, 35)
        get_distance(car, wall_mask, -35)
        get_distance(car, wall_mask, 75)
        get_distance(car, wall_mask, -75)

        car.display(screen)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_train(model, optimizer, replay_buffer, car_count):


    def choose_action(state, network, epsilon=0.3, output_dim=7):
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = network(state_tensor)
            return torch.argmax(q_values).item()


    running = True
    cars = [Car(200, 100, random_color()) for _ in range(car_count)]
    track_image = pygame.image.load("game/maps/map1.png").convert()
    wall_mask = pygame.mask.from_threshold(track_image, (255, 127, 39), (5, 5, 5, 255))

    act = 5
    action = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(track_image, (0, 0))

        for car in cars:

            if len(replay_buffer.buffer) >= 64:
                train_step(replay_buffer, 64, model, optimizer)

            if act == 5:
                state = [
                    get_distance(car, wall_mask, 0),
                    get_distance(car, wall_mask, 35),
                    get_distance(car, wall_mask, -35),
                    get_distance(car, wall_mask, 75),
                    get_distance(car, wall_mask, -75),
                    car.speed,
                    car.steering
                ]

                action = choose_action(state, model)
                act -=1
            elif act == 0:
                act = 5
            else:
                act -= 1

            if action == 1:
                car.turn_left()
            elif action == 2:
                car.turn_right()
            elif action == 3:
                car.forward()
            elif action == 4:
                car.backward()
            elif action == 5:
                car.forward()
                car.turn_left()
            elif action == 6:
                car.forward()
                car.turn_right()
            else:
                car.stop()
            car.move()


            if check_collision(wall_mask, car):
                state = [
                    get_distance(car, wall_mask, 0),
                    get_distance(car, wall_mask, 35),
                    get_distance(car, wall_mask, -35),
                    get_distance(car, wall_mask, 75),
                    get_distance(car, wall_mask, -75),
                    car.speed,
                    car.steering
                ]

                reward = car.score

                replay_buffer.push(state, action, reward, state, False)
                print(car.score)
                car.reset(200, 100)

            checkpoint(screen, car)

            car.score -= 0.1
            car.score += car.speed
            if car.speed == 0:
                car.score -= 10

            car.display(screen)


        pygame.display.flip()

        try:
            clock.tick(30)
        except KeyboardInterrupt:
            break

    pygame.quit()


def check_collision(wall_mask, car):
    rotated_image = pygame.transform.rotate(car.image, car.angle)
    car_mask = pygame.mask.from_surface(rotated_image)

    rect = rotated_image.get_rect(center=(car.x, car.y))
    offset = (rect.left - 0, rect.top - 0)
    collision_point = wall_mask.overlap(car_mask, offset)

    if collision_point is not None:
        return True
    return False


def create_checkpoint(scrn):
    checkpoints = [
        pygame.Surface((10, 230), pygame.SRCALPHA),
        pygame.Surface((10, 230), pygame.SRCALPHA),
        pygame.Surface((10, 230), pygame.SRCALPHA),
        pygame.Surface((200, 10), pygame.SRCALPHA),
        pygame.Surface((10, 170), pygame.SRCALPHA),
        pygame.Surface((10, 130), pygame.SRCALPHA),
        pygame.Surface((200, 10), pygame.SRCALPHA),
    ]

    for cp in checkpoints:
        cp.fill((80, 80, 80))

    checkpoints_placement = [
        scrn.blit(checkpoints[0], (300, 20)),
        scrn.blit(checkpoints[1], (600, 20)),
        scrn.blit(checkpoints[2], (1000, 20)),
        scrn.blit(checkpoints[3], (1180, 300)),
        scrn.blit(checkpoints[4], (600, 300)),
        scrn.blit(checkpoints[5], (1000, 520)),
        scrn.blit(checkpoints[6], (30, 500))
    ]

    return checkpoints, checkpoints_placement


def checkpoint(scrn, car):
    checkpoints, checkpoints_placement = create_checkpoint(scrn)
    checkpoint_mask = pygame.mask.from_surface(checkpoints[car.next_checkpoint])
    rotated_image = pygame.transform.rotate(car.image, car.angle)
    car_mask = pygame.mask.from_surface(rotated_image)

    rect = rotated_image.get_rect(center=(car.x, car.y))
    cp_offset = (rect.left - checkpoints_placement[car.next_checkpoint].left, rect.top - checkpoints_placement[car.next_checkpoint].top)
    checkpoint_touch = checkpoint_mask.overlap(car_mask, cp_offset)

    if checkpoint_touch is not None:
        car.next_checkpoint += 1 if car.next_checkpoint < len(checkpoints)-1 else -len(checkpoints)+1
        car.score = round(car.score + 20 + car.speed * 0.01, 2)


def get_distance(car, wall_mask, angle_offset=0, max_distance=200):
    angle_rad = math.radians(car.angle + angle_offset)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)

    for distance in range(0, max_distance, 4):
        x = int(car.x + dx * distance)
        y = int(car.y + dy * distance)

        if wall_mask.get_at((x, y)):
            #dist = pygame.Surface((distance, 2), pygame.SRCALPHA)
            #dist.fill((50, 50, 50))
            #rotated = pygame.transform.rotate(dist, car.angle + angle_offset)
            #rect = rotated.get_rect(center=(int(car.x + dx * distance/2), int(car.y + dy * distance/2)))
            #screen.blit(rotated, rect)
            return distance

    return max_distance


def random_color():
    return (random.randint(0, 225), random.randint(0, 225), random.randint(0, 225))


if __name__ == "__main__":
    run()
