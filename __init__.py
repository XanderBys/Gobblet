import Player
import Environment
import State
import time
import train
import play_vs_human

if __name__ == '__main__':
    NUM_ROUNDS = 500
    DECAY_RATE = 0#.99985

    environment = Environment.Environment(4, 4, 4)
    p1 = Player.Player('p1', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
    p2 = Player.Player('p2', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
    
    print('Program started at {}'.format(time.strftime("%H:%M:%S")))
    train.main(p1, p2, environment, NUM_ROUNDS)
    print('{} rounds of training finished at {}'.format(NUM_ROUNDS, time.strftime("%H:%M:%S")))
    
    p1.save_policy(time.strftime("%H:%M:%S"))
    p2.save_policy(time.strftime("%H:%M:%S"))