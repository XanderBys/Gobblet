#!/usr/bin/env python3.5
import multiprocessing
import sys
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from Environment import Environment
from Player import Player
from Model import Model
from State import State

def main(queue, player1, player2, env, rounds, output=False, display=False, prefix=""):
    players = [player1, player2]
    print("\nRunning {} episodes with batch size of {} . . .\n".format(rounds, player1.BATCH_SIZE))
    for i in range(rounds):
        while True:
            if display:
                env.display()
                
            game_over = False
            result = None
            
            for player in players:
                # this is a player's turn
                prev_state = State(np.array(env.state.board))
                
                # choose action
                action = player.choose_action(env.state)

                # take action
                next_state, result = env.update(action, player)
                
                # record the action and next_state
                player.samples.append([prev_state, action, next_state, int(result is not None)])
                
                if result is not None:
                    # the game is over here
                    game_over = True
                    break
                
            if game_over:
                player1.invalid_moves.append(player1.invalid_moves[-1])
                player2.invalid_moves.append(player2.invalid_moves[-1])
                # dispense rewards and update values
                if result == 0:
                    player1.reset(0.5)
                    player2.reset(0.5)
                    
                    player1.draw += 1
                    player2.draw += 1
                    
                elif result == 1 or result == -1:
                    # if result is 1, p1 wins
                    # if result is 1, p2 wins
                    
                    winner = player1 if result == 1 else player2
                    loser  = player1 if result == -1 else player2
                    
                    winner.reset(1)
                    loser.reset(0)
                    
                    winner.win += 1
                    loser.losses += 1
                
                else:
                    # this is where someone made an illegal move
                    # don't penalize or reward the other player here
                    loser = player1 if result == 10 else player2
                    
                    loser.reset(-5)
                    if (not loser.was_random):
                        loser.invalid_moves[-1] += 1
                    
                # adjust weights in the neural network
                player1.train()
                player2.train()
                
                env.reset()
                
                # send the learned values to the queue
                
                # update the players' learned values from things recieved on the queue
                
                game_over = False
                break
        
        if i % (rounds / 20) == 0 and output:
            print('{}% complete at {}. player1 W/L/D: {}/{}/{}'
                  .format(i/(rounds / 100), time.strftime("%H:%M:%S"), player1.win, player1.losses, player1.draw))
            # add the data to the queue
            player1.save_policy(prefix)
            player2.save_policy(prefix)
        
        if i % TAU == 0 and i > 0:
            for player in players:
                player.update_targets()
        
        # switch who starts the game
        players = players[::-1]
        
        # make sure players keep the same symbol
        # if it's an even round, p2 starts
        env.turn = -1 if i % 2 == 0 else 1
    
    fout_p1 = open("{}data_p1".format(prefix), 'wb')
    pickle.dump(player1.get_metrics(), fout_p1)
    
    fout_p2 = open("{}data_p2".format(prefix), 'wb')
    pickle.dump(player2.get_metrics(), fout_p2)
    
    plt.subplot(2, 2, 1)
    plt.plot(range(len(player1.total_rewards)), [i for i in player1.total_rewards])
    plt.ylabel("Total rewards p1")
    plt.subplot(2, 2, 2)
    plt.plot(range(len(player1.regret)), [i for i in player1.regret])
    plt.ylabel("Regret p1")
    plt.xlabel("Rounds of training")
    plt.subplot(2, 2, 3)
    plt.plot(range(len(player2.total_rewards)), [i for i in player2.total_rewards])
    plt.ylabel("Total rewards")
    plt.subplot(2, 2, 4)
    plt.plot(range(len(player2.regret)), [i for i in player2.regret])
    plt.ylabel("Regret p2")
    plt.xlabel("Rounds of training")
    plt.show()
    
if __name__ == '__main__':
    num_args = len(sys.argv)

    NUM_ROUNDS = int(sys.argv[1].split("*")[0]) if num_args >= 2 else 50000
    BATCH_SIZE = int(sys.argv[1].split("*")[1]) if num_args >= 2 else 100
    LAMBDA = 4/NUM_ROUNDS # ensures that epsilon has the desired decay rate regardless of NUM_ROUNDS
    TAU = NUM_ROUNDS // 10
    
    p1_dueling = True if num_args < 3 else sys.argv[2][0].lower() == 't'
    p2_dueling = True if num_args < 4 else sys.argv[3][0].lower() == 't'
    
    p1_PER = True if num_args < 5 else sys.argv[4][0].lower() == 't'
    p2_PER = True if num_args < 6 else sys.argv[5][0].lower() == 't'
    
    file_prefix = "{}_dueling_{}_PER_".format(p1_dueling + p2_dueling, p1_PER + p2_PER)
    
    environment = Environment(4, 4, 4)
    #(self, name, env, symbol, memory_capacity, model=None, BATCH_SIZE=0, EPSILON_ARGS=(0,0,0), maximize_entropy=False, use_PER=False, PER_hyperparams=(0,0,0)):
    p1 = Player('p1', environment, 1, NUM_ROUNDS//4, model=Model(16, [16, 12], dueling=p1_dueling)
                , BATCH_SIZE=BATCH_SIZE, EPSILON_ARGS=(0.9, 0.1, LAMBDA),
                use_PER=p1_PER, PER_hyperparams=(0.01, 0.6, 0.4))
    p2 = Player('p2', environment, -1, NUM_ROUNDS//4, model=Model(16, [16, 12], dueling=p2_dueling)
                , BATCH_SIZE=BATCH_SIZE, EPSILON_ARGS=(0.9, 0.1, LAMBDA),
                use_PER=p2_PER, PER_hyperparams=(0.01, 0.6, 0.4))
    
    print("\nInitialized players with p1 {}dueling, p2 {}dueling \np1 {}using PER, and p2 {}using PER"
          .format("" if p1_dueling else "not ", "" if p2_dueling else "not ", "" if p1_PER else "not ", "" if p2_PER else "not "))
    print("\nOutput prefix will be {}\n".format(file_prefix))
    
    main(None, p1, p2, environment, NUM_ROUNDS, True, False, file_prefix)