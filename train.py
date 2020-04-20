import time
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt
from Environment import Environment
from State import State
from Player import Player

def main(num_processes, num_rounds, decay_rate):
    processes = []
    
    q = multiprocessing.Queue()
    for idx in range(num_processes):
        # setup and start all the processes
        environment = Environment(4, 4, 4)
        p1 = Player('p1', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], decay_rate)
        p2 = Player('p2', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], decay_rate)
        processes.append(multiprocessing.Process(target=run_training, args=(q, p1, p2, environment, num_rounds, (idx==0),)))
    
    print("Training started at {}".format(time.strftime("%H:%M:%S")))
    for process in processes:
        process.start()
    
    policy_p1 = {}
    policy_p2 = {}

    # gather the data
    players = []
    while True:
        data = q.get()
        policy_p1.update(data[0].states_values)
        policy_p2.update(data[1].states_values)
        players.extend(data)
        if q.empty():
            break
        
    print("Data successfully gathered")
    
    policies = [policy_p1, policy_p2]
    
    for idx in range(2):
        fout = open("policy_p{}".format(idx+1), 'w')
        for state, value in policies[idx].items():
            fout.write("{};{}\n".format(state, value))
            
        fout.close()
    
    print("Data loading complete")
    
    for i in processes:
        if i.is_alive():
            i.terminate()
            
    print("Cleanup complete")
    
    print("Run complete at {}".format(time.strftime("%H:%M:%S")))
    
    print("Visualizing data . . .")
    plt.plot(list(range(NUM_ROUNDS)), players[0].total_reward_values)
    plt.show()

def display_data(data):
    pass

def run_training(queue, player1, player2, env, rounds, output=False, display=False):
    players = [player1, player2]
    times = []
    l = 0
    for i in range(rounds):
        while True:
            if display:
                env.display()

            game_over = False
            result = None
            
            for player in players:
                # this is a player's turn
                
                # choose action
                action = player.choose_action(env.state, env.turn)
                if action['size'] == -1:
                    # ERROR CASE
                    for state in player.states:
                        print(state)
                 
                # take action
                next_state, result = env.update(action, player)
                
                # record the action and next_state
                player.states.append(deepcopy(next_state))
                
                if len(env.get_legal_moves(player))==0:
                    result = 0
                
                if result != None:
                    # the game is over here
                    l = len(player.states)
                    game_over = True
                    break

            if game_over:
                # dispense rewards and update values
                if result == 0:
                    player1.update_values(0.1)
                    player2.update_values(0.5)
                    
                    player1.draw += 1
                    player2.draw += 1
                    
                else:
                    # if p1 went first and result is 1, p1 wins
                    # if p2 went first and result is 1, p2 wins
                    winner = player1 if result == 1 else player2
                    loser  = player1 if result == -1 else player2
                    
                    winner.update_values(1)
                    loser.update_values(0)
                    
                    winner.win += 1
                    loser.loss += 1
                
                # reset everything for the next game
                player1.reset()
                player2.reset()
                env.reset()
            
                game_over = False
                break
        
        if i % (rounds / 20) == 0 and output:
            print('{}% complete at {}. player1 W/L/D: {}/{}/{}'.format(i/(rounds / 100), time.strftime("%H:%M:%S"), player1.win, player1.loss, player1.draw))
        
        # switch who starts the game
        players = players[::-1]
        # make sure players keep the same symbol
        # if it's an even round, p2 starts
        env.turn = -1 if i % 2 == 0 else 1
    
    # enqueue results
    queue.put(players)

if __name__ == '__main__':
    #1: 30 min  4: 
    NUM_PROCESSES = 4 # the number of processes to be run
    NUM_ROUNDS = 2000 # the number of rounds to be run on each process
    DECAY_RATE = 0.99986
    
    # run training on multiple processors simultaneously
    main(NUM_PROCESSES, NUM_ROUNDS, DECAY_RATE)