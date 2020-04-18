import time
import multiprocessing
from copy import deepcopy
from Environment import Environment
from State import State
from Player import Player

def main(queue, player1, player2, env, rounds, output=False, display=False):
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
                start_time = time.time()
                action = player.choose_action(env.state, env.turn)
                times.append(time.time() - start_time)
                if action['size'] == -1:
                    # ERROR CASE
                    for state in player.states:
                        print(state)
                 
                # take action
                next_state, result = env.update(action, player)
                
                # record the action and next_state
                player.states.append(deepcopy(next_state))
                
                if i >= rounds-1:
                    env.display()
                
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
    NUM_PROCESSES = 10 # the number of processes to be run
    NUM_ROUNDS = 5000 # the number of rounds to be run on each process
    DECAY_RATE = 0.9998
    
    # run training on each of the pi's four processors
    processes = []
    
    q = multiprocessing.Queue()
    for idx in range(NUM_PROCESSES):
        # setup and start all the processes
        environment = Environment(4, 4, 4)
        p1 = Player('p1', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
        p2 = Player('p2', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
        processes.append(multiprocessing.Process(target=main, args=(q, p1, p2, environment, NUM_ROUNDS, (idx==0),)))
    
    print("Training started at {}".format(time.strftime("%H:%M:%S")))
    for process in processes:
        process.start()
    
    policy_p1 = {}
    policy_p2 = {}

    # gather the data
    while True:
        data = q.get()
        policy_p1.update(data[0].states_values)
        policy_p2.update(data[1].states_values)
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