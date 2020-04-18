import time
from copy import deepcopy
from Environment import Environment
from State import State
from Player import Player

def main(player1, player2, env, rounds, display=False):
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
        
        if i % (rounds / 20) == 0:
            print('{}% complete at {}. player1 W/L/D: {}/{}/{}'.format(i/(rounds / 100), time.strftime("%H:%M:%S"), player1.win, player1.loss, player1.draw))
            print('Average move time | Length of game: {} | {}'.format(sum(times)/len(times), l))
        # switch who starts the game
        players = players[::-1]
        # make sure players keep the same symbol
        # if it's an even round, p2 starts
        env.turn = -1 if i % 2 == 0 else 1


if __name__ == '__main__':
    NUM_ROUNDS = 50
    DECAY_RATE = 1#.99985

    environment = Environment(4, 4, 4)
    p1 = Player('p1', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
    p2 = Player('p2', environment, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], DECAY_RATE)
    
    print('Program started at {}'.format(time.strftime("%H:%M:%S")))
    main(p1, p2, environment, NUM_ROUNDS)
    print('{} rounds of training finished at {}'.format(NUM_ROUNDS, time.strftime("%H:%M:%S")))
    
    p1.save_policy(time.strftime("%H:%M:%S"))
    p2.save_policy(time.strftime("%H:%M:%S"))