import copy
import Environment
import Player
import State

env = Environment.Environment(4, 4, 4)
player = Player.Player('p1', env, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], 0.9, False)
human = Player.Player('human', env, [[{'location':j, 'size':4-j} for j in range(4)] for i in range(3)], 0, False)
player.load_policy('policy_p2')
while True:
    while True:
        row = input("Type a row to move to: ")
        col = input("Type a col to move to: ")
        size = input("Type a size of piece to use: ")
        
        next_state, result = env.update({'destination':(int(row), int(col)), 'size':int(size), 'origin':[0]}, human)
        env.state.board = next_state.board
        
        env.display()
        
        if result != None:
            # the game is over here
            game_over = True
            print("RESULT: {}".format(result))
            break

        game_over = False
        board = copy.deepcopy(env.state)
        
        # choose action
        #negative_board = State.State(list(map(lambda x:list(map(lambda y: -1 * y, x)), board.board)))
        action = player.choose_action(board, env.turn)
        
        # take action
        next_state, result = env.update(action, player)
        
        # record the action and next_state
        env.state = next_state
        
        env.display()
        if result != None:
            # the game is over here
            game_over = True
            print("RESULT: {}".format(result))
            break

    cont = input("Continue(y/n)? ")
    if cont.lower() != 'y':
        break
    
    env.reset()
    player.reset()
    human.reset()
