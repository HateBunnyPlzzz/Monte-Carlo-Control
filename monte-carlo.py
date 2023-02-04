import numpy as np
import gym 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    EPS = 0.01
    GAMMA = 1.0

    Q={}
    agentSumSpace = [i for i in range(4,22)]
    dealerShowCardSpace = [i+1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0,1] # stick or hit

    stateSpace = []
    returns = {}
    pairsVisited = {}
    # iterate over all the elements of the state-space and create tuples out of those, 
    # set the agents estimate for the state-action combinations to 0
    # used to keep track of how many times we visited the particular point in state-action space  
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    returns[((total, card, ace), action)] = 0
                    pairsVisited[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))
    
    policy = {}
    for state in stateSpace:
        # starting with a complete random Equiprobable policy
        policy[state] = np.random.choice(actionSpace)
    
    numEpisodes = 1000000
    for i in range(numEpisodes):
        statesActionsReturns = []
        memory = []
        if i % 100000 == 0:
            print('starting episode', i)
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            # observation vector is the playerSum, dealer's showing card, and whether player has usable Ace or not
            memory.append((observation[0], observation[1], observation[2], action, reward))
            # reset the obsv to new state.
            observation = observation_ 
        memory.append((observation[0], observation[1], observation[2], action, reward))

        #Iterate over agent's reverse memory and calculate the returns from that memory.
        G = 0
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            if last:
                last = True
            # this will tell you the returns that followed the 
            # agent's first visit to particular point in state-action space
            else:
                statesActionsReturns.append((playerSum, dealerCard, usableAce, action, G))
            G = GAMMA*G + reward
        
        statesActionsReturns.reverse()
        statesActionsVisited = []

        for playerSum, dealerCard, usableAce, action, G in statesActionsReturns:
            sa = ((playerSum, dealerCard, usableAce), action)
            if sa not in statesActionsVisited:
                pairsVisited[sa] += 1
                # incremental implementation
                # new estimate = 1 / N * [sample - old estimate]
                returns[(sa)] += (1 / pairsVisited[(sa)])*(G - returns[(sa)])
                Q[sa] = returns[sa]
                rand = np.random.random()
                if rand < 1 - EPS:
                    state = (playerSum, dealerCard, usableAce)
                    values = np.array([Q[(state, a)] for a in actionSpace])
                    best = np.random.choice(np.where(values==values.max())[0])
                    policy[state] = actionSpace[best]
                else:
                    policy[state] = np.random.choice(actionSpace)
                statesActionsVisited.append(sa)
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0
    
    numEpisodes = 1000
    rewards = np.zeros(numEpisodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test policy')
    for i in range(numEpisodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
    
    wins /= numEpisodes
    losses /= numEpisodes
    draws /= numEpisodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    plt.plot(rewards)
    plt.show()