# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPosition(1)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #oldGhostStates = currentGameState.getGhostStates()
        #oldScaredTimes = [ghostState.scaredTimer for ghostState in oldGhostStates]

        "*** YOUR CODE HERE ***"
        oldFood = currentGameState.getFood()
        oldFoodList = oldFood.asList()
        #scaredDifference = newScaredTimes - oldScaredTimes
        score = 0
        closestDist = None
        closestDistIndex = None
        for i in range(len(successorGameState.data.agentStates[1:])):
            ghostPosition = successorGameState.getGhostPosition(i+1)
            posDif = abs(newPos[0] - ghostPosition[0]) + abs(newPos[1] - ghostPosition[1])
            if closestDist is None or posDif < closestDist:
                closestDist = posDif
                closestDistIndex = i
        if closestDist is not None:
            if newScaredTimes[closestDistIndex] > 5:
                if closestDist > 0:
                    score += 1/closestDist
                else:
                    score += 1

            else:
                if closestDist > 0:
                    score -= 1/closestDist
                else:
                    score -= 1
        foodDiff = len(oldFoodList) - len(newFoodList)
        score += foodDiff
        if foodDiff == 0:
            closestFoodDist = None
            for i in range(len(newFoodList)):
                posDif = abs(newPos[0] - newFoodList[i][0]) + abs(newPos[1] - newFoodList[i][1])
                if closestFoodDist is None or posDif < closestFoodDist:
                    closestFoodDist = posDif
            score += 1/(closestFoodDist *2)
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        toReturn = self.minimaxFunction(gameState, 0)
        return toReturn[1]


    def minimaxFunction (self, gameState, d):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif d > (self.depth*numAgents - 1):
            return self.evaluationFunction(gameState)
        else:
            if d%numAgents == 0:
                #pacman's turn, max
                max = None
                maxAction = None
                for action in gameState.getLegalActions(0):
                    current = self.minimaxFunction(gameState.generateSuccessor(0, action), d+1)
                    if max is None or current > max:
                        max = current
                        maxAction = action
                if d==0 and maxAction is not None:
                    return [max, maxAction]
                return max
            else:
                # ghost's turn, min
                min = None
                for action in gameState.getLegalActions(d%numAgents):
                    current = self.minimaxFunction(gameState.generateSuccessor(d%numAgents, action), d+1)
                    if min is None or current < min:
                        min = current
                return min


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        toReturn = self.alphaBetaPruning(gameState, 0, - math.inf, math.inf)
        return toReturn[1]

    def alphaBetaPruning (self, gameState, d, alpha, beta):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif d > (self.depth*numAgents - 1):
            return self.evaluationFunction(gameState)
        else:
            if d%numAgents == 0:
                #pacman's turn, max
                max = None
                maxAction = None
                for action in gameState.getLegalActions(0):
                    current = self.alphaBetaPruning(gameState.generateSuccessor(0, action), d + 1, alpha, beta)
                    if max is None or current > max:
                        max = current
                        maxAction = action
                    if max > beta:
                        if d == 0:
                            return [max, maxAction]
                        return max
                    if alpha < max:
                        alpha = max
                if d == 0 and maxAction is not None:
                    return [max, maxAction]
                return max
            else:
                # ghost's turn, min
                min = None
                for action in gameState.getLegalActions(d % numAgents):
                    current = self.alphaBetaPruning(gameState.generateSuccessor(d % numAgents, action), d + 1, alpha, beta)
                    if min is None or current < min:
                        min = current
                    if min < alpha:
                        return min
                    if beta > min:
                        beta = min
                return min

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        toReturn = self.expectimaxFunction(gameState, 0)
        return toReturn[1]
        util.raiseNotDefined()

    def expectimaxFunction (self, gameState, d):
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif d > (self.depth*numAgents - 1):
            return self.evaluationFunction(gameState)
        else:
            if d%numAgents == 0:
                #pacman's turn, max
                max = None
                maxAction = None
                for action in gameState.getLegalActions(0):
                    current = self.expectimaxFunction(gameState.generateSuccessor(0, action), d+1)
                    if max is None or current > max:
                        max = current
                        maxAction = action
                if d==0 and maxAction is not None:
                    return [max, maxAction]
                return max
            else:
                # ghost's turn, rand
                numLegalActions = len(gameState.getLegalActions(d%numAgents))
                totalUtility = 0
                for action in gameState.getLegalActions(d%numAgents):
                    current = self.expectimaxFunction(gameState.generateSuccessor(d%numAgents, action), d+1)
                    totalUtility += current
                toReturn = totalUtility/numLegalActions
                return toReturn

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    score = 0
    closestDist = None
    closestDistIndex = None
    for i in range(len(currentGameState.data.agentStates[1:])):
        ghostPosition = currentGameState.getGhostPosition(i + 1)
        posDif = abs(position[0] - ghostPosition[0]) + abs(position[1] - ghostPosition[1])
        if closestDist is None or posDif < closestDist:
            closestDist = posDif
            closestDistIndex = i
    if closestDist is not None:
        if scaredTimes[closestDistIndex] > 4:
            if closestDist > 0:
                score += 5 / closestDist
        else:
            if closestDist > 0:
                if closestDist >= 7:
                   score -= 1 / 14
                else:
                    score -= 1 / (closestDist*2)
            else:
                score -= 1
    if len(foodList) > 0:
        foodNumber = 1/len(foodList)
        score += foodNumber*8
        closestFoodDist = None
        secondFoodDist = None
        for i in range(len(foodList)):
            posDif = abs(position[0] - foodList[i][0]) + abs(position[1] - foodList[i][1])
            if closestFoodDist is None:
                closestFoodDist = posDif
            elif secondFoodDist is None:
                if posDif < closestFoodDist:
                    secondFoodDist = closestFoodDist
                    closestFoodDist = posDif
                else:
                    secondFoodDist = posDif
            elif posDif < secondFoodDist:
                if posDif < closestFoodDist:
                    secondFoodDist = closestFoodDist
                    closestFoodDist = posDif
                else:
                    secondFoodDist = posDif
        if secondFoodDist is None or closestFoodDist is None or secondFoodDist-closestFoodDist <= 3:
                score += 1 / (closestFoodDist * 15)
    else:
        score += 10
    return score

# Abbreviation
better = betterEvaluationFunction