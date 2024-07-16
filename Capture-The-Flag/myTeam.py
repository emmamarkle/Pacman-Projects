# myTeam.py
# ---------
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
import distanceCalculator
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game, capture
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState: capture.GameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState: capture.GameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.weights = util.Counter()
    self.discount = 0.5
    self.alpha = 0.5
    self.consideredStates = []

  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState: capture.GameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState: capture.GameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState: capture.GameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return self.weights

  def updateWeights(self, state, action, nextState, reward: float):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    gameState = capture.GameState
    if state is not None:
      actionMax = None
      for nextAction in gameState.getLegalActions(nextState):
        current = self.getQValue(nextState, nextAction)
        if actionMax is None or current > actionMax:
          actionMax = current
      if actionMax is None:
        actionMax = 0
      difference = (reward + self.discount * actionMax) - self.getQValue(state, action)
      for feature in self.getFeatures(state, action):
        featureVal = self.getFeatures(state, action)[feature]
        self.weights[feature] += self.alpha * difference * featureVal
    if len(gameState.getLegalActions(nextState)) == 0:
      action = None
    else:
      action = self.getPolicy(nextState)
    self.consideredStates.append(state)
    return action

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    gameState = capture.GameState
    toReturn = 0
    if state == 'TERMINAL_STATE':
      return 0
    else:
      for feature in self.getFeatures(state, action):
        featureVal = self.getFeatures(state, action)[feature]
        toReturn += self.weights[feature] * featureVal
      return toReturn

  def getPolicy(self, state):
    gameState = capture.GameState
    return self.computeActionFromQValues(state)

  def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        gameState = capture.GameState
        if len(gameState.getLegalActions(self.index)) == 0:
            return None
        max = None
        maxAction = None
        for action in gameState.getLegalActions(self.index):
            current = self.getQValue(state, action)
            if max is None or current > max:
                max = current
                maxAction = action
        return maxAction


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def getFeatures(self, gameState: capture.GameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        capList = self.getCapsules(successor)
        features['successorScore'] = self.getScore(successor)
        ownPos = successor.getAgentPosition(self.index)

        if self.index in gameState.getRedTeamIndices():
            isRed = True
        else:
            isRed = False
        side = gameState.data.layout.width//2
        if isRed:
            ownEdge = side
        else:
            ownEdge = side + 1
        if successor.getAgentState(self.index).isPacman:
            distToEdge = abs(ownPos[0]-ownEdge)
        else:
            distToEdge = 0
        #print(distToEdge)
        features['distToEdge'] = 30

        distToEdgeGhost = abs(successor.getAgentPosition(self.index)[0] - ownEdge)
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(ownPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
            if gameState.getAgentState(self.index).numCarrying < successor.getAgentState(self.index).numCarrying:
                features['distanceToFood'] = 0
        if len(capList)>0:
            minCapDistance = min([self.getMazeDistance(ownPos, cap) for cap in capList])
            features['distanceToCap'] = minCapDistance

        features['numCarrying'] = successor.getAgentState(self.index).numCarrying
        if successor.getAgentState(self.index).numCarrying >= 2:
            #features['numCarrying'] = successor.getAgentState(self.index).numCarrying*0.7
            features['distToEdge'] = distToEdge

        if (distToEdge == 0 or distToEdge == 1 or distToEdge == 2) and gameState.getAgentState(self.index).numCarrying > 0:
            features['distToEdge'] = -5

        # if successor.getAgentState(self.index).isPacman:

        opponents = self.getOpponents(successor)
        minDist = 9999
        features['distanceToOpp'] = 0
        numPacOpps = 0
        for opponent in opponents:
            '''if gameState.getAgentPosition(self.index) == gameState.getAgentPosition(opponent):
                features['numCarrying'] = -1000
                self.consideredPos = []'''
            dist = self.getMazeDistance(ownPos, successor.getAgentPosition(opponent))
            if not successor.getAgentState(opponent).isPacman and successor.getAgentState(opponent).scaredTimer <= 5:
                numPacOpps += 1
                if dist < minDist:
                    minDist = dist
                    features['distanceToOpp'] = minDist
            if not successor.getAgentState(opponent).isPacman and successor.getAgentState(opponent).scaredTimer > 5:
                if dist < minDist:
                    minDist = dist
                    features['distanceToOpp'] = -minDist
            if distToEdgeGhost <= 2 and not successor.getAgentState(self.index).isPacman and not successor.getAgentState(opponent).isPacman and self.getMazeDistance(ownPos,successor.getAgentPosition(opponent))<=7:
                diff = abs(successor.getAgentPosition(self.index)[1]-successor.getAgentPosition(opponent)[1])
                if diff!=0:
                    features['yDist'] = 25*(1/diff)
                else:
                    features['yDist'] = 25
            else:
                features['yDist'] = 25

        if successor.getAgentState(self.index).isPacman and not gameState.getAgentState(self.index).isPacman and minDist<=4:
            features['distanceToOpp'] = minDist-500

        if minDist < 5:
            features['numCarrying'] *= -1
            #features['numCarrying'] += 1
            features['distToEdge'] = distToEdge
        if features['distanceToOpp'] == 1 or features['distanceToOpp'] == 2 or features['distanceToOpp'] == 0:
            features['numCarrying'] = -100

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState: capture.GameState, action):
        return {'successorScore': 1000, 'distanceToFood': -1, 'numCarrying': 2, 'distanceToOpp': 0.8, 'distToEdge': -2, 'yDist':-10, 'stop':-10, 'reverse':-2}
    


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState: capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    ownCapsuleList = self.getCapsulesYouAreDefending(successor)
    opponents = self.getOpponents(gameState)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFood(successor).asList()
    ownFoodList = self.getFoodYouAreDefending(successor).asList()
    if self.index in gameState.getRedTeamIndices():
            isRed = True
    else:
            isRed = False

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    if self.getScore(successor) > 6:
      features['onDefense'] *= -1
    elif self.getScore(successor) < -2:
      features['onDefense'] *= 100

    features['numOwnFoodRemaining'] = len(ownFoodList) - 2
    features['numFoodRemaining'] = len(foodList) - 2

    maxCarry = -9999
    for opponent in opponents:
      carrying = gameState.getAgentState(opponent).numCarrying
      if carrying > maxCarry:
        maxCarry = carrying
        features['numOppCarrying'] = carrying

    if successor.getAgentState(self.index).scaredTimer > 5 and features['invaderDistance'] < 10:
       features['invaderDistance'] *= -1

    side = gameState.data.layout.width//2
    if isRed:
        ownEdge = side
    else:
        ownEdge = side + 1
    if not successor.getAgentState(self.index).isPacman:
        distToEdge = abs(successor.getAgentPosition(self.index)[0]-ownEdge)
    else:
        distToEdge = 0
    features['distToEdge'] = distToEdge  

    opponent1 = self.getOpponents(gameState)[0]
    opponent2 = self.getOpponents(gameState)[1]

    if not successor.getAgentState(self.index).isPacman and not successor.getAgentState(opponent).isPacman:
      if features['distToEdge'] < 5: # following wrong agent
        features['yDist'] = abs(successor.getAgentPosition(self.index)[1] - ((successor.getAgentPosition(opponent1)[1] + (successor.getAgentPosition(opponent2)[1]))/2))
    elif not isRed and successor.getAgentPosition(opponent)[0] in range(0, side):
        features['yDist'] = 0
    elif isRed and successor.getAgentPosition(opponent)[0] in range(side, gameState.data.layout.width):
        features['yDist'] = 0

    if len(ownFoodList) > 0:  # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, food) for food in ownFoodList])
        features['distanceToOwnFood'] = minDistance

        if len(ownCapsuleList) > 0:
            minCapDistance = min([self.getMazeDistance(myPos, capsule) for capsule in ownCapsuleList])
            features['distanceToOwnCap'] = minCapDistance

    return features

  def getWeights(self, gameState: capture.GameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2, 'distToEdge': -2, 'numOppCarrying': -100, 'yDist' : -0.8}