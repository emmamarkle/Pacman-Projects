
import distanceCalculator
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game, capture
from util import nearestPoint


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
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


class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState: capture.GameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        self.discount = 0.8
        self.alpha = 0.2
        self.consideredStates = []
        self.lastAction = None

    def chooseAction(self, gameState: capture.GameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)

        prev = CaptureAgent.getPreviousObservation(self)
        curr = CaptureAgent.getCurrentObservation(self)
        if prev != None:
            # action = CaptureAgent.getAction(self, prev)
            reward = curr.data.score - prev.data.score
            self.updateWeights(prev, self.lastAction, curr, reward)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        #print("max",maxValue)
        #print(values)
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
            self.lastAction = bestAction
            return bestAction
        #print(bestActions)
        self.lastAction = random.choice(bestActions)
        return self.lastAction

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
        if state is not None:
            actionMax = None
            for nextAction in nextState.getLegalActions(self.index):
                current = self.getQValue(nextState, nextAction)
                if actionMax is None or current > actionMax:
                    actionMax = current
            if actionMax is None:
                actionMax = 0
            difference = (reward + self.discount * actionMax) - self.getQValue(state, action)
            for feature in self.getFeatures(state, action):
                featureVal = self.getFeatures(state, action)[feature]
                self.weights[feature] += (self.alpha * difference * featureVal)
                #print("here", self.alpha, difference, featureVal)
        if len(nextState.getLegalActions(self.index)) == 0:
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
        toReturn = 0
        if state.isOver():
            return 0
        else:
            for feature in self.getFeatures(state, action):
                featureVal = self.getFeatures(state, action)[feature]
                toReturn += self.weights[feature] * featureVal
            return toReturn

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(state.getLegalActions(self.index)) == 0:
            return None
        max = None
        maxAction = None
        for action in state.getLegalActions(self.index):
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
    def registerInitialState(self, gameState: capture.GameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        file1 = open("OffensiveWeights","r+")
        featureList = ['successorScore', 'distanceToFood', 'stop', 'reverse']
        for featureInd in range(len(featureList)):
            featureNow = featureList[featureInd]
            self.weights[featureNow] = float(file1.readline().rstrip('\n'))
        file1.close()
        self.discount = 0.8
        self.alpha = 0.2
        self.consideredStates = []
        self.lastAction = None

    def getFeatures(self, gameState: capture.GameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        print(foodList)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0: # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
    def registerInitialState(self, gameState: capture.GameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.weights = util.Counter()
        file1 = open("DefensiveWeights", "r+")
        featureList = ['numInvaders', 'onDefense', 'invaderDistance', 'stop', 'reverse', 'successorScore']
        for featureInd in range(len(featureList)):
            featureNow = featureList[featureInd]
            self.weights[featureNow] = float(file1.readline().rstrip('\n'))
        file1.close()
        print(self.weights)
        self.discount = 0.8
        self.alpha = 0.2
        self.consideredStates = []
        self.lastAction = None

    def getFeatures(self, gameState: capture.GameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) != 0: 
            features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if sum(features.values()) != 0:
            factor = 1.0 / sum(features.values())
        else:
            factor = 1
        normalized_features = util.Counter()
        for feature in features:
            normalized_features[feature] = features[feature]*factor

        return normalized_features


