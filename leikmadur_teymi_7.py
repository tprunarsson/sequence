# DEMO how to submit you players, example random and SelfPlay100
import torch
from torch.autograd import Variable
import numpy as np

cards_on_board = np.matrix([[-1, 0,11,10, 9, 8, 7, 6, 5,-1],
                            [24,18,19,20,21,22,23,12, 4,13],
                            [35,17, 9, 8, 7, 6, 5,25, 3,14],
                            [34,16,10,43,42,41, 4,26, 2,15],
                            [33,15,11,44,37,40, 3,27, 1,16],
                            [32,14, 0,45,38,39, 2,28,36,17],
                            [31,13,24,46,47,36, 1,29,47,18],
                            [30,37,35,34,33,32,31,30,46,19],
                            [29,38,39,40,41,42,43,44,45,20],
                            [-1,28,27,26,25,12,23,22,21,-1]])

class leikmadur_teymi_7a:

  def __init__(self, player):
    self.name = "Teymi_7b Random"
    self.player = player 

  def policy(self, discs_on_board, cards_in_hand, legal_moves, legal_moves_1J, legal_moves_2J):
    na = len(legal_moves) + len(legal_moves_1J) + len(legal_moves_2J)
    move_played = [m for count in range(3) for m in [legal_moves,legal_moves_1J,legal_moves_2J][count]]
    for ell in range(len(legal_moves)):  
        (i,j) = legal_moves[ell]
    if (na > 0):
        k = int(np.random.choice(range(na),1))
    else:
        return (None,None), None
    (i,j) = move_played[k]
    if k < len(legal_moves):
        spil = cards_on_board[i,j]
    elif k < (len(legal_moves) + len(legal_moves_1J)):
        spil = 48
    else:
        spil = 49
    return (i,j), spil


class leikmadur_teymi_7b:

  def __init__(self, player):
    self.name = "Teymi_7a SelfPlay100"
    self.player = player
    self.model = 4 * [None]
    self.model[0] = torch.load('./teymi7/b1_trained_100.pth')
    self.model[1] = torch.load('./teymi7/w1_trained_100.pth')
    self.model[2] = torch.load('./teymi7/b2_trained_100.pth')
    self.model[3] = torch.load('./teymi7/w2_trained_100.pth')

  def getfeatures(self, discs_on_board, move, disc, card, normal_moves, len1J, len2J, kmove, player, p1J=0, p2J=0, n = 2):
    b = discs_on_board.copy()
    b[move[0],move[1]] = disc # the look-ahead
    otherplayer = player % n + 1
    # lets put our player in the corners
    b[0,0] = b[0,-1] = b[-1,0] = b[-1,-1] = player
    # other potential moves we can make with our current hand
    for k in range(len(normal_moves)):
        if k != kmove:
            (i,j) = normal_moves[k]
            b[i,j] = -1 # potential move for the player in the future!
    b = b.reshape(b.size)
    x = np.concatenate((b == 0, b == player, b == otherplayer, b == -1, np.array([len1J]), np.array([len2J]),np.array([p1J]), np.array([p2J])))
    return(x)

  def policy(self, discs_on_board, cards_in_hand, legal_moves, legal_moves_1J, legal_moves_2J):
    discs_on_board = discs_on_board.copy()
    nx = discs_on_board.size*4+2+2
    # now we start by finding all after-states for the possible moves
    lenJ1 = np.sum(cards_in_hand == 48)
    lenJ2 = np.sum(cards_in_hand == 49)
    na = len(legal_moves) + len(legal_moves_1J) + len(legal_moves_2J)
    move_played = [m for count in range(3) for m in [legal_moves,legal_moves_1J,legal_moves_2J][count]]   
    # walk through all the possible moves, k = 0,1,...
    k = 0
    if na > 0:
      card_played = np.zeros(na)
      xa = np.zeros((nx,na))
      for ell in range(len(legal_moves)):  
        (i,j) = tuple(legal_moves[ell])
        card_played[k] = cards_on_board[i,j]
        xa[:,k] = self.getfeatures(discs_on_board, (i,j), self.player, cards_on_board[i, j], legal_moves, lenJ1, lenJ2, k, player = self.player)
        k += 1
      for ell in range(len(legal_moves_1J)):
        (i,j) = tuple(legal_moves_1J[ell])
        card_played[k] = 48
        xa[:,k] = self.getfeatures(discs_on_board, (i,j), 0, 48, legal_moves, lenJ1-1, lenJ2, k, player = self.player,p1J=1.)
        k += 1
      for ell in range(len(legal_moves_2J)):
        (i,j) = tuple(legal_moves_2J[ell])
        card_played[k] = 49
        xa[:,k] = self.getfeatures(discs_on_board, (i,j), self.player, 49, legal_moves, lenJ1, lenJ2-1, k, player = self.player,p2J=1.)
        k += 1
      # now choose an epsilon greedy move
      x = Variable(torch.tensor(xa, dtype = torch.float, device = torch.device('cpu'))) 
      # now do a forward pass to evaluate the board's after-state value
      h = torch.mm(self.model[1],x) + self.model[0] @ torch.ones((1,na))  # matrix-multiply x with input weight w1 and add bias
      h_tanh = h.tanh() # squash this with a sigmoid function
      y = torch.mm(self.model[3],h_tanh) + self.model[2] # multiply with the output weights w2 and add bias
      va = y.sigmoid().detach() # .cpu()
      k = torch.argmax(va[0,:]) # greedy, should we break ties here, I assume there will be none...
      (i,j) = move_played[k] # get the actual corresponding epsilon greedy move
      spil = int(card_played[k]) # here we actually update the board
    else:
      return (None,None), None
    return (i,j), spil

