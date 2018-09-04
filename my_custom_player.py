
from sample_players import DataPlayer
import random


class CustomPlayer(DataPlayer):

    def minimax(self, state, depth, player_id):
        def min_value(state, depth, player_id):
            if state.terminal_test(): return state.utility(player_id)
            if depth <= 0: return self.score(state, player_id)
            value = float("inf")
            for action in state.actions():
                value = min(value,
                            max_value(state.result(action), depth - 1,
                                      player_id))
            return value

        def max_value(state, depth, player_id):
            if state.terminal_test(): return state.utility(player_id)
            if depth <= 0: return self.score(state, player_id)
            value = float("-inf")
            for action in state.actions():
                value = max(value,
                            min_value(state.result(action), depth - 1,
                                      player_id))
            return value

        return max(state.actions(),
                   key=lambda x: min_value(state.result(x), depth - 1,
                                           player_id))

    def score(self, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        ########################################################################
        # In order to play the game against itself
        # opening_book strategy substitutes mini-max strategy for player 1
        if self.player_id == 1:
            if state.ply_count < 2:
                self.queue.put(random.choice(state.actions()))
            else:
                self.queue.put(
                    self.minimax(state, depth=3, player_id=self.player_id))
        else:
            # player_id==0 uses opening book
            # the saved opening book contains a few good opening moves
            # so, we choose one from [43, 14, 87, 30, 114, 23, 56, 69, 47, 60]
            OPENING_MOVE = 47
            if self.context is None:
                # check if its the opening move
                if state.ply_count == 0:
                    self.context = self.data[OPENING_MOVE][1]  # future moves
                    self.queue.put(self.data[OPENING_MOVE][0])  # initial cell
                else:
                    # if not opening move, opening book strategy does not apply
                    # apply mini-max for remaining game
                    self.context = []
                    self.queue.put(self.minimax(state, depth=3, player_id=self.player_id))
            elif len(self.context) > 0:
                next_action = self.context.pop(0)
                if next_action not in state.actions():
                    # next move suggested by genome is invalid for current board state
                    # substitute with mini-max
                    self.queue.put(self.minimax(state, depth=3, player_id=self.player_id))
                else:
                    self.queue.put(next_action)
            else:
                # opening book is exhausted, use mini-max for remaining game
                self.queue.put(self.minimax(state, depth=3, player_id=self.player_id))
