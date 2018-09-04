
# Genetic Algorithm based Opening Book for Adversarial Game Playing Agent

### Synopsis

In this project, I augment the minimax adversarial search algorithm with a book of opening books to achieve 78% win-rate (against a vanilla minimax agent).

### Isolation

In the game Isolation, two players each control their own single token and alternate taking turns moving the token from one cell to another on a rectangular grid.  Whenever a token occupies a cell, that cell becomes blocked for the remainder of the game.  An open cell available for a token to move into is called a "liberty".  The first player with no remaining liberties for their token loses the game, and their opponent is declared the winner.

In knights Isolation, tokens can move to any open cell that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board.  On a blank board, this means that tokens have at most eight liberties surrounding their current location.  Token movement is blocked at the edges of the board (the board does not wrap around the edges), however, tokens can "jump" blocked or occupied spaces (just like a knight in chess).

Finally, agents have a fixed time limit (150 milliseconds by default) to search for the best move and respond.  The search will be automatically cut off after the time limit expires, and the active agent will forfeit the game if it has not chosen a move.

### Final Report

Please see [report.pdf](https://github.com/ataxali/genetic_opening_book/blob/master/report.pdf) for a summary of the evolutionary genetic algorithm used to construct the opening-move book. 

