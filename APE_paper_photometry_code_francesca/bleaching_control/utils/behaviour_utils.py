import numpy as np

# Code from Hernado Martinez Vergara
def RBias(FirstPokes, FirstPokesCorrect):
    '''
    %Returns the bias to the right
    % FirstPokes is a vector of 1s and 2s (Left or Right), indicating the poked port
    % FirstPokesCorrect is a 0 and 1 vector (wrong or correct poke)
    % Both could have NaN values

    % Returns from -1 to 1. 0 Being not biased, 1 being Right-biased, and
    % -1 being left-biased. It is a conservative function. E.g, in a 50-50
    % trial chance, and being totally biased to one side, only half of the
    % trials would be wrong, so the function would output +/-0.5.

    % Correct trials based on proportion of wrong pokes
    % Determine the proportion of wrong pokes to the right side
    '''
    WrongSides = FirstPokes[FirstPokesCorrect == 0]
    if len(WrongSides) < 1:
        RBias = 0
    else:
        WrongSideProportion = len(WrongSides) / len(FirstPokes)  # from 0 to 1
        WrongRightsProportion = WrongSideProportion * np.nansum(WrongSides == 2) / len(WrongSides)
        WrongLeftsProportion = WrongSideProportion * np.nansum(WrongSides == 1) / len(WrongSides)

        RBias = WrongRightsProportion - WrongLeftsProportion
    return RBias


def CalculateRBiasWindow(FirstPokes, FirstPokesCorrect, Window):
    '''Calculates RBias over the lenght of the vectors FirstPokes and
    FirstPokesCorrect using a Window. Returns vector of same lenght'''
    # Create empty vector
    RBiasVector = np.empty(len(FirstPokes))
    RBiasVector[:] = np.nan
    for i in range(Window, len(FirstPokes)):
        win = range((i - Window), i)
        RBiasVector[i] = RBias(FirstPokes[win], FirstPokesCorrect[win])

    return RBiasVector