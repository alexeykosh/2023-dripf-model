import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def letter_subplots(axes=None, 
                    letters=None, 
                    xoffset=-0.1, 
                    yoffset=1.0, 
                    **kwargs):
    """Add letters to the corners of subplots (panels). By default each axis is
    given an uppercase bold letter label placed in the upper-left corner.
    Args
        axes : list of pyplot ax objects. default plt.gcf().axes.
        letters : list of strings to use as labels, default ["A", "B", "C", ...]
        xoffset, yoffset : positions of each label relative to plot frame
          (default -0.1,1.0 = upper left margin). Can also be a list of
          offsets, in which case it should be the same length as the number of
          axes.
        Other keyword arguments will be passed to annotate() when panel letters
        are added.
    Returns:
        list of strings for each label added to the axes
    Examples:
        Defaults:
            >>> fig, axes = plt.subplots(1,3)
            >>> letter_subplots() # boldfaced A, B, C
        
        Common labeling schemes inferred from the first letter:
            >>> fig, axes = plt.subplots(1,4)        
            >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
        Fully custom lettering:
            >>> fig, axes = plt.subplots(2,1)
            >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
        Per-axis offsets:
            >>> fig, axes = plt.subplots(1,2)
            >>> letter_subplots(axes, xoffset=[-0.1, -0.15])
            
        Matrix of axes:
            >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
            >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix

    See also: https://github.com/matplotlib/matplotlib/issues/20182
    """

    # get axes:
    if axes is None:
        axes = plt.gcf().axes
    # handle single axes:
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)])
    llets = list('abcdefghijklmnopqrstuvwxyz'[:len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [ "({})".format(lett) for lett in llets ]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [ "({})".format(lett) for lett in ulets ]
        fontweight = "normal"
    elif letters in ("lower", "lowercase", "a"):
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset]*len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset]*len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(fontweight=fontweight, fontsize='large', ha="center",
                       va='center', xycoords='axes fraction', annotation_clip=False)
    kwargs = dict( list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax,lbl,xoff,yoff in zip(axes,letters,xoffset,yoffset):
        t = ax.annotate(lbl, xy=(xoff,yoff), **kwargs)
        list_txts.append(t)
    return list_txts


def model(selection, mutation_rate, 
          zipf=1.5, 
          n_meanings=100, 
          n_agents=1000, 
          n_generations=20, 
          detailed=False, 
          num_encounters=None):  
    """
    Optimized model with vectorized operations.

    Parameters
    ----------
    selection : float
        Selection rate.
    mutation_rate : float
        Mutation rate.
    zipf : float
        Zipf parameter.
    n_meanings : int
        Number of meanings.
    n_agents : int
        Number of agents.
    n_generations : int
        Number of generations.
    detailed : bool
        Whether to return detailed output.
    num_encounters : int
        Number of encounters per generation.

    Returns
    -------
    If detailed is False:
    corr_coeff : float
        Correlation coefficient.
    ~~~~~~~~~~
    If detailed is True:
    corr_coeffs : list
        List of correlation coefficients.
    word_meaning_matrix : np.array
        Word-meaning matrix.
    freq_meanings : np.array
        Frequencies of meanings.
    """
    
    # Default number of encounters per generation
    n_encounters = num_encounters if num_encounters is not None else n_meanings * 5

    # Initialize frequency of meanings and normalize
    freq_meanings = np.random.zipf(zipf, size=n_meanings).astype(np.float64)
    freq_meanings /= np.sum(freq_meanings)

    # Initialize word-meaning matrix and correlation coefficient list
    word_meaning_matrix = np.random.uniform(0, 20, (n_agents, n_meanings))
    corr_coeffs = []

    # Compute initial correlation
    corr_coeffs.append(np.corrcoef(freq_meanings, np.mean(word_meaning_matrix, axis=0))[0, 1])

    # Pre-generate mutation decisions for each generation
    mutation_indices = np.random.rand(n_generations, n_agents) < mutation_rate

    for gen in range(n_generations):
        # Generate agent pairs and meanings for encounters
        agent_pairs = np.random.choice(n_agents, (n_encounters, 2), replace=True)
        meanings = np.random.choice(n_meanings, n_encounters, p=freq_meanings)

        # Encounter loop with selection and copying
        for i in range(n_encounters):
            agent1, agent2 = agent_pairs[i]
            meaning = meanings[i]
            wordform1, wordform2 = word_meaning_matrix[agent1, meaning], word_meaning_matrix[agent2, meaning]

            if np.random.rand() <= selection:
                # Selection: both agents adopt the minimum wordform
                min_wordform = min(wordform1, wordform2)
                word_meaning_matrix[agent1, meaning] = min_wordform
                word_meaning_matrix[agent2, meaning] = min_wordform
            else:
                # Random copying without selection
                if np.random.rand() < 0.5:
                    word_meaning_matrix[agent1, meaning] = wordform2
                else:
                    word_meaning_matrix[agent2, meaning] = wordform1

        # Apply mutation to a subset of agents if selected
        mutated_agents = np.where(mutation_indices[gen])[0]
        word_meaning_matrix[mutated_agents, :] = np.random.uniform(0, 20, size=(len(mutated_agents), n_meanings))

        # Compute and store correlation for the current generation
        corr_coeffs.append(spearmanr(freq_meanings, np.mean(word_meaning_matrix, axis=0)).correlation)

    # Return detailed or final results based on `detailed` parameter
    if detailed:
        return corr_coeffs, word_meaning_matrix, freq_meanings
    else:
        return corr_coeffs[-1]


# def model(selection, mutation_rate, 
#           zipf=1.5, 
#           n_meanings=100, 
#           n_agents=1000, 
#           n_generations = 20,
#           detailed=False,
#           num_encounters=None):  
#     """
#     Run the model with the given parameters.

#     Parameters
#     ----------
#     selection : float
#         Selection rate.
#     mutation_rate : float
#         Mutation rate.
#     zipf : float
#         Zipf parameter.
#     n_meanings : int
#         Number of meanings.
#     n_agents : int
#         Number of agents.
#     n_generations : int
#         Number of generations.
#     detailed : bool
#         Whether to return detailed output.
#     num_encounters : int
#         Number of encounters per generation.

#     Returns
#     -------
#     If detailed is False:
#     corr_coeff : float
#         Correlation coefficient.
#     ~~~~~~~~~~
#     If detailed is True:
#     corr_coeffs : list
#         List of correlation coefficients.
#     word_meaning_matrix : np.array
#         Word-meaning matrix.
#     freq_meanings : np.array
#         Frequencies of meanings.
#     """

#     # number of encounters per generation is 
#     # 5 times the number of meanings by default,
#     # but can be set to a fixed number if desired
#     if num_encounters is not None:
#         n_encounters = num_encounters
#     else:
#         n_encounters = n_meanings * 5

#     corr_coeffs = []

#     freq_meanings = np.random.zipf(zipf, size=n_meanings)
#     freq_meanings = freq_meanings / np.sum(freq_meanings)
#     word_meaning_matrix = np.random.uniform(0, 20, (n_agents, n_meanings))

#     corr_coeffs.append(np.corrcoef(freq_meanings, np.mean(word_meaning_matrix, axis=0))[0, 1])

#     for _ in range(n_generations):
#         for _ in range(n_encounters):
#             agent1, agent2 = np.random.choice(n_agents, 2, replace=False)
#             meaning = np.random.choice(n_meanings, p=freq_meanings)
#             wordform1, wordform2 = word_meaning_matrix[agent1, meaning], word_meaning_matrix[agent2, meaning]

#             if np.random.rand() <= selection:
#                 word_meaning_matrix[agent2, meaning] = np.minimum(wordform1, wordform2)
#                 word_meaning_matrix[agent1, meaning] = np.minimum(wordform1, wordform2)
#             else:
#                 if np.random.rand() < 0.5:
#                     word_meaning_matrix[agent1, meaning] = wordform2
#                 else:
#                     word_meaning_matrix[agent2, meaning] = wordform1

#         mutation_agents = np.random.rand(n_agents) < mutation_rate
#         mutated_agents = np.where(mutation_agents)[0]
#         word_meaning_matrix[mutated_agents, :] = np.random.uniform(0, 20, size=(len(mutated_agents), n_meanings))

#         corr_coeffs.append(spearmanr(freq_meanings, np.mean(word_meaning_matrix, axis=0)).correlation)

#     if detailed:
#         return corr_coeffs, word_meaning_matrix, freq_meanings
#     else:
#         return spearmanr(freq_meanings, np.mean(word_meaning_matrix, axis=0)).correlation
