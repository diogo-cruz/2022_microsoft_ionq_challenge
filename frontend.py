from collections import namedtuple
import numpy as np
from random import choice
from scipy.spatial.transform import Rotation as R
from backend import game_step, qasm_backend, cartesian_positions, letter_positions, sorted_keys, prob

# Let's start by describing the state in terms of `namedtuple`s.
# This will give us an idea of how to organize the code below,
# by detatching state from the visualization.
# We consider our state immutable.

Guess = namedtuple(
    'Guess', [
        'word', # str: the word the player has guessed.
        'probabilities', # list[list[float]; 26]; 5]
                         #   the probability for each letter we
                         #   got for each letter of the guess.
        'shots', # int: the number of shots used for this guess.
    ])

State = namedtuple(
    'State', [
        'word', # str: the word the player has to guess.
        'guesses', # tuple[Guess]; guesses[-1] is the most recent guess.
    ])


# Step one: draw the five Wordle squares :)
# We'll be using ipycanvas for this.
from ipycanvas import Canvas, Path2D, hold_canvas

def draw_letter_space(canvas, start_at, letter=None):
    """Draws one of those hollow rounded rectangles.
    
    Arguments:
        canvas:   Canvas         The canvas to draw to.
        start_at: Tuple[int,int] The top-left corner of the rectangle to be drawn
        letter:   str?           A single letter to draw inside the rectangle.
                                 If None, no letter is drawn.
    Returns:
        Nothing.
    """
    roundness = 10
    top = 25
    side = 50
    canvas.begin_path()
    x, y = start_at
    canvas.move_to(x, y + roundness)
    x, y = x, y + roundness
    canvas.arc_to(x, y - roundness, x + roundness, y - roundness, roundness)
    x, y = x + roundness, y - roundness
    canvas.line_to(x + top, y)
    x, y = x + top, y
    canvas.arc_to(x + roundness, y, x + roundness, y + roundness, roundness)
    x, y = x + roundness, y + roundness
    canvas.line_to(x, y + side)
    x, y = x, y + side
    canvas.arc_to(x, y + roundness, x - roundness, y + roundness, roundness)
    x, y = x - roundness, y + roundness
    canvas.line_to(x - top, y)
    x, y = x - top, y
    canvas.arc_to(x - roundness, y, x - roundness, y - roundness, roundness)
    x, y = x - roundness, y - roundness
    canvas.line_to(x, y - side)
    canvas.line_width = 2
    canvas.line_cap = 'round'
    canvas.stroke()
    #canvas.fill_style = 'red'
    #canvas.fill()
    
    if letter is not None:
        x, y = start_at
        center_x, center_y = x + (roundness*2 + top)/2, y + (roundness*2 + side)/2
        baseline = 10
        
        kern = 1
        canvas.font = '32px sans-serif'
        canvas.fill_style = 'black'
        canvas.text_align = 'center'
        canvas.fill_text(letter, center_x + kern, center_y + baseline)


def display_word(canvas, word, position=(50, 20), space=50):
    """Shows a (five-letter) word in squares.
    
    Arguments:
        canvas: Canvas               The canvas to draw to.
        word: str                    The word to draw.
                                      Expected to be 5-letter long.
        position: Tuple[int, int]?   The (x,y) at which to start
                                      drawing [default: (50, 20)].
        space: int?                  How much to skip for
                                      each square [default: 100].
    Returns:
        Nothing.
    """
    to_space_and_letter =  lambda x: (position[0] + x[0]*space, x[1])
    for h_offset, letter in map(to_space_and_letter, enumerate(word)):
        draw_letter_space(canvas, (h_offset, position[1]), letter)


def draw_letter_space_color(canvas, start_at, color, alpha, letter=None):
    """Draws one of those hollow rounded rectangles.
    
    Arguments:
        canvas:   Canvas         The canvas to draw to.
        start_at: Tuple[int,int] The top-left corner of the rectangle to be drawn
        letter:   str?           A single letter to draw inside the rectangle.
                                 If None, no letter is drawn.
    Returns:
        Nothing.
    """
    roundness = 10
    top = 25
    side = 50
    
    canvas.begin_path()
    x, y = start_at
    canvas.move_to(x, y + roundness)
    x, y = x, y + roundness
    canvas.arc_to(x, y - roundness, x + roundness, y - roundness, roundness)
    x, y = x + roundness, y - roundness
    canvas.line_to(x + top, y)
    x, y = x + top, y
    canvas.arc_to(x + roundness, y, x + roundness, y + roundness, roundness)
    x, y = x + roundness, y + roundness
    canvas.line_to(x, y + side)
    x, y = x, y + side
    canvas.arc_to(x, y + roundness, x - roundness, y + roundness, roundness)
    x, y = x - roundness, y + roundness
    canvas.line_to(x - top, y)
    x, y = x - top, y
    canvas.arc_to(x - roundness, y, x - roundness, y - roundness, roundness)
    x, y = x - roundness, y - roundness
    canvas.line_to(x, y - side)
    canvas.line_width = 2
    canvas.line_cap = 'round'
    canvas.stroke()
    
    if letter is not None:
        x, y = start_at
        center_x, center_y = x + (roundness*2 + top)/2, y + (roundness*2 + side)/2
        baseline = 20
        
        kern = 1

        canvas.text_align = 'center'

        # Letter
        canvas.font = '32px sans-serif'
        canvas.fill_style = color
        canvas.global_alpha = 1.
        canvas.fill_text(letter, center_x + kern, center_y)
        
        # Probability
        canvas.font = '10px sans-serif'
        canvas.fill_style = 'black'
        canvas.global_alpha = 1.
        canvas.fill_text(str(int(100*alpha))+'%', center_x + kern, center_y + baseline)
        

def display_word_color(canvas, word, position=(50, 20), space=50):
    """Shows a (five-letter) word in squares.
    
    Arguments:
        canvas: Canvas               The canvas to draw to.
        word: str                    The word to draw.
                                      Expected to be 5-letter long.
        position: Tuple[int, int]?   The (x,y) at which to
                                      drawing [default: (50, 20)].
        space: int?                  How much to skip for
                                      each square [default: 100].
    Returns:
        Nothing.
    """
    to_space_and_letter =  lambda x: (position[0] + x[0]*space, x[1][0], x[1][1].color, x[1][1].overlap)
    for h_offset, letter, color, alpha in map(to_space_and_letter, enumerate(word.probabilities)):
        draw_letter_space_color(canvas, (h_offset, position[1]), color, alpha, letter)


import matplotlib.pyplot as plt
# import numpy as np
from matplotlib import cm


# Kind of a HACK; but this is on Matplotlib! It's hard to convince
# it to show a (1,1,1) aspect ratio on 3d plots.
# This is from John Henckel, https://stackoverflow.com/a/70522240/1564310
def set_aspect_equal(ax):
    """ 
    Fix the 3D graph to have similar scale on all the axes.
    Call this after you do all the plot3D, but before show
    """
    X = ax.get_xlim3d()
    Y = ax.get_ylim3d()
    Z = ax.get_zlim3d()
    a = [X[1]-X[0],Y[1]-Y[0],Z[1]-Z[0]]
    b = np.amax(a)
    ax.set_xlim3d(X[0]-(b-a[0])/2,X[1]+(b-a[0])/2)
    ax.set_ylim3d(Y[0]-(b-a[1])/2,Y[1]+(b-a[1])/2)
    ax.set_zlim3d(Z[0]-(b-a[2])/2,Z[1]+(b-a[2])/2)
    ax.set_box_aspect(aspect = (1,1,1))


def draw_sphere(ax, overlaps=None, positions=None):
    """Draws a sphere on the provided ax."""
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.contour3D(x, y, z, 30, alpha=0.5, linewidths=.5, cmap='binary')
    
    # Draw also a band to show the most likely positions
    v_overlaps = 2 * np.arccos(np.sqrt(overlaps))

    u = np.linspace(0, 2*np.pi, 50)
    for i, v in enumerate(v_overlaps):
        if v < 0.05 * np.pi or v > 0.95 * np.pi:
            continue
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.ones_like(u) * np.cos(v)
        data = np.array([x,y,z])
        point = positions[i]
        rotation_matrix = R.from_euler('xz', [point.theta_x * np.pi, point.theta_z * np.pi]).as_matrix()
        data = rotation_matrix @ data
        ax.plot(*data, alpha=0.5)


def draw_line(ax, from_point, to_point, density=200, *args, **kwargs):
    """Draws a straight line in 3D in the ax.
    
    Arguments:
        ax: Axes                         The ax instance to draw to.
                                          (Usually from add_subplot or similar)
        from_point: tuple[float; 3]      Where to start the line (x,y,z).
        to_point: tuple[float; 3]        Where to end the line (x,y,z).
        density: float?                  Number of intermediate points to sample.
        *args                            Passed to ax.plot
        **kwargs                         Passed to ax.plot
    Returns:
        Nothing.
    """
    xpoints = np.linspace(from_point[0], to_point[0], density)
    ypoints = np.linspace(from_point[1], to_point[1], density)
    zpoints = np.linspace(from_point[2], to_point[2], density)
    ax.plot(xpoints, ypoints, zpoints, *args, **kwargs)


def draw_axis(ax):
    """Draws the XYZ axis around (0,0,0) on the ax."""
    length = 1.1
    color_x, color_y, color_z = cm.get_cmap('viridis')(np.linspace(0,1,3))[0: 3]
    draw_line(ax, (0,0,0), (length,0,0), color=color_x)
    draw_line(ax, (0,0,0), (0,length,0), color=color_y)
    draw_line(ax, (0,0,0), (0,0,length), color=color_z)
    ax.text(length + .2, 0, 0, 'x', color=color_x)
    ax.text(0, length + .2, 0, 'y', color=color_y)
    ax.text(0, 0, length + .2, 'z', color=color_z)


from aux_functions import *
import string


def add_letter_labels(ax, colors=None, *args, **kwargs):
    """Adds the 26 letters to the Bloch sphere."""
    # letter_positions is from the global state. This isn't ideal, but helps us
    # keep things consistent across the backend and frontend (since it's imported
    # from the backend at the top).
    for pos, letter in enumerate(letter_positions):
        point = cartesian_positions[pos]
        color = '#3690ae' if colors is None else colors[pos]
        ax.plot(*point, marker="$"+letter+"$", color=color, lw=0, *args, **kwargs)


# We'll need a way to color each letter based on the probability.
# This function converts the probabilities in alphabetical order to
# colors.
def probabilities_to_colors(probabilities):
    """Convert letter probabilities to colors."""
    probabilities = np.array(probabilities)
    colors = cm.get_cmap('RdYlGn')(probabilities**(1/8))
    return colors


def plot_letters(ax, probabilities, overlaps=None, positions=None):
    """Plot the letter distribution on a Bloch sphere."""
    ax.grid(visible=None)
    ax.axis('off')
    ax.tick_params(labelsize=8)
    ax.dist = 6

    draw_axis(ax)
    draw_sphere(ax, overlaps=overlaps, positions=positions)
    add_letter_labels(ax, colors=probabilities_to_colors(probabilities), markersize=15)

    
def render_guess(guess, overlaps=None, positions=None, show_bloch=True):
    """Render a Guess."""
    canvas = Canvas(width=400, height=100)
    with hold_canvas(canvas):
        display_word_color(canvas, guess, position=(50, 20), space=60)
    display(canvas)

    if show_bloch:

        fig = plt.figure()
        fig.set_size_inches(6, 6)
        axs = fig.subplots(1, 1)
        axs.remove()
        ax = fig.add_subplot(projection='3d')

        plot_letters(ax, np.ones(26), overlaps=overlaps, positions=positions)
        set_aspect_equal(ax)
        plt.show()

        
def render_state(state, show_bloch=False):
    """Render all the Guesses of a State."""
    # Show the guesses so far
    for guess in state.guesses:
        render_guess(guess, show_bloch=show_bloch)

        
def render_interactive_guess(guess, nshots, overlaps=None, positions=None, show_bloch=True):
    """Render a Guess with an interactive Bloch sphere."""
    canvas = Canvas(width=410, height=100)
    
    if show_bloch:

        fig = plt.figure()
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(projection='3d')
        plot_letters(ax, np.ones(26), overlaps=overlaps, positions=positions)
        set_aspect_equal(ax)
    
    def on_click(x, y):
        index = max(0, min(4, int(x/82)))
        ax.clear()
        probabilities = [
            prob(guess.word[index], overlaps[index], nshots)[l]
            for l in sorted_keys
        ]
        plot_letters(ax, probabilities, overlaps=[overlaps[index]], positions=[positions[index]])
        set_aspect_equal(ax)
        fig.canvas.show()
        fig.canvas.flush_events()
    
    with hold_canvas(canvas):
        display_word_color(canvas, guess, position=(5, 20), space=80)
    canvas.on_mouse_down(on_click)
    
    display(canvas)

    
class Quordle:
    
    def __init__(self, word=None, backend=None, nshots=1024, only_real_words=False):

        self.only_real_words = only_real_words

        if word is None:
            with open('wordlist.txt', 'r') as wordlist_buf:
                self.wordlist = [w.strip().upper() for w in wordlist_buf.readlines()]
                self.word = choice(self.wordlist)
        else:
            if only_real_words:
                with open('wordlist.txt', 'r') as wordlist_buf:
                    self.wordlist = [w.strip().upper() for w in wordlist_buf.readlines()]
            self.word = word.upper()
        #print(self.word)
        self.n_qb = len(list(self.word))
        
        if backend is None:
            self.backend = qasm_backend
        else:
            self.backend = backend
        self.nshots = nshots
        
        self.state = State(word=self.word,
                           guesses=[])
        
    def guess(self, guess, show=True):
        """Send a guess to the backend."""
        WIN_THRESHOLD = 0.95
        
        guess = guess.upper()
        if self.only_real_words:
            if guess not in self.wordlist:
                print("You must guess a valid word.")
                return
        
        result = game_step(guess, self.word, self.backend, self.nshots)
        
        guess_data = Guess(word=guess,
                      probabilities=result,
                      shots=self.nshots)
        
        self.state.guesses.append(guess_data)

        if show:
            good_positions = []
            overlaps = []
            for letter, score in result:
                pos = letter_positions[letter]
                overlap = score.overlap
                good_positions.append(pos)
                overlaps.append(overlap)
            
            if all(overlap >= WIN_THRESHOLD for overlap in overlaps):
                print('You (probably) won!')
                print("Here's your history")
                render_state(self.state)
            else:
                render_interactive_guess(guess_data, self.nshots, overlaps, good_positions)


            #render_guess(guess_data, overlaps, good_positions)