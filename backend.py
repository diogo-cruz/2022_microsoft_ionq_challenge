import numpy as np
from math import log, exp, pow, sin, cos
import matplotlib.pyplot as plt

from string import ascii_uppercase
from typing import NamedTuple

letters = ascii_uppercase

class Point(NamedTuple):
    theta_x: float
    theta_z: float
    
    def cartesian(self):
        theta, phi = self.theta_x*np.pi, (self.theta_z - 0.5)*np.pi
        return (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))

positions = [
    Point(0,0),
    Point(1,0),
    Point(0.5,0.5),
    Point(0.5,-0.5),
    Point(-0.5,0),
    Point(0.5,0),
    Point(0.5,0.25),
    Point(0.5,0.75),
    Point(0.5,1.25),
    Point(0.5,1.75),
    Point(0.25,0),
    Point(0.75,0),
    Point(1.25,0),
    Point(1.75,0),
    Point(0.25,0.5),
    Point(0.75,0.5),
    Point(0.75,-0.5),
    Point(0.25,-0.5),
    Point(0.25,0.25),
    Point(0.25,0.75),
    Point(0.25,1.25),
    Point(0.25,1.75),
    Point(0.75,0.25),
    Point(0.75,0.75),
    Point(0.75,1.25),
    Point(0.75,1.75),
]

cartesian_positions = [point.cartesian() for point in positions]

# Option 1
# Assigns a letter to each position in alphabetical order
# letter_positions = {
#     letter: position
#     for letter, position in zip(letters, positions)
# }

# Option 2
# Assigns a letter to each position in decreasing frequency order
abs_freq = {}
with open('wordlist.txt', 'r') as wordlist_buf:
    for word in wordlist_buf.readlines():
        for letter in word.strip().upper():
            abs_freq[letter] = abs_freq.get(letter, 0) + 1
            
sorted_keys = sorted(list(abs_freq.keys()), key=lambda k: abs_freq[k], reverse=True)

# fig = plt.figure()
# ax = fig.subplots()
# ax.bar(np.arange(len(sorted_keys)), [abs_freq[k] for k in sorted_keys])

# rects = ax.patches
# for rect, label in zip(rects, sorted_keys):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
#             ha='center', va='bottom')
    
# plt.show()

letter_positions = {
    letter: positions[i]
    for i, letter in enumerate(sorted_keys)
}

# Option 3
# letter_positions = {
#     'A': 1,
#     'B': ,
#     'C': ,
#     'D': ,
#     'E': 0,
#     'F': ,
#     'G': ,
#     'H': ,
#     'I': 5,
#     'J': ,
#     'K': ,
#     'L': ,
#     'M': ,
#     'N': ,
#     'O': 4,
#     'P': ,
#     'Q': ,
#     'R': 3,
#     'S': 2,
#     'T': ,
#     'U': ,
#     'V': ,
#     'W': ,
#     'X': ,
#     'Y': ,
#     'Z': 
# }

from qiskit.circuit import QuantumCircuit

def letter_to_gate(letter: str) -> QuantumCircuit:
    """Create a gate that prepares the quantum state associated
    to :letter: when given as input |0>."""
    gate = QuantumCircuit(1, name=letter)
    
    letter_point = letter_positions[letter]
    gate.rx(letter_point.theta_x * np.pi, 0)
    gate.rz(letter_point.theta_z * np.pi, 0)
    
    return gate

def word_to_circuit(word: str) -> QuantumCircuit:
    """Create a gate that prepares the state associated
    with the given word, when given as input |0...0>.
    """
    circuit = QuantumCircuit(len(word), name=word)
    
    for qb, letter in enumerate(word):
        circuit.append(letter_to_gate(letter), [qb])
        
    return circuit
        
def comparison_circuit(guess:str , target:str) -> QuantumCircuit:
    """Generates a circuit to compare the :guess: with the :target:"""
    guess_circuit = word_to_circuit(guess)
    target_circuit = word_to_circuit(target)
    
    return QuantumCircuit.compose(
        guess_circuit,
        target_circuit.inverse()
    )

from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

qasm_backend = Aer.get_backend('qasm_simulator')

def measure_overlaps(guess: str, target: str, backend, nshots: int) -> list:
    """Experimentally measure overlap between :guess: and :target:, letter-wise."""
    circuit = comparison_circuit(guess, target).decompose()
    circuit.measure_all()
    
    counts = backend.run(circuit, shots=nshots).result().get_counts()
    
    overlaps = [
        (letter, sum((obs[-qubit-1]=='0')*count for obs, count in counts.items()) / nshots)
        for qubit, letter in enumerate(guess)
    ]
    
    return overlaps

class Score(NamedTuple):
    overlap: float
    color: str

def game_step(guess, target, backend, nshots, calculate_yellows=False):
    """One step of the game. The user sends a :guess: string, which
    will be compared with the :target: string via a quantum circuit.
    
    We return a list of the letters and an associated score.
    
    The score is given as such:
      - Green, if the target letter in the same position has a high overlap
      - Yellow, otherwise. We then return the maximum overlap with all the other letters
      in the target
    """
    
    GREEN_THRESHOLD = 0.95
    YELLOW_THRESHOLD = 0.95 if calculate_yellows else 1.10
    
    # First, compare guess with target, letter-by-letter
    overlaps_same_position = measure_overlaps(guess, target, backend, nshots)
    
    # Now, compare the target with the following words:
    # GGGGG, UUUUU, EEEEE, SSSSS, SSSSS
    max_overlap = {}
    if calculate_yellows:
        for letter in guess:
            new_word = letter * len(guess)
            overlaps = measure_overlaps(new_word, target, backend, nshots)
            max_overlap[letter] = max(overlap for _, overlap in overlaps)
    
    results = []
    for i, letter in enumerate(guess):
        _, overlap = overlaps_same_position[i]
        if not calculate_yellows:
            max_overlap[letter] = overlap
        
        if overlap >= GREEN_THRESHOLD:
            results.append((letter, Score(overlap, 'green')))
        elif max_overlap[letter] >= YELLOW_THRESHOLD:
            #overlap = max_overlap[letter]
            results.append((letter, Score(overlap, 'yellow')))
        else:
            results.append((letter, Score(overlap, 'black')))
            
    return results

from math import log, exp, pow, sin, cos, fsum
import cmath
from math import log, exp, pow, sin, cos, fsum
import cmath

def expected_overlap(a, b):
    """Calculate the expected overlap (absolute value of inner value squared)
    of two letters :a: and :b:."""
    def Rx(x):
        return np.array([
            [      cos(x/2), -1j * sin(x/2)],
            [-1j * sin(x/2),       cos(x/2)],
        ])
    
    def Rz(x):
        return np.array([
            [cmath.exp(- 1j * x/2), 0                  ],
            [0,                     cmath.exp(1j * x/2)],
        ])
    
    a, b = letter_positions[a], letter_positions[b]
    
    v = np.array([[1], [0]])
    
    v_ = Rx(b.theta_x * np.pi).conjugate().T @ Rz(b.theta_z * np.pi).conjugate().T @ Rz(a.theta_z * np.pi) @ Rx(a.theta_x * np.pi) @ v
    
    return np.abs(v_[0,0])**2

def prob(letter, overlap, nshots):
    """Given a :letter: and a measured :overlap: with
    some unknown vector, return a dictionary with the alphabet
    and the probability that each letter is the correct one.
    
    To estimate p(some letter | data) we use conditional probabilities.
    We calculate how likely it is that a given letter would have an overlap of :overlap:
    with :letter: after measuring :nshots times.
    """

    f = overlap
    N = nshots
    probabilities = {}
    
    for letter_ in letters:
        p = expected_overlap(letter, letter_)
        p = max(p, 1.e-20)
        
        q = max(1 - p, 1.e-20)
        
        probabilities[letter_] = exp(
            -N * (f-p)**2 / (2*p*q)
        ) / np.sqrt(2*np.pi*N*p*q)
            
        
    normalization = fsum(probabilities.values())
    return {letter_: prob/normalization for letter_, prob in probabilities.items()}
        
        