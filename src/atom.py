import numpy as np
from enum import Enum

#bwaaa why doenst pytohn have const
U_TO_KG = 1.66053906660e-27

#value should be atomic number, useful later
class Element(Enum):
    HYDROGEN = 1

#in atomic mass units for readability, convert later
ELEMENT_MASS_U = {
    Element.HYDROGEN: 1.008,
}

class Atom:
    #hydrogen default element
    def __init__(self, position, element: Element = Element.HYDROGEN):
        self.position = np.array(position, dtype=float)
        self.element = element
        self.velocity = np.zeros(3)

    @property
    def mass(self):
        return ELEMENT_MASS_U[self.element] * U_TO_KG
