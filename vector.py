import numbers
import numpy as np

class Vector(object):
    """ A vector with useful vector / matrix operations.
    """
    def __init__(self, *args):     

        # IMPLEMENT VECTOR FROM QUATERNION

        self.components = np.zeros(4)
           
        for i in range(len(args)):
            self.components[i] = args[i]
        
        if len(args) == 3:
            self.components[3] = 1
        
        

    @property
    def x(self):
        assert(len(self) >= 1)
        return self.components[0]

    @x.setter
    def x(self, val):
        self.components[0] = val

    @property
    def y(self):
        assert(len(self) >= 2)
        return self.components[1]

    @y.setter
    def y(self, val):
        self.components[1] = val

    @property
    def z(self):
        assert(len(self) >= 3)
        return self.components[2]

    @z.setter
    def z(self, val):
        self.components[2] = val
    
    @property
    def w(self):
        assert(len(self) >= 4)
        return self.components[3]

    @w.setter
    def w(self, val):
        self.components[3] = val

    def norm(self):
        """ Return the norm (magnitude) of this vector."""
        return np.linalg.norm(self.components[:-1])

    def normalize(self):
        """ Return a normalized unit vector from this vector."""
        magnitude = self.norm()
        return Vector(*(self.components[:-1] / magnitude))

    def dot(self, other):
        """ Return the dot product of this and another vector."""
        return np.dot(self.components[:-1], other.components[:-1])

    def cross(self, other):
        """ Return the cross product of this and another vector."""
        assert len(self) == len(other), "Vectors must be the same size."
        assert len(self.components[:-1]) == 3, "Cross product only implemented for 3D vectors."
        return Vector(*np.cross(self.components[:-1], other.components[:-1]))


    # Overrides
    def __mul__(self, other):
        """ If multiplied by another vector, return the dot product. 
            If multiplied by a number, multiply each component by other.
        """
        if type(other) == type(self):
            return self.dot(other)
        elif isinstance(other, numbers.Real):
            return Vector(*(self.components * other))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vector(*(self.components / other))
    
    def __add__(self, other):
        return Vector(*(self.components + other.components))
    
    def __sub__(self, other):
        return Vector(*(self.components - other.components))

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return self.components.__iter__()

class Matrix(object):
    def __init__(self, *args):
        if len(args) == 0:
            self.components = np.eye(4)
        else:
            self.components = args[0]
    
    def __mul__(self, other):
        """ If multiplied by another vector, return the dot product. 
            If multiplied by a number, multiply each component by other.
        """
        
        if isinstance(other, Matrix):
            return Matrix(self.components @ other.components)
        elif isinstance(other, Vector):
            return Vector(*(self.components @ other.components))
        elif isinstance(other, numbers.Real):
            return Matrix(self.components * other)

class TranslationMatrix(Matrix):
    def __init__(self, t):
        self.t = t
        self.components = np.array(
            [
                [1, 0, 0, t.x],
                [0, 1, 0, t.y],
                [0, 0, 1, t.z],
                [0, 0, 0, 1]
            ]
        )
    
    def inv(self):
        return TranslationMatrix(self.t * -1)

class ScaleMatrix(Matrix):
    def __init__(self, s):
        self.s = s
        self.components = np.array(
            [
                [s.x, 0, 0, 0],
                [0, s.y, 0, 0],
                [0, 0, s.z, 0],
                [0, 0, 0, 1]
            ]
        )
    
    def inv(self):
        ScaleMatrix(Vector(1/self.s.x, 1/self.s.y, 1/self.s.z))

class Quaternion(Vector):
    def __init__(self, *args):
        self.components = np.zeros(4)

        # Convert Vector to quaternion
        if len(args) == 1:

            # IMPLEMENT AXIS AND ANGLE COMPUTATION

            self.components[0] = 0
            self.components[1] = args[0].x
            self.components[2] = args[0].y
            self.components[3] = args[0].z
        # Create quaternion from 4 components
        elif len(args) == 4:

            # IMPLEMENT AXIS AND ANGLE COMPUTATION

            self.components[0] = args[0]
            self.components[1] = args[1]
            self.components[2] = args[2]
            self.components[3] = args[3]
        # Create quaternion from angle and axis
        else:
            self.angle = args[0]
            self.a = args[1]

            self.components[0] = np.cos(self.angle/2)

            sinangle = np.sin(self.angle/2)
            self.components[1] = self.a.x * sinangle
            self.components[2] = self.a.y * sinangle
            self.components[3] = self.a.z * sinangle
    
    def inv(self):
        return Quaternion(self.angle, self.a * -1)
    
    def __mul__(self, other):
        # Quaternion multiplication
        if type(other) == type(self):       
            return Quaternion(
                self.x * other.x - self.y * other.y - self.z * other.z - self.w * other.w,
                self.x * other.y + other.x * self.y + self.z * other.w - other.z * self.w,
                self.x * other.z + other.x * self.z + other.y * self.w - self.y * other.w,
                self.x * other.w + other.x * self.w + self.y * other.z - other.y * self.z
            )
