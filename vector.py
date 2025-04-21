import numbers
import numpy as np
import math

def clamp(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val

class Vector(object):
    """ A vector with useful vector / matrix operations. """
    def __init__(self, *args):     

        self.components = np.zeros(4)

        if isinstance(args[0], Quaternion):
            self.components[0] = args[0].y
            self.components[1] = args[0].z
            self.components[2] = args[0].w
            self.components[3] = 1
        else:        
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
        """ Initialise a matrix as a 2D numpy array"""
        if len(args) == 0:
            self.components = np.eye(4)
        else:
            self.components = args[0]
    
    def __mul__(self, other):       
        if isinstance(other, Matrix):
            """ Matrix multiplication"""
            return Matrix(self.components @ other.components)
        elif isinstance(other, Vector):
            """ Vector-Matrix multiplication"""
            return Vector(*(self.components @ other.components))
        elif isinstance(other, numbers.Real):
            """ Scalar multiplication"""
            return Matrix(self.components * other)

class TranslationMatrix(Matrix):
    def __init__(self, t):
        """ Takes a vector t and creates a matrix
            which translates all components of a vector by the corresponding t components
        """

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
        """ Return inverse translation matrix"""
        return TranslationMatrix(self.t * -1)

class ScaleMatrix(Matrix):
    def __init__(self, s):
        """ Takes a scalar s and creates a matrix
            which scales all components of a vector by s
        """

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
        """ Return inverse scalar matrix"""
        ScaleMatrix(Vector(1/self.s.x, 1/self.s.y, 1/self.s.z))

class Quaternion(Vector):
    def __init__(self, *args):
        self.components = np.zeros(4)

        if len(args) == 1:
            """ Create quaternion from vector """

            self.components[0] = 0
            self.components[1] = args[0].x
            self.components[2] = args[0].y
            self.components[3] = args[0].z

            self.angle, self.axis = self.getAxisAngle()
        
        elif len(args) == 2:
            """ Create quaternion from angle and axis """
            
            self.angle = args[0]
            self.axis = args[1]

            self.components[0] = math.cos(self.angle/2)

            sinangle = math.sin(self.angle/2)
            self.components[1] = self.axis.x * sinangle
            self.components[2] = self.axis.y * sinangle
            self.components[3] = self.axis.z * sinangle
        
        elif len(args) == 3:
            """ Create quaternion from Euler angles """

            pitch = args[0] / 2 # x is pitch
            yaw = args[1] / 2   # y is yaw
            roll = args[2] / 2  # z is roll

            # From Wikipedia formulae
            cp = math.cos(pitch)
            sp = math.sin(pitch)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            cr = math.cos(roll)
            sr = math.sin(roll)

            self.components[0] = cr * cp * cy + sr * sp * sy
            self.components[1] = cr * sp * cy + sr * cp * sy
            self.components[2] = cr * cp * sy - sr * sp * cy
            self.components[3] = sr * cp * cy - cr * sp * sy

            self.angle, self.axis = self.getAxisAngle()

        elif len(args) == 4:
            """ Create quaternion from 4 components """

            self.components[0] = args[0]
            self.components[1] = args[1]
            self.components[2] = args[2]
            self.components[3] = args[3]    

            self.angle, self.axis = self.getAxisAngle()        
    
    def getAxisAngle(self):
        """ Calculates and returns this quaternion's corresponding angle and axis """
        
        angle = clamp(self.components[0], -1, 1)

        angle = 2 * math.acos(angle)

        if angle == 0:
            return 0, Vector(0, 0, 0)
        
        scalar = 1/math.sin(angle/2)
        axis = Vector(*self.components[1:]) * scalar

        return angle, axis
    
    # From Wikipedia formulae
    def getEuler(self):
        """ Calculates and returns this quaternion's corresponding Euler angles """
        
        # Account for floating point error
        def fpCorrection(val):
            if abs(val) < 1e-10:
                return 0
            else:
                return val

        wx = self.components[0] * self.components[1]
        wy = self.components[0] * self.components[2]
        wz = self.components[0] * self.components[3]

        x2 = self.components[1] * self.components[1]

        p = clamp(2 * (wx - (self.components[2] * self.components[3])), -1, 1)
        y1 = 2 * (wy + (self.components[1] * self.components[3]))
        y2 = 1 - 2 * (x2 + (self.components[2] * self.components[2]))
        r1 = 2 * (wz + (self.components[1] * self.components[2]))
        r2 = 1 - 2 * (x2 + (self.components[3] * self.components[3]))

        p = fpCorrection(p)
        y1 = fpCorrection(y1)
        y2 = fpCorrection(y2)
        r1 = fpCorrection(r1)
        r2 = fpCorrection(r2)

        pitch = math.asin(p)
        yaw = math.atan2(y1, y2)
        roll = math.atan2(r1, r2)

        return pitch, yaw, roll

    def inv(self):
        """ Return the inverse of this quaternion """
        return Quaternion(self.angle, self.axis * -1)
    
    def norm(self):
        """ Return the norm (magnitude) of this quaternion"""
        return np.linalg.norm(self.components)

    def normalize(self):
        """ Return a normalized unit vector from this quaternion"""
        magnitude = self.norm()
        return Quaternion(*(self.components / magnitude))
    
    def __mul__(self, other):
        if type(other) == type(self):
            """ Quaternion multiplication"""
            return Quaternion(
                self.x * other.x - self.y * other.y - self.z * other.z - self.w * other.w,
                self.x * other.y + other.x * self.y + self.z * other.w - other.z * self.w,
                self.x * other.z + other.x * self.z + other.y * self.w - self.y * other.w,
                self.x * other.w + other.x * self.w + self.y * other.z - other.y * self.z
            )
        
        elif isinstance(other, numbers.Real):
            """ Scalar multiplication"""  
            return Quaternion(*(self.components * other))
    
    def __add__(self, other):
        if type(other) == type(self):
            """ Quaternion addition"""
            return Quaternion(*(self.components + other.components))
