""" Module for building an image and writing png files,
	written using only the Python standard library.

	Example usage:
		image = Image(50, 50)
		image.setPixel(0, 49, Color(255, 0, 0, 255))
		image.saveAsPNG("redDot.png")
"""

import zlib, struct
import numpy as np
import cv2
import numbers

class Color(object):
	""" A small class representing a 32-bit RGBA color."""
	def __init__(self, r, g, b, a):
		self.color = (r, g, b, a)

	def r(self):
		return self.color[0]

	def g(self):
		return self.color[1]

	def b(self):
		return self.color[2]

	def a(self):
		return self.color[3]

	def getTuple(self):
		return self.color

	def getHexString(self):
		return "0x%02X%02X%02X%02X" % self.color

	def getByteString(self):
		""" Pack the color as a C-style byte string."""
		return struct.pack('>4B', self.color[0], self.color[1], self.color[2], self.color[3])

	def getAlphaBlend(self, destColor):
		""" Alpha blend this color with the provided destination color."""
		alpha = self.a() / 255
		outR = int(self.r() * alpha) + int(destColor.r() * (1 - alpha))
		outG = int(self.g() * alpha) + int(destColor.g() * (1 - alpha))
		outB = int(self.b() * alpha) + int(destColor.b() * (1 - alpha))
		outA = self.a() + int(destColor.a() * (1-alpha))
		return Color(outR, outG, outB, outA)
	
	def __mul__(self, other):
		""" Multiplies all color channel components by a value. """

		if isinstance(other, numbers.Real):
			return Color(self.color[0]*other, self.color[1]*other, self.color[2]*other, self.color[3])

class Image(object):
	""" An image class capable of generating and saving a PNG.
		Attributes:
			width: The width of the image
			height: The height of the image
			buffer: Representation of the image storing Color values for each pixel
	"""
	def __init__(self, width, height, color = Color(0, 0, 0, 255)):
		""" Create the buffer, fill it with black pixels."""
		self.width = width
		self.height = height

		self.buffer = np.zeros((height, width, 4), dtype=np.uint8)
		self.fill(color)
	
	def fill(self, color):
		self.buffer[:] = color.getTuple()

	def setPixel(self, x, y, color):
		""" Set the color value for the pixel at (x, y)."""
		if (x not in range(0, self.width)) or (y not in range (0, self.height)):
			return

		# Flip Y coordinate so that up is positive
		pixel = self.buffer[-y, x]

		# # Blend the new color with the destination color
		outColor = color.getAlphaBlend(Color(*pixel)).getTuple()

		# Set the new pixel colors in the buffer
		self.buffer[-y, x, :] = outColor

	def saveAsPNG(self, filename = "render.png"):
		cv2.imwrite(filename, self.buffer)