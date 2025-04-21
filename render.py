from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector, Matrix, Quaternion, TranslationMatrix, ScaleMatrix
from track import Data

import cv2
import numpy as np
import math

width = 512
height = 512
image = Image(width, height, Color(200, 200, 200, 255))

data = Data("IMUData.csv")
data_size = data.data.shape[0] - 1

near = -1
far = -3

# Init z-buffer
zBuffer = np.full(width * height, -float('inf'))

t = 1
time = 0

# Perspective Matrix
Tp = Matrix(np.array(
	[
		[near, 0, 0, 0],
		[0, near, 0, 0],
		[0, 0, near + far, -far * near],
		[0, 0, 1, 0]
	]
))

# After perspective scaling matrix
Tst = Matrix(np.array(
	[
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 2/(near-far), (near+far)/(far-near)],
		[0, 0, 0, 1]
	]
))

# Viewport Matrix
Tvp = Matrix(np.array(
	[
		[width/2, 0, 0, (width-1)/2],
		[0, height/2, 0, (height-1)/2],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	]
))

Tfin = Tvp * Tst

# Load the model
model = Model('data/headset.obj')
model.normalizeGeometry()

def getPerspectiveProjection(p):
	""" Converts a point to screen space with perspective """

	# Apply perspective matrix
	screenP = Tp * p

	# Return to homogenous coordinates
	screenP = screenP / screenP.w

	# Scale z value to fit within [-1, 1]
	# Fit points to viewport
	screenP = Tfin * screenP

	# Extract screen points
	screenX, screenY = screenP.x, screenP.y

	return Vector(screenX, screenY, p.z)

def getVertexNormal(vertIndex, adjFaces, faceNormals):
	""" Compute vertex normals by averaging the normals of adjacent faces """
	normal = Vector(0, 0, 0)
	for faceInd in adjFaces[vertIndex]:
		adjNormal = faceNormals[faceInd]
		normal = normal + adjNormal

	return normal / len(adjFaces[vertIndex])

translation = TranslationMatrix(Vector(0, -0.5, -2))
scale = ScaleMatrix(Vector(1, 1, 1))
Tmodel = translation * scale

transformedVerts = [0] * len(model.vertices)

# Turn all vertices into quaternions
vertQuaternions = []
for i in range(len(model.vertices)):
	vertQuaternions.append(Quaternion(model.vertices[i]))

# Define the light direction
lightDir = Vector(0, 0, -1)

# A list to store all face normals by index
faceNormals = [0] * len(model.faces)

# A dictionary which maps vertices to each face it is connected to
adjFaces = {}
for i in range(len(model.faces)):
	# foreach vertex index in the face
	for j in model.faces[i]:
		if not j in adjFaces:
			adjFaces[j] = []

		# Add this face's normal to this vertex index's normal list
		adjFaces[j].append(i)

orientation = Quaternion(1, 0, 0, 0)
trueUp = Vector(0, 1, 0)

alpha = 0.0001

while True:

	print(t)

	# Read rotation data and get rotation as a quaternion
	angVel = data.getRot(t)
	angVel = Vector(*angVel)
	angVel = Quaternion(angVel.norm(), angVel)

	currentTime = data.getTime(t)
	
	# Apply the rotation
	orientationDelta = (orientation * 0.5) * angVel
	orientation = orientation + (orientationDelta * (currentTime - time))
	orientation = orientation.normalize()

	# Read accelerometer data for tilt correction
	up = data.getAcc(t)
	up = Vector(*up)

	# CHECK IF ACCELERATION IS TOO HIGH TO BE VALID

	# Apply tilt correction with parameter alpha
	up = up.normalize()
	phi = math.acos(trueUp.dot(up))
	tiltAxis = Vector(up.z, 0, -up.x).normalize()
	driftQuaternion = Quaternion(phi * alpha, tiltAxis)

	orientation = driftQuaternion * orientation

	time = currentTime

	# Apply orientation quarternions and model matrix to vertices
	orientationInv = orientation.inv()
	for i in range(len(transformedVerts)):
		transformedVerts[i] = orientation * vertQuaternions[i] * orientationInv
		transformedVerts[i] = Vector(transformedVerts[i])
		transformedVerts[i] = Tmodel * transformedVerts[i]

	# A set of indicies for faces to cull
	culledFaces = set()

	# Calculate face normals
	for i in range(len(model.faces)):
		face = model.faces[i]

		# Get the world coordinates for this face's vertices
		p0, p1, p2 = [transformedVerts[j] for j in face]

		# Calculate this face's normal
		faceNormal = (p2-p0).cross(p1-p0).normalize()

		# How much light should this face recieve
		faceIntensity = faceNormal * lightDir

		# Intensity < 0 means light is shining through the back of the face
		# In this case, don't draw the face at all ("back-face culling")
		if faceIntensity < 0:
			culledFaces.add(i)

		faceNormals[i] = faceNormal

	# Calculate vertex normals and projections
	for vertIndex in range(len(transformedVerts)):

		# If all the faces this vertex is connected to are to be culled, don't process this vertex
		if all(face in culledFaces for face in adjFaces[vertIndex]):
			continue

		vertNorm = getVertexNormal(vertIndex, adjFaces, faceNormals)

		# How much light does this vertex recieve
		intensity = vertNorm * lightDir
		
		# Though this vertex isn't visible, its shading can affect the visible polygon
		if intensity < 0:
			intensity = 0
		
		# Get this vertex's screen point and colour
		projectedVert = getPerspectiveProjection(transformedVerts[vertIndex])
		transformedVerts[vertIndex] = Point(projectedVert.x, projectedVert.y, projectedVert.z, Color(intensity*255, intensity*255, intensity*255, 255))

	# Render the image iterating through faces
	for i in range(len(model.faces)):

		# Don't render culled faces
		if i in culledFaces:
			continue

		face = model.faces[i]

		Triangle(transformedVerts[face[0]], transformedVerts[face[1]], transformedVerts[face[2]]).draw_faster(image, zBuffer)

	cv2.imshow("render", image.buffer)

	# Reset image and z buffers
	image.fill(Color(200, 200, 200, 255))
	zBuffer.fill(-float('inf'))

	# Close program if 'Q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	t += 1

	# Stop if all input data has been read
	if t > data_size:
		break

cv2.destroyAllWindows()