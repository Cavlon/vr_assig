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
far = -6

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

def getPerspectiveProjection(p):
	""" Converts a point to screen space with perspective """

	# Apply perspective matrix
	screenP = Tp * p

	# Prevent a division by 0 error with a slight offset
	if screenP.w == 0:
		screenP.w = -1e-4

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

# All models to be loaded
modelPaths = [
	'data/headset.obj',
	'data/floor.obj'
]

# Holds each model dictionary
models = []

for i in range(len(modelPaths)):
	model = Model(modelPaths[i])
	model.normalizeGeometry()

	vertCount = len(model.vertices)
	faceCount = len(model.faces)

	# A dictionary holding key model data
	models.append(dict())
	models[i]['geometry'] = model
	models[i]['vertCount'] = vertCount
	models[i]['faceCount'] = faceCount
	models[i]['transformedVerts'] = [None] * vertCount
	models[i]['transMatrix'] = None

	# A list to store all face normals by index
	models[i]['faceNormals'] = [None] * faceCount

	# Turn all vertices into quaternions
	models[i]['vertQuaternions'] = [None] * vertCount
	for j in range(vertCount):
		models[i]['vertQuaternions'][j] = Quaternion(model.vertices[j])
	
	# A dictionary which maps vertices to each face it is connected to
	models[i]['adjFaces'] = dict()
	for j in range(faceCount):
		# foreach vertex index in the face
		for k in model.faces[j]:
			if not k in models[i]['adjFaces']:
				models[i]['adjFaces'][k] = []

			# Add this face's normal to this vertex index's normal list
			models[i]['adjFaces'][k].append(j)

# Set each model's unique transformation matrix
models[0]['transMatrix'] = TranslationMatrix(Vector(0, 0, -3))
models[1]['transMatrix'] = TranslationMatrix(Vector(0, -2, -3)) * ScaleMatrix(Vector(4, 1, 3))

# Define the light direction
lightDir = Vector(0, -0.5, -1)

orientation = Quaternion(1, 0, 0, 0)
trueUp = Vector(0, 1, 0)

startFrom = 0

alpha = 0.0001

while True:

	print(t)

	# Read rotation data and get rotation as a quaternion
	angVel = data.getRot(t)
	angVel = Vector(*angVel)

	currentTime = data.getTime(t)

	# Apply orientation change
	orientationDelta = Quaternion(angVel.norm() * (currentTime - time), angVel.normalize())
	orientation = orientation * orientationDelta
	orientation = orientation.normalize()

	orientationInv = orientation.inv()

	# Read accelerometer data for tilt correction
	up = data.getAcc(t)
	up = Vector(*up)
	upNorm = up.norm()

	# Only apply drift correction if accelerometer is probably only measuring gravity
	# In a default state, the norm is ~1 so deviations suggest external acceleration
	if upNorm > 0.95 and upNorm < 1.05:

		# Bring local 'up' reading into world space
		up = Quaternion(up)
		up = orientation * up * orientationInv
		up = Vector(up)

		# Apply tilt correction with parameter alpha
		up = up.normalize()
		phi = math.acos(trueUp.dot(up))
		tiltAxis = Vector(up.z, 0, -up.x).normalize()
		driftQuaternion = Quaternion(-phi * alpha, tiltAxis)

		orientation = driftQuaternion * orientation

	time = currentTime

	# Skip rendering until a certain point
	if t < startFrom:
		t += 1
		continue

	# Process each model
	for modelIndex in range(len(models)):

		modelDict = models[modelIndex]
		modelVerts = modelDict['transformedVerts']

		# Apply orientation quarternions and model matrix to vertices
		for vertIndex in range(modelDict['vertCount']):
			if modelIndex == 0:
				modelVerts[vertIndex] = orientation * modelDict['vertQuaternions'][vertIndex] * orientationInv
				modelVerts[vertIndex] = Vector(modelVerts[vertIndex])
			else:
				modelVerts[vertIndex] = Vector(modelDict['vertQuaternions'][vertIndex])

			if modelDict['transMatrix'] is not None:
				modelVerts[vertIndex] = modelDict['transMatrix'] * modelVerts[vertIndex]

		# A set of indicies for faces to cull
		culledFaces = set()

		## CHANGE THIS TO ONLY PRE-CALCULATE NORMALS AND ROTATE THEM HERE INSTEAD OF RECALCULATE THEM
		## POSSIBLY RESTRICT NORMAL CALCULATIONS ONLY FOR THE ROTATING HEADSET AS THE OTHER MODELS HAVE CONSTANT NORMALS
		# Calculate face normals
		for faceIndex in range(modelDict['faceCount']):
			face = modelDict['geometry'].faces[faceIndex]

			# Get the world coordinates for this face's vertices
			p0, p1, p2 = [modelVerts[k] for k in face]

			# Calculate this face's normal
			faceNormal = (p2-p0).cross(p1-p0).normalize()

			# How much light should this face recieve
			faceIntensity = faceNormal * lightDir

			# Intensity < 0 means light is shining through the back of the face
			# In this case, don't draw the face at all ("back-face culling")
			if faceIntensity < 0:
				culledFaces.add(faceIndex)

			modelDict['faceNormals'][faceIndex] = faceNormal

		# Calculate vertex normals and projections
		for vertIndex in range(modelDict['vertCount']):

			# If all the faces this vertex is connected to are to be culled, don't process this vertex
			if all(face in culledFaces for face in modelDict['adjFaces'][vertIndex]):
				continue

			vertNorm = getVertexNormal(vertIndex, modelDict['adjFaces'], modelDict['faceNormals'])

			# How much light does this vertex recieve
			intensity = vertNorm * lightDir
			
			# Though this vertex isn't visible, its shading can affect the visible polygon
			if intensity < 0:
				intensity = 0
			
			# Get this vertex's screen point and colour
			projectedVert = getPerspectiveProjection(modelVerts[vertIndex])
			modelVerts[vertIndex] = Point(projectedVert.x, projectedVert.y, projectedVert.z, Color(intensity*255, intensity*255, intensity*255, 255))

		# Render the image iterating through faces
		for j in range(modelDict['faceCount']):

			# Don't render culled faces
			if j in culledFaces:
				continue

			face = modelDict['geometry'].faces[j]

			Triangle(modelVerts[face[0]], modelVerts[face[1]], modelVerts[face[2]]).draw_faster(image, zBuffer)

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