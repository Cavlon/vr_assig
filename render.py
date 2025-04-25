from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector, Matrix, Quaternion, TranslationMatrix, ScaleMatrix
from track import Data

import cv2
import numpy as np
import math
import copy

width = 512
height = 512
image = Image(width, height, Color(200, 200, 200, 255))

data = Data("IMUData.csv")
data_size = data.data.shape[0] - 1

near = -1
far = -6

# Define the light direction
lightDir = Vector(0, -0.5, -1).normalize()

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

def handleOrientation(orientation):
	""" Process rotation data and update orientation with tilt correction """
	global time

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

	return orientation, orientationInv

def physicsPass(models, positions, velocities, friction = 0.98, radius = 0.5):
	new_velocities = copy.deepcopy(velocities)

	for i in range(0, len(models)-2):
		# Update the model matrix to visually move the object to the position
		models[i+2]['transMatrix'] = TranslationMatrix(positions[i]) * ScaleMatrix(Vector(0.75, 0.75, 0.75))

		for j in range(i+1, len(models)-2):
			# Vector between objects collision occurs along
			collision_normal = positions[j] - positions[i]
			dist = collision_normal.norm()

			# Bounding spheres overlap therefore collision
			if dist < 2 * radius:
				collision_normal = collision_normal / dist

				# Velocity of this object relative to the other object along the collision normal
				rel_vel = (velocities[i] - velocities[j]) * collision_normal

				# Apply collision impulse on both objects along the collision normal
				new_velocities[i] = new_velocities[i] - collision_normal * rel_vel
				new_velocities[j] = new_velocities[j] + collision_normal * rel_vel

	# Update velocities and positions separately so collision velocities and positions are consistent
	for i in range(0, len(models)-2):
		velocities[i] = new_velocities[i] * friction
		positions[i] = positions[i] + new_velocities[i]


# All models to be loaded
modelPaths = [
	'data/headset.obj',
	'data/floor.obj',
	'data/headset.obj',
	'data/headset.obj',
	'data/headset.obj',
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
	models[i]['culledFaces'] = set()

	# A list to store all face normals by index
	models[i]['faceNormals'] = [None] * faceCount
	for faceIndex in range(faceCount):
		face = model.faces[faceIndex]

		# Get the world coordinates for this face's vertices
		p0, p1, p2 = [model.vertices[k] for k in face]

		# Calculate this face's normal
		faceNormal = (p2-p0).cross(p1-p0).normalize()

		# How much light should this face recieve
		faceIntensity = faceNormal * lightDir

		# Intensity < 0 means light is shining through the back of the face
		# In this case, don't draw the face at all ("back-face culling")
		if faceIntensity < 0:
			models[i]['culledFaces'].add(faceIndex)

		models[i]['faceNormals'][faceIndex] = faceNormal

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
	
	# A list to store all vertex intensities by index
	models[i]['vertIntensities'] = [None] * vertCount
	for vertIndex in range(vertCount):

		# If all the faces this vertex is connected to are to be culled, don't process this vertex
		if all(face in models[i]['culledFaces'] for face in models[i]['adjFaces'][vertIndex]):
			continue

		vertNorm = getVertexNormal(vertIndex, models[i]['adjFaces'], models[i]['faceNormals'])

		# How much light does this vertex recieve
		intensity = vertNorm * lightDir
		
		# Though this vertex isn't visible, its shading can affect the visible polygon
		if intensity < 0:
			intensity = 0
		
		models[i]['vertIntensities'][vertIndex] = intensity

# Set each model's unique transformation matrix
models[0]['transMatrix'] = TranslationMatrix(Vector(0, 0, -2))
models[1]['transMatrix'] = TranslationMatrix(Vector(0, -2, -3)) * ScaleMatrix(Vector(4, 1, 3))

velocities = [Vector(-0.1, 0, 0.03), Vector(0.05, 0, 0), Vector(0.08, 0, 0)]
positions = [Vector(2, -2, -4), Vector(-2, -2, -3), Vector(-3, -2, -4)]

orientation = Quaternion(1, 0, 0, 0)
trueUp = Vector(0, 1, 0)

startFrom = 0

alpha = 0.0001

while True:

	print(t)

	orientation, orientationInv = handleOrientation(orientation)
	physicsPass(models, positions, velocities)

	# Skip rendering until a certain point
	if t < startFrom:
		t += 1
		continue

	# Process each model
	for modelIndex in range(len(models)):

		modelDict = models[modelIndex]
		modelVerts = modelDict['transformedVerts']

		# Only recalculate normals for the rotating objects
		if modelIndex == 0:
			# Apply orientation quarternions and model matrix to vertices
			for vertIndex in range(modelDict['vertCount']):
				modelVerts[vertIndex] = orientation * modelDict['vertQuaternions'][vertIndex] * orientationInv
				modelVerts[vertIndex] = Vector(modelVerts[vertIndex])

				if modelDict['transMatrix'] is not None:
					modelVerts[vertIndex] = modelDict['transMatrix'] * modelVerts[vertIndex]

			modelDict['culledFaces'] = set()

			# Recalculate face normals
			faceNormals = [None] * modelDict['faceCount']
			for faceIndex in range(modelDict['faceCount']):

				# Rotate normals to match object orientation
				faceNormals[faceIndex] = orientation * Quaternion(modelDict['faceNormals'][faceIndex]) * orientationInv
				faceNormals[faceIndex] = Vector(faceNormals[faceIndex]).normalize()

				# How much light should this face recieve
				faceIntensity = faceNormals[faceIndex] * lightDir

				# Intensity < 0 means light is shining through the back of the face
				# In this case, don't draw the face at all ("back-face culling")
				if faceIntensity < 0:
					modelDict['culledFaces'].add(faceIndex)
			
			# Calculate vertex normals and projections
			for vertIndex in range(modelDict['vertCount']):

				# If all the faces this vertex is connected to are to be culled, don't process this vertex
				if all(face in modelDict['culledFaces'] for face in modelDict['adjFaces'][vertIndex]):
					continue

				vertNorm = getVertexNormal(vertIndex, modelDict['adjFaces'], faceNormals).normalize()

				# How much light does this vertex recieve
				intensity = vertNorm * lightDir
				
				# Though this vertex isn't visible, its shading can affect the visible polygon
				if intensity < 0:
					intensity = 0
				
				projectedVert = getPerspectiveProjection(modelVerts[vertIndex])
				modelVerts[vertIndex] = Point(projectedVert.x, projectedVert.y, projectedVert.z, Color(intensity*255, intensity*255, intensity*255, 255))
		else:
			# Apply orientation quarternions and model matrix to vertices
			for vertIndex in range(modelDict['vertCount']):
				if all(face in modelDict['culledFaces'] for face in modelDict['adjFaces'][vertIndex]):
					continue

				modelVerts[vertIndex] = Vector(modelDict['vertQuaternions'][vertIndex])

				if modelDict['transMatrix'] is not None:
					modelVerts[vertIndex] = modelDict['transMatrix'] * modelVerts[vertIndex]

				# Get this vertex's screen point and colour
				intensity = modelDict['vertIntensities'][vertIndex]

				projectedVert = getPerspectiveProjection(modelVerts[vertIndex])
				modelVerts[vertIndex] = Point(projectedVert.x, projectedVert.y, projectedVert.z, Color(intensity*255, intensity*255, intensity*255, 255))

		# Render the image iterating through faces
		for j in range(modelDict['faceCount']):

			# Don't render culled faces
			if j in modelDict['culledFaces']:
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