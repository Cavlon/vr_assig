from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector, Matrix, Quaternion, TranslationMatrix, ScaleMatrix

import cv2
import numpy as np

width = 512
height = 512
image = Image(width, height, Color(200, 200, 200, 255))

near = -1
far = -3

# Init z-buffer
zBuffer = np.full(width * height, -float('inf'))

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

# Converts a point to screen space with perspective
def getPerspectiveProjection(p):
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

# Compute vertex normals by averaging the normals of adjacent faces
def getVertexNormal(vertIndex, adjFaces, faceNormals):
	normal = Vector(0, 0, 0)
	for faceInd in adjFaces[vertIndex]:
		adjNormal = faceNormals[faceInd]
		normal = normal + adjNormal

	return normal / len(adjFaces[vertIndex])

translation = TranslationMatrix(Vector(0, -0.5, -2))
scale = ScaleMatrix(Vector(1, 1, 1))
mat = translation * scale

angle = 0

transformedVerts = [0] * len(model.vertices)

# Define the light direction
lightDir = Vector(0, 0, -1)

faceNormals = [0] * len(model.faces)

adjFaces = {}
for i in range(len(model.faces)):
	# foreach vertex index in the face
	for j in model.faces[i]:
		if not j in adjFaces:
			adjFaces[j] = []

		# Add this face's normal to this vertex index's normal list
		adjFaces[j].append(i)

while True:

	q = Quaternion(angle, Vector(0, 1, 0))
	qi = q.inv()

	angle -= 0.05
	if angle < -2 * np.pi:
		angle += 2 * np.pi

	for i in range(len(transformedVerts)):
		transformedVerts[i] = q * Quaternion(model.vertices[i]) * qi
		transformedVerts[i] = Vector(transformedVerts[i].y, transformedVerts[i].z, transformedVerts[i].w)
		transformedVerts[i] = mat * transformedVerts[i]

	# A set of indicies for faces to cull
	culledFaces = set()

	# Calculate face normals
	for i in range(len(model.faces)):
		face = model.faces[i]
		p0, p1, p2 = [transformedVerts[j] for j in face]
		faceNormal = (p2-p0).cross(p1-p0).normalize()

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

		# Dot product of the vertex normal and light direction
		intensity = vertNorm * lightDir
		
		if intensity < 0:
			intensity = 0
		
		projectedVert = getPerspectiveProjection(transformedVerts[vertIndex])

		transformedVerts[vertIndex] = Point(projectedVert.x, projectedVert.y, projectedVert.z, Color(intensity*255, intensity*255, intensity*255, 255))

	# Render the image iterating through faces
	for i in range(len(model.faces)):

		# Don't render culled faces
		if i in culledFaces:
			continue

		face = model.faces[i]

		Triangle(transformedVerts[face[0]], transformedVerts[face[1]], transformedVerts[face[2]]).draw_faster(image, zBuffer)

	cv2.imshow("render", image.convertToNumpy())

	image.fill(Color(200, 200, 200, 255))
	zBuffer.fill(-float('inf'))

	# Close program if 'Q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()