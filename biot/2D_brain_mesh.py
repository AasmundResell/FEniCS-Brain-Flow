from mshr import *
from dolfin import *
from matplotlib.pyplot import show


def generate_2D_brain_mesh():

    # Get as much output as possible by setting debug type log level.

    origin = Point(0.0, 0.0)
    
    channelCoord1 = Point(-0.25,0.0)
    channelCoord2 = Point(0.25,-5.0)
    
    r1 = 5.0  # Outer radius (mm)
    r2 = 1.0  # Inner radius  (mm)

    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    aqueduct = Rectangle(channelCoord1,channelCoord2)

    geometry = parenchyma - ventricles - aqueduct
    dolfin.info(geometry, True)

    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 10
    mesh = generate_mesh(geometry, N)

    plot(mesh, title="mesh")
    show()
    
generate_2D_brain_mesh()
