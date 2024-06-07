import sys
import os
# Agregar la direccion fuera de la carpeta
print(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.abspath('..')
print("dir")
print(parent_dir)
sys.path.append(parent_dir)