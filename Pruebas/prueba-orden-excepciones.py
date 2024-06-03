# Este codigo prueba el orden en que son llamadas las excepciones al tener subclases de excepciones


# Se genera la clase B, extension de Exception        
class B(Exception):
    pass

# Se genera la clase C, extension de B   
class C(B):
    pass

# Se genera la clase D, extension de C   
class D(C):
    pass

# Se prueban varios errores y se imprime que excepcion lo controla
# Lo ideal es cambiar el orden del llamado de las excepciones para ver si hay cambios
for cls in [C, D, B]:
    try:
        raise cls()
    except D:
        print("D")
    except B:
        print("B")
    except C:
        print("C")