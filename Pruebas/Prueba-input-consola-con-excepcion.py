# Este codigo prueba extraer un numero de consola
# Primero intanta leer de consola, si el input es numerico termina, sino salta un error que se maneja reiniciando la ejecucion 


while True:
    # Se intenta leer un numero de consola
    try:
        x = int(input("Please enter a number: "))
        print("Your number is :", x)
        break
    # si se lee algo que no es un numero se maneja con una excepcion
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")