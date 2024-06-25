def filtro(a, b, condition=True):
    # List A in B
    aux_True = []
    # List A not in B
    aux_False = []
    for i in a:
        # Check if A is in B
        if i in b:
            aux_True.append(i)
        else:
            aux_False.append(i)
    # Check return condition
    if condition:
        # Return A in B
        return aux_True
    else:
        # Return A not in B
        return aux_False

# Takes '0' from a string number
def reducir_0(a):
    filtrada = []
    for i in a:
        aux = i
        while aux[0] == '0':
            aux = aux[1:]
        filtrada.append(aux)
    return filtrada