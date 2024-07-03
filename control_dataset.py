def intersection(A, B, condition=True):
    # List A in B
    aux_True = []
    # List A not in B
    aux_False = []
    for i in A:
        # Check if A is in B
        if i in B:
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
def takeout_0(a):
    without_0 = []
    for i in a:
        aux = int(i)
        aux = str(aux)
        without_0.append(aux)
    return without_0