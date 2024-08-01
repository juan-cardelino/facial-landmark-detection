'''
Module use in control-dataset-* programs
'''

def intersection(A, B, condition=True):
    '''
    This function takes 2 list (A and B) and a condition. If condition in true, return a list with the elements in A that are in B, in not, returns a list with the elements in A that ar not in B
    
    Args:
        A (list): First set
        
        B (list): Second set
        
        condition (Boolean): conditin that set return, if True: return aux_True, else: return aux_False
        
    Returns:
        aux_True (list): List of elements of A that are also in B
        
        aux_False (list): List of elements of A that are not in B
    '''
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


def takeout_0(a):
    """
    This function takes a list of sting numbers with left 0 (ex: 0001) and return a list of string number without left 0 (ex:1)
    
    Args:
        a (String): String number with left 0
        
    Returns:
        without_0 (String): String number  without left 0
    """
    
    without_0 = []
    for i in a:
        aux = int(i)
        aux = str(aux)
        without_0.append(aux)
    return without_0