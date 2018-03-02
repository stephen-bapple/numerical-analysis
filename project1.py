def pt(p):
    '''
    The differential equation used in this project.
    '''
    return 10 * p * (1 - p)
    
    
def ptt(p):
    '''
    The second derivative. 
    For use in Newton's method.
    '''
    return 10 * (1 - 2 * p)
    
    
def pttt(p):
    '''
    The third derivative.
    For use in Newton's method.
    '''
    return -2

    
def p(p):
    '''
    The true solution to the DE used in this project.
    '''
    pass # TODO: Add true solution.