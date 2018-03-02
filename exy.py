def example(func_or_list):
    
    if type(func_or_list) == type(list):
        print("Das a list")
    elif callable(func_or_list):
        print("Das a func!")
    else:
        print("Neither a function or a list?")

def print_the_arg(x):
    print("x is ", x);


print("Starting up, the example!")
example(print_the_arg)
