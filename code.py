try:
    from sympy import *

    def max_yellow_numbers():
        min_blue = 222
        max_blue = 888
        num_blue = max_blue / 2 + 1
        return num_blue
    
    result = max_yellow_numbers()
    print(int(result))
    
except Exception as e:
    print(e)
    print('FAIL')
