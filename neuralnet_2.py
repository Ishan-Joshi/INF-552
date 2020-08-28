import numpy as np
def input_pgm():
    with open('downgesture_train.list', 'rb') as pgmf:
        P5 = pgmf.readline()
        comment = pgmf.readline()
        width = [int(value) for value in pgmf.readline()]
        print(width)
    return None
print(input_pgm())
