#!/usr/bin/python
import numpy as np


def FourSixSidedDiceDropLowest():
    randnums= np.random.randint(1,7,4)
    result=sum(np.delete(np.sort(randnums),0))
    return result
numbers=np.empty([])

for _ in range(0,10):
    a=FourSixSidedDiceDropLowest()
    numbers=np.append(numbers, a)

numbers=numbers.astype(int)

np.savetxt('rolls.out', numbers, delimiter=',',fmt='%u')
