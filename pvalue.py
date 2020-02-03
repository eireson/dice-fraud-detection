import numpy as np
#Distribution simulated numerically and briefly checked against theoretical grounds
probabilities=[0.00071875,
  0.00312875,
  0.0077125,
  0.0162825,
  0.02939,
  0.04776625,
  0.07059625,
  0.09390375,
  0.1142025,
  0.12895375,
  0.13270125,
  0.123425,
  0.10065875,
  0.07233,
  0.041995,
  0.016235]

mean = sum([probabilities[i] * (i+3) for i in range(16)])
stderr=np.sqrt(sum([probabilities[i] * (i+3)**2 for i in range(16)])-mean**2)

xvalues=[(i+3 -mean)/(np.sqrt(6) * stderr) for i in range(16)]
xpredict=dict(zip(xvalues,probabilities))

def pvalue(x):
    p=0
    if x<0:
        for k in xpredict:
            if x > k:
                p+=xpredict[k]
    if x>0:
        for k in xpredict:
            if x < k:
                p+=xpredict[k]
    return 1-2*p

def xscore(scores):
    return np.mean([int(item) -mean for item in scores])/( stderr / np.sqrt(6))


print("This script produces a p-value statistic evaluation of the likelihood of a D&D 5e stat block.")
conti=True

while conti == True:
    stats=input("Enter stat block as comma separated values: ").split(',')
    if len(stats)!=6:
        print('Invalid syntax dectected.')
    else:
        print(
        "This stat block has x-score {0}, with p-value {1} i.e. a {2}% degree of confidence".format(
        xscore(stats),pvalue(xscore(stats)),100*round(1-pvalue(xscore(stats)),4)
            )
        )
    next=input('Try again? [y]/n: ')
    if next == 'n':
        conti = False
        print('Thank you, have a great day!')
