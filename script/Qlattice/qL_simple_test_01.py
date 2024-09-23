import feyn 
from feyn.datasets import make_regression
import matplotlib.pyplot as plt


ql = feyn.QLattice()

train, test = make_regression()
models = ql.auto_run(train, output_name = 'y')

best = models[0]
best.plot(train, test)
best.plot_regression(test)
plt.show()