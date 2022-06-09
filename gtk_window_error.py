from matplotlib import pyplot as plt
import sympy
import numpy as np
import ipdb

random_value=np.random.rand(901)
print(random_value)

# ipdb.set_trace()
plt.plot(np.arange(len(random_value)),random_value)
plt.title('MADDPG: average rewards in ')
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

plt.figure()
patch = plt.imshow()
plt.axis('off')