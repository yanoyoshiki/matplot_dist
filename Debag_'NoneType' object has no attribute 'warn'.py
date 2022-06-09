import numpy as np
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt

Figure = plt.figure()

lines_plotted = plt.plot([])

line_plotted = lines_plotted[0]

plt.xlim(0, 2 * np.pi)

plt.ylim(-1.1, 1.1)

x = np.linspace(0, 2 * np.pi, 100)

y = 0


def AnimationFunction(frame):
    y = np.cos(x + 2 * np.pi * frame / 100)
    line_plotted.set_data((x, y))


anim_created = FuncAnimation(Figure, AnimationFunction, frames=100, interval=25)
video = anim_created.to_html5_video()
html = display.HTML(video)
display.display(html)
plt.close()