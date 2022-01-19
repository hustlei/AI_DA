import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.subplots()

plt.rcParams["text.usetex"]=True
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"y=$\alpha^2$")

eq=(r"\begin{eqnarray*}"
    r"a=b^2+c^2 \\"
    r"\Delta=\sqrt{a^2+c^2} \\"
    r"\end{eqnarray*}")
ax.text(0.5,0.5,eq)

plt.show()