import matplotlib.pyplot as plt
import pandas as pd

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

df = pd.DataFrame({
    'Metrics' : ['Error(Sampled)', 'Error(inferred)'],
    'Mean' : [0.55568, 0.68937],
    'Variance' : [.00007, .00541],
    'Times': [40, 40],
})

plt.table(cellText=df.values, colWidths = [0.25]*len(df.columns),
          colLabels=df.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='top')
plt.show()