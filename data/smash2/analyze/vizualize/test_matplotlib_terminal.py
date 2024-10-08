from matplotlib_terminal import plt

# Create a figure and axis
fig, ax = plt.subplots()

# Plot data on the axis
ax.plot([0, 1], [0, 1])
ax.plot([1, 0], [0, 1], lw=3)
ax.scatter([0], [.5])

# Show the plot using different renderers
render(fig, renderer='img2unicode', mode='fast/noblock')
render(fig, renderer='img2unicode', mode='fast/block')
render(fig, renderer='img2unicode', mode='fast/braille')

# Close the plot
plt.close(fig)