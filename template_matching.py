import cv2
import numpy as np
from matplotlib import pyplot as plt
from common import* 

waldo_path = 'data/finding_waldo/puzzle1/waldo.jpg'
puzzle_path = 'data/finding_waldo/puzzle1/pic1.jpeg'

waldo = cv2.imread(waldo_path)
puzzle = cv2.imread(puzzle_path)

result = ssd_matching(waldo, puzzle)

(waldoHeight, waldoWidth) = waldo.shape[:2]
# result = cv2.matchTemplate(puzzle, waldo, cv2.TM_SQDIFF)
topLeft = np.where(result == np.amin(result))
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

# grab the bounding box of waldo and extract him from the puzzle image
topLeft = minLoc
botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = puzzle[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the map except for Waldo
mask = np.zeros(puzzle.shape, dtype = "uint8")
map = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

map[topLeft[1] : botRight[1], topLeft[0] : botRight[0]] = roi

# display the images
result_rgb = cv2.cvtColor(map, cv2.COLOR_RGB2BGR)
plt.figure(figsize = (15, 15))
plt.imshow(result_rgb)
plt.axis('off')
plt.savefig('output/finding_waldo/puzzle2/foundP1.png')
plt.show()