#!/usr/bin/env python3

## Reference: https://www.pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/

from imutils.object_detection import non_max_suppression
import numpy as np
import glob
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time


def image_template_search(image, template, threshold):
  (tH, tW) = template.shape[:2]

  result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

  # find all locations in the result map where the matched value is
  # greater than the threshold
  (yCoords, xCoords) = np.where(result >= threshold)

  # initialize our list of rectangles
  rects = []
  # loop over the starting (x, y)-coordinates again
  for (x, y) in zip(xCoords, yCoords):
    # update our list of rectangles
    rects.append((x, y, x + tW, y + tH))
  # apply non-maxima suppression to the rectangles
  picks = non_max_suppression(np.array(rects))
  ret = []

  for pick in picks:
    # find the local maximum corresponding to the current pick
    x = pick[0]
    y = pick[1]
    corr = result[y][x]

    # Simple search algorithm: look at all eight neighbors,
    # if point is larger than all neighbors, stop;
    # else, move to the largest point and loop again.
    while True:
      neighbors = []
      for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
          if not (i == 0 and j == 0):
            neighbor_x = x + i
            neighbor_y = y + j
            try:
              neighbor_corr = result[neighbor_y][neighbor_x]
              neighbors.append((neighbor_x, neighbor_y, neighbor_corr))
            except IndexError:
              pass

      neighbors.sort(key=lambda t: t[2])
      if neighbors[-1][2] > corr:
        x = neighbors[-1][0]
        y = neighbors[-1][1]
        corr = neighbors[-1][2]
      else:
        break

    ret.append((x, y, result[y][x]))

  return ret


time_series = []
current_time = 0

invalid = []
not_found = []

template_files = [
  "templates/0.png",
  "templates/1.png",
  "templates/2.png",
  "templates/3.png",
  "templates/4.png",
  "templates/5.png",
  "templates/6.png",
  "templates/7.png",
  "templates/8.png",
  "templates/9.png"
]

# Convert templates to grayscale
templates_color = [cv2.imread(x) for x in template_files]
templates = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in templates_color]


# Correlation threshold. Determined by trial and error.
threshold = 0.8

crops_list = sorted(glob.glob("crops/out_*"))

for image_file in crops_list:
  image_color = cv2.imread(image_file)

  # Convert image to grayscale
  image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

  matches = []

  # Correlate each digit and store all local maxima found.
  for digit in range(10):
    template = templates[digit]
    picks = image_template_search(image, template, threshold)

    for pick in picks:
      # We don't care about the y-coordinate.
      # Only store the digit, x-coordinate, and correlation value.
      matches.append((digit, pick[0], pick[2]))

  # Sort by x-coordinate, i.e. from left to right
  matches.sort(key=lambda t: t[1])


  # Multiple templates can be matched by the same digit. For example,
  # "8" may be matched by both "6" and "8". Here, we group matches
  # according to x-coordinate, so that if they are close together
  # (within 15, determined experimentally) we consider them to
  # refer to the same digit, and we choose the option that gives
  # the highest correlation value.
  matched_digits = []
  current_digit = []

  for match in matches:
    if len(current_digit) == 0:
      current_digit.append(match)
    else:
      if match[1] - current_digit[0][1] <= 15:
        current_digit.append(match)
      else:
        matched_digits.append(current_digit)
        current_digit = [match]

  if current_digit:
    matched_digits.append(current_digit)

  matches = []
  for digit in matched_digits:
    digit.sort(key=lambda t: t[2])

    # Do not allow the number to start with zero
    if len(matches) == 0 and digit[-1][0] == 0:
      del digit[-1]
    matches.append(digit[-1])

  # Construct the number
  numstr = "".join(str(t[0]) for t in matches)

  if numstr:
    num = int(numstr)
    # Validation: if it is out of this range, probably something
    # went wrong with the image detection (e.g. extra digit, missing
    # digit). Ignore the data point.
    if num > 150 or num < 60:
      num = None
      invalid.append(image_file)
  else:
    # Failed to find the number in the image.
    num = None
    not_found.append(image_file)

  if num:
    time_series.append((current_time, num))

  current_time += 1

if invalid:
  print("Invalid:")
  print("\n".join(["    " + x for x in invalid]))
if not_found:
  print("Not found:")
  print("\n".join(["    " + x for x in not_found]))

t = [x[0] for x in time_series if x[1]]
pulse = [x[1] for x in time_series if x[1]]

fig, ax = plt.subplots()

plt.plot(t, pulse)
plt.xlabel("Time since stream starts (hh:mm)")
plt.ylabel("Heart rate (bpm)")

# https://stackoverflow.com/a/40397028
# https://stackoverflow.com/a/19972993
formatter = matplotlib.ticker.FuncFormatter(lambda sec, x: time.strftime("%H:%M", time.gmtime(sec)))
loc = matplotlib.ticker.MultipleLocator(base=3600)

ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(loc)

plt.show()
