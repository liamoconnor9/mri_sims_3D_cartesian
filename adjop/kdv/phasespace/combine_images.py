import sys
from PIL import Image

abvec = []
abvec.append((5.0,    0.03))
abvec.append((1.5,    0.1 ))
abvec.append((1.0,    0.15 ))
abvec.append((0.5,    0.3))
abvec.append((0.15,   1.0))
abvec.append((0.05,   3.0))

paths = []

for (a, T) in abvec:
  a_str = str(a).replace('.', 'p')
  T_str = str(T).replace('.', 'p')
  paths.append('objtest_a{}T{}.png'.format(a_str, T_str))

images = [Image.open(x) for x in paths]
widths, heights = zip(*(i.size for i in images))

rows, cols = 2, 3

total_width = round(sum(widths) / rows)
max_height = round(sum(heights) / cols)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images[:3]:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

x_offset = 0
for im in images[3:]:
  new_im.paste(im, (x_offset,im.size[1]))
  x_offset += im.size[0]

new_im.save('kdv_objectives_nuT0p15.png')