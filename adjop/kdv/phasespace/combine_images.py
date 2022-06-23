import sys
from PIL import Image

abvec = []
abvec.append((0.2, 0.05))
abvec.append((0.2, 1.0))
abvec.append((0.2, 3.0))
abvec.append((0.05, 0.05))
abvec.append((0.05, 1.0))
abvec.append((0.05, 3.0))

paths = []
# paths.append('objtest_a0p0b1p0.png')
# paths.append('objtest_a0p0b8p0.png')
# paths.append('objtest_a0p25b1p0.png')
# paths.append('objtest_a0p5b1p0.png')

for (a, b) in abvec:
  a_str = str(a).replace('.', 'p')
  b_str = str(b).replace('.', 'p')
  paths.append('objtest_a{}b{}.png'.format(a_str, b_str))

images = [Image.open(x) for x in paths]
widths, heights = zip(*(i.size for i in images))

total_width = round(sum(widths) / 2.0)
max_height = round(sum(heights) / 3.0)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images[:3]:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

x_offset = 0
for im in images[3:]:
  new_im.paste(im, (x_offset,im.size[1]))
  x_offset += im.size[0]

new_im.save('kdv_objectives_T5.png')