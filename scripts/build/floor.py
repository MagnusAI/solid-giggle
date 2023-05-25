from math import ceil
import os
import sys

if (len(sys.argv) < 2 or (sys.argv[1] == "")):
    distance = 20
else:
    value = float(sys.argv[1])
    distance = ceil((value + 0.1) / 0.05) + 20

filename = os.path.basename(__file__)

xml_elements = ""
for j in range(int(distance)):
    y = (j * 0.05) - 1
    for i in range(40):
        if (i + j) % 2 == 0:
            color = "1 1 1" # white
        else:
            color = "0 0 0" # black
        xml_elements += f'<geom type="box" size=".025 .025 .025" pos="{(i * 0.05) - 0.95} {y} 0" rgba="{color} 1" />\n'

xml_file = f'''
<!-- This file was generated by build_floor.py -->
<!-- It is not recommended to edit this file -->
<mujoco>
    <default>
        <geom density="100" friction=".5" rgba="1 .6 .8 1" />
    </default>
    <worldbody>
        {xml_elements}
    </worldbody>
</mujoco>
'''

# Save the XML file
with open('model/surface/floor.xml', 'w') as f:
    f.write(xml_file)

print(f'Complete. ({filename})')
