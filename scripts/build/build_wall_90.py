import random
import os
import sys


if (len(sys.argv) < 2 or (sys.argv[1] == "")):
    distance = 0.8
else:
    distance = float(sys.argv[1])

filename = os.path.basename(__file__)

xml_elements = ""
for j in range(10):
    z = (j * -0.05)+.5
    for i in range(40):
        # Without Color
        gray_value = random.random()
        xml_elements += f'<geom type="box" size=".025 .025 .025" pos="{(i*0.05)-.95} {distance} {z}" rgba="1 {gray_value} {gray_value} 1" />\n'
        # With Color
        # r, g, b = random.random(), random.random(), random.random()
        # xml_elements += f'<geom type="box" size=".025 .025 .025" pos="{(i*0.1)-.95} .4 {z}" rgba="{r} {g} {b} 1" />\n'

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
with open('model/walls/wall_90_40.xml', 'w') as f:
    f.write(xml_file)

print(f'Complete. ({filename}))')
