import random

xml_elements = ""
for j in range(10):
    y = (j * 0.1)-.5
    for i in range(20):
        r, g, b = random.random(), random.random(), random.random()
        xml_elements += f'<geom type="box" size=".05 .05 .05" pos="{(i*0.1)-.95} {y} -.5" rgba="{r} {g} {b} 1" />\n'

xml_file = f'''
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
with open('model/floor.xml', 'w') as f:
    f.write(xml_file)
