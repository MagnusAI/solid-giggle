import random

xml_elements = ""
for j in range(10):
    z = (j * -0.1)+.5
    for i in range(20):
        # Without Color
        gray_value = random.random()
        xml_elements += f'<geom type="box" size=".05 .05 .05" pos="{(i*0.1)-.95} .4 {z}" rgba="1 {gray_value} {gray_value} 1" />\n'
        # With Color
        # r, g, b = random.random(), random.random(), random.random()
        # xml_elements += f'<geom type="box" size=".05 .05 .05" pos="{(i*0.1)-.95} .4 {z}" rgba="{r} {g} {b} 1" />\n'

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
with open('model/wall.xml', 'w') as f:
    f.write(xml_file)
