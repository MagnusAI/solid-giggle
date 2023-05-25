# Run all scripts in the scripts directory
import subprocess
import sys

if len(sys.argv) < 3:
    distance = 0.4
    angle = 90
else:
    distance = float(sys.argv[1])
    angle = float(sys.argv[2])

# if there is no argument, use the default
if (distance == ""):
    distance = 0.4
    angle = 90

scripts = [
    'floor.py',
    'angle_wall.py',
]

print("Building XML files...\n")
for script in scripts:
    print(f'Running {script}')
    # run the script and pass the distance argument
    subprocess.call(['python', f'scripts/build/{script}', f'{distance}', f'{angle}'])
