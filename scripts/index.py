# Run all scripts in the scripts directory
import subprocess

scripts = [
    'build_floor.py',
    'build_wall_90.py',
]

print("Building XML files...\n")
for script in scripts:
    print(f'Running {script}')
    subprocess.run(['python', f'scripts/{script}'])
