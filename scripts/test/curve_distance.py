import subprocess
import shutil
import os

for j in range(1):
    curve = 1 + (j * 0.2)
    for i in range(8):
        distance = 0.15 + (i * 0.05)

        subprocess.call(['python', f'scripts/build/main_curve.py', f'{distance}', f'{curve}'])

        # Clear results file
        if os.path.exists('output/results.txt'):
            os.remove('output/results.txt')

        # Run the simulation X times
        for i in range(10):
            print(f'Running simulation {i+1}')
            subprocess.run(['python', f'controllers/rule/main.py'])

            # Copy the results to a new file to avoid overwriting
            filename = f'output/rule/curve/ruled_based_curve_{round(curve, 2)}_{round(distance*100, 0)}.txt'
            if not os.path.isfile('output/results.txt'):
                open('output/results.txt', 'w').close()

            shutil.copy('output/results.txt', filename)

            # Load results file in order to calculate the success rate
            with open(filename, 'r') as f:
                results = f.read().splitlines()
