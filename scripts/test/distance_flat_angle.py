import subprocess
import shutil
import os

for j in range(20):
    angle = -50 + (j * 5)
    for i in range(40):
        distance = 0.15 + (i * 0.01)

        subprocess.call(['python', f'scripts/build/main_angle.py', f'{distance}', f'{angle}'])

        # Clear results file
        if os.path.exists('output/results.txt'):
            os.remove('output/results.txt')

        # Run the simulation X times
        for i in range(10):
            print(f'Running simulation {i+1}')
            subprocess.run(['python', f'controllers/open-loop/main.py'])

            # Copy the results to a new file to avoid overwriting
            filename = f'output/results_distace_{round(distance*100, 0)}_{round(angle)}.txt'
            shutil.copy('output/results.txt', filename)

            # Load results file in order to calculate the success rate
            with open(filename, 'r') as f:
                results = f.read().splitlines()

            # Calculate success rate
            success = 0
            for result in results:
                # if the last word on each line is "True}" then it was successful
                if result.split()[-1] == "True'}":
                    success += 1

            # Add a line to the results file with the success rate
            with open(filename, 'a') as f:
                # get Line count
                line_count = len(open(filename).readlines())
                f.write(f'Success rate: {success/line_count*100}%')
