import subprocess

for j in range(20):
    angle = -50 + (j * 5)
    for i in range(40):
        distance = 0.25 + (i * 0.01)

        subprocess.call(['python', f'scripts/build/main_angle.py', f'{distance}', f'{angle}'])

        # Clear results file
        subprocess.run(['rm', f'output/results.txt'])

        # Run the simulation X times
        for i in range(100):
            print(f'Running simulation {i+1}')
            subprocess.run(['python', f'controller/open-loop/main.py'])

            # Copy the results to a new file to avoid overwriting
            filename = f'output/results_distace_90_{round(distance*100, 0)}_{round(angle)}.txt'
            subprocess.run(['cp', f'output/results.txt', filename])

            # Load results file in order to calculate the success rate
            with open(filename, 'r') as f:
                results = f.read().splitlines()

            # Calculate success rate
            success = 0
            for result in results:
                # if the last word on each line is "True}" then it was successful
                if result.split()[-1] == "True":
                    success += 1

            # Add a line to the results file with the success rate
            with open(filename, 'a') as f:
                # get Line count
                line_count = len(open(filename).readlines())
                f.write(f'Success rate: {success/line_count*100}%')
