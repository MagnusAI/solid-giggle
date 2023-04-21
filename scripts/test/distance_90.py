import subprocess

for i in range(30):
    distance = 0.25 + (i * 0.01)

    subprocess.call(['python', f'scripts/build/index.py', f'{distance}'])

    # Clear results file
    subprocess.run(['rm', f'output/results.txt'])

    # Run the simulation 10 times
    for i in range(5):
        print(f'Running simulation {i+1}')
        subprocess.run(['python', f'simulate/main.py'])

        # Copy the results to a new file to avoid overwriting
        filename = f'output/results_distace_90_{round(distance*100, 0)}.txt'
        subprocess.run(['cp', f'output/results.txt', filename])

        # Load results file in order to calculate the success rate
        with open(filename, 'r') as f:
            results = f.read().splitlines()

        # Calculate success rate
        success = 0
        for result in results:
            # if the last word on each line is "True}" then it was successful
            if result.split()[-1] == "True}":
                success += 1

        # Add a line to the results file with the success rate
        with open(filename, 'a') as f:
            # get Line count
            line_count = len(open(filename).readlines())
            f.write(f'Success rate: {success/line_count*100}%')
