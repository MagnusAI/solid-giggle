import time

for i in range(10):
    print("Countdown: ", 10 - i, end="\r")
    time.sleep(1)

print("Blast off!")