from time import sleep
import argparse
# get the environment variables
import os



# get the first argument
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Your name")
parser.add_argument("-i", "--iterations", help="Number of iterations")
args = parser.parse_args()

# get the number of iterations
iterations = int(args.iterations)
env_name = os.environ['NAME']

while True:
    # print the name
    print(env_name)
    # sleep for 1 second
    sleep(1)
    # decrement the number of iterations
    iterations -= 1
    # if the number of iterations is 0, break
    if iterations == 0:
        break

print("Done")
