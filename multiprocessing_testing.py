import train
import timeit

#log = open("Testing_log", 'w')

# each time is a dict of the format
# {'num_processors': __, 'time': __}
times = []
num_rounds = 25

for num_processors in range(1, 21):
    setup = "import train"
    time = timeit.Timer("train.main({}, {}, 0.93)".format(num_processors, num_rounds), "import train").timeit(1)
    time /= 10
    times.append({'num_processors': num_processors, 'time': time})
    print("{} of 21 done".format(num_processors))
    
print(times)