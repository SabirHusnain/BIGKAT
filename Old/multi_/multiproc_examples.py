# -*- coding: utf-8 -*-
"""

"""

import multiprocessing
import os
import time


def func(jobs, results):
    while True:
        j = jobs.get()
        if j == "KILL":
            break
        else:

            # Do something
            #            results.put("Done")
            pass
    #            print(j)

    return  # End function


def main():
    fake_jobs = list(range(1000))

    max_jobs = 1000  # Maximum number of items that can go in queue (may have to be small on RPi)
    jobs = multiprocessing.Queue(max_jobs)
    results = multiprocessing.Queue()  # Queue to place the results into

    n_workers = os.cpu_count()
    workers = []

    for i in range(n_workers):
        print("Starting Worker {}".format(i))
        tmp = multiprocessing.Process(target=func, args=(jobs, results))
        tmp.start()
        workers.append(tmp)

    print("There are {} workers".format(len(workers)))

    while len(fake_jobs) > 0:

        if not jobs.full():
            j = fake_jobs.pop()
            #            print("Job: {}".format(j))
            jobs.put(j)  # Put a job in the Queue

    ##Tell all workers to Die
    for worker in workers:
        jobs.put("KILL")

    # Wait for workers to Die
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("That took: {} seconds".format(t1 - t0))

# for worker in workers:
#        worker.join() 
# super_job = list(range(1000))
#
# while True:
#    
#    if not jobs.full():
#        jobs.put(super_job.pop())
