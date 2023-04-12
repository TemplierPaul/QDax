import subprocess
import os
from time import time

# get cli args without parsing
import sys
N = int(sys.argv[1])
cmd = sys.argv[2:]

print("cmd", cmd)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(1.0 / N)

processes = {}
t0 = time()

try:
    for i in range(N):
        print(f"Launching {i} - mem {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
        to_run =['python'] + cmd + ["--seed", str(i)]
        print(" ".join(to_run))
        p = subprocess.Popen(to_run)
        processes[i] = p

    for id, p in processes.items():
        print(f"Waiting for {id}")
        p.wait()
except KeyboardInterrupt:
    for id, p in processes.items():
        print(f"Killing {id}")
        p.kill()

print("Done")