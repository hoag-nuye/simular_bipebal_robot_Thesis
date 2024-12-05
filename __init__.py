import os
import multiprocessing
from interface.plot_param import plot_param_process
# Số lõi logic (logical cores)
logical_cores = os.cpu_count()
print(f"Số logical cores: {logical_cores}")

# Số lõi vật lý (physical cores)
physical_cores = multiprocessing.cpu_count()
print(f"Số physical cores: {physical_cores}")

print(multiprocessing)
plot_param_process("models/param/")

