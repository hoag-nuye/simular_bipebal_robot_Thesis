import os
import multiprocessing
# Số lõi logic (logical cores)
logical_cores = os.cpu_count()
print(f"Số logical cores: {logical_cores}")

# Số lõi vật lý (physical cores)
physical_cores = multiprocessing.cpu_count()
print(f"Số physical cores: {physical_cores}")

print(multiprocessing)
