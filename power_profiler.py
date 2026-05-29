import threading
import time
import psutil
import pynvml
import torch
import numpy as np

# --- Monitoring Functions ---

def monitor_system(process, results, stop_event, interval=0.1):
    """Monitors CPU and RAM usage for a given process."""
    while not stop_event.is_set():
        try:
            cpu_percent = process.cpu_percent(interval=interval)
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            results.append({'type': 'system', 'timestamp': time.time(), 'cpu_%': cpu_percent, 'ram_mb': rss_mb})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

def monitor_gpu(handle, gpu_index, results, stop_event, interval=0.1):
    """Monitors a specific GPU's utilization, VRAM, and power."""
    while not stop_event.is_set():
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            results.append({
                'type': 'gpu',
                'timestamp': time.time(),
                'gpu_index': gpu_index, # <<< NEW: Tag data with GPU index
                'gpu_util_%': util.gpu,
                'vram_used_mb': mem_info.used / (1024 * 1024),
                'power_w': power_watts
            })
        except pynvml.NVMLError:
            break
        time.sleep(interval)

def monitor_cpu_power_estimate(results, stop_event, cpu_tdp, interval=0.1):
    """Monitors and estimates CPU power consumption based on utilization and TDP."""
    while not stop_event.is_set():
        total_cpu_util = psutil.cpu_percent(interval=interval)
        estimated_power_w = cpu_tdp * (total_cpu_util / 100.0)
        results.append({
            'type': 'cpu_power',
            'timestamp': time.time(),
            'cpu_util_total_%': total_cpu_util,
            'estimated_power_w': estimated_power_w
        })

# --- Profiler Class for External Control ---

class Profiler:
    """A class to start and stop hardware resource monitoring for multiple GPUs."""
    def __init__(self, poll_interval=0.1, cpu_tdp=125):
        self.poll_interval = poll_interval
        self.cpu_tdp = cpu_tdp
        self.results = []
        self._stop_event = threading.Event()
        self._threads = []
        self._process = None
        self._gpu_handles = [] # <<< MODIFIED: Store a list of handles
        self.is_active = False
        # print(f"Profiler initialized. Using estimated CPU TDP of {self.cpu_tdp}W.")

    def start(self):
        """Starts the monitoring threads."""
        if self.is_active:
            print("Profiler is already active.")
            return

        self.results = []
        self._stop_event.clear()
        self._threads = []
        self._gpu_handles = []
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                print("Warning: No NVIDIA GPUs found. GPU monitoring disabled.")
            else:
                # print(f"Found {device_count} NVIDIA GPU(s). Starting monitor for each.")
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self._gpu_handles.append(handle)
                    gpu_thread = threading.Thread(
                        target=monitor_gpu,
                        args=(handle, i, self.results, self._stop_event, self.poll_interval)
                    )
                    self._threads.append(gpu_thread)
                    gpu_thread.start()
        except pynvml.NVMLError as e:
            print(f"Warning: Could not initialize NVML. GPU monitoring disabled. Error: {e}")

        self._process = psutil.Process()
        
        sys_thread = threading.Thread(
            target=monitor_system,
            args=(self._process, self.results, self._stop_event, self.poll_interval)
        )
        self._threads.append(sys_thread)
        sys_thread.start()
        
        cpu_power_thread = threading.Thread(
            target=monitor_cpu_power_estimate,
            args=(self.results, self._stop_event, self.cpu_tdp, self.poll_interval)
        )
        self._threads.append(cpu_power_thread)
        cpu_power_thread.start()

        self.is_active = True

    def stop(self):
        """Stops the monitoring threads and returns the collected data."""
        if not self.is_active:
            print("Profiler is not active.")
            return []

        self._stop_event.set()
        for t in self._threads:
            t.join()
        
        if self._gpu_handles:
            pynvml.nvmlShutdown()

        self.is_active = False
        return self.results

# --- Processing Function ---

def calculate_energy(power_data):
    """Calculates total energy in Joules from a list of power readings."""
    if not power_data or len(power_data) < 2:
        return 0
    total_energy_joules = 0
    for i in range(len(power_data) - 1):
        if 'estimated_power_w' in power_data[i]:
            power_w = power_data[i]['estimated_power_w']
        elif 'power_w' in power_data[i]:
            power_w = power_data[i]['power_w']
        else:
            continue
        dt_s = power_data[i+1]['timestamp'] - power_data[i]['timestamp']
        total_energy_joules += power_w * dt_s
    return total_energy_joules

def process_and_print_results(results, duration_s):
    """Processes and prints the results, including multi-GPU energy calculations."""
    if not results:
        print("No results to process.")
        return

    print("\n--- Profiling Results ---")
    results.sort(key=lambda r: r['timestamp'])
    
    sys_data = [r for r in results if r['type'] == 'system']
    gpu_data = [r for r in results if r['type'] == 'gpu']
    cpu_power_data = [r for r in results if r['type'] == 'cpu_power']

    if sys_data:
        max_ram = max(r['ram_mb'] for r in sys_data)
        avg_cpu = sum(r['cpu_%'] for r in sys_data) / len(sys_data) if sys_data else 0
        print(f"\nSystem Usage (per-process):")
        print(f"  - Max RAM Usage: {max_ram:.2f} MB")
        print(f"  - Average Process CPU Usage: {avg_cpu:.2f} %")

    # --- GPU Power and Energy (per-GPU and total) ---
    all_gpus_total_energy_j = 0
    if gpu_data:
        # Find out how many unique GPUs were monitored
        gpu_indices = sorted(list(set(r['gpu_index'] for r in gpu_data)))
        
        for index in gpu_indices:
            single_gpu_data = [r for r in gpu_data if r['gpu_index'] == index]
            if not single_gpu_data: continue

            avg_power = sum(r['power_w'] for r in single_gpu_data) / len(single_gpu_data)
            max_vram = max(r['vram_used_mb'] for r in single_gpu_data)
            avg_gpu_util = sum(r['gpu_util_%'] for r in single_gpu_data) / len(single_gpu_data)
            total_gpu_energy_j = calculate_energy(single_gpu_data)
            all_gpus_total_energy_j += total_gpu_energy_j
            
            print(f"\n--- GPU {index} Usage ---")
            print(f"  - Average Power Draw: {avg_power:.2f} W")
            print(f"  - Average Utilization: {avg_gpu_util:.2f} %")
            print(f"  - Max VRAM Usage: {max_vram:.2f} MB")
            print(f"  - Total Energy Consumed: {total_gpu_energy_j:.2f} Joules ({total_gpu_energy_j / 3600:.6f} Wh)")

        if len(gpu_indices) > 1:
            print(f"\n--- Total All GPUs ---")
            print(f"  - Total Energy Consumed: {all_gpus_total_energy_j:.2f} Joules ({all_gpus_total_energy_j / 3600:.6f} Wh)")

    total_cpu_energy_j = 0
    if cpu_power_data:
        avg_power = sum(r['estimated_power_w'] for r in cpu_power_data) / len(cpu_power_data)
        total_cpu_energy_j = calculate_energy(cpu_power_data)
        print(f"\nCPU Usage (Estimated):")
        print(f"  - Average Estimated Power Draw: {avg_power:.2f} W")
        print(f"  - Total Estimated Energy Consumed: {total_cpu_energy_j:.2f} Joules ({total_cpu_energy_j / 3600:.6f} Wh)")
    
    total_system_energy_j = all_gpus_total_energy_j + total_cpu_energy_j
    if total_system_energy_j > 0:
        avg_system_power = total_system_energy_j / duration_s
        print(f"\n--- Total Estimated System Consumption (CPU + All GPUs) ---")
        print(f"  - Average Combined Power: {avg_system_power:.2f} W")
        print(f"  - Total Energy: {total_system_energy_j:.2f} Joules")
        print(f"  - Total Energy: {total_system_energy_j / 3600:.6f} Wh")
        print(f"  - Total Energy: {total_system_energy_j / 3600 / 1000:.9f} kWh")


# --- Demo Function ---

def my_multi_gpu_function():
    """A sample function that uses CPU and potentially multiple GPUs."""
    print("Starting heavy computation...")
    # CPU-intensive task
    cpu_tensor = np.random.rand(8000, 8000)
    cpu_tensor = cpu_tensor @ cpu_tensor.T
    
    # GPU-intensive tasks
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} CUDA devices. Spreading work...")
        
        # Run a task on each GPU
        for i in range(num_gpus):
            device = f'cuda:{i}'
            print(f"Running task on {device}...")
            gpu_tensor = torch.randn(12000, 12000, device=device)
            gpu_tensor = gpu_tensor @ gpu_tensor.T
        
        # Simulate some sustained work
        time.sleep(2)

    else:
        print("CUDA not available, skipping GPU task.")
    
    print("Computation finished.")
    return True

# --- Usage Example ---

if __name__ == "__main__":
    # IMPORTANT: Change the cpu_tdp value to match your CPU.
    profiler = Profiler(cpu_tdp=385)

    profiler.start()

    start_time = time.time()
    my_multi_gpu_function()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nTotal Execution Time: {duration:.2f} seconds")

    results = profiler.stop()

    process_and_print_results(results, duration)