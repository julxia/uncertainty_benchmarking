#!/usr/bin/env python3

"""
GPU Monitoring and Task Runner for Inference Benchmarking

This script monitors GPU usage while running inference benchmark tests on different models.
It outputs power metrics and logs execution output for analysis.
"""
import argparse
import subprocess
import sys
import os
import signal
import time
import threading
import shlex
from datetime import datetime

class GPUMonitor:
    """Monitors GPU metrics during benchmarking."""
    def __init__(self, i, server_id, model_name, interval=1):
        self.i = i
        self.output_file = None
        self.interval = interval
        self.process = None
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.server_id = server_id
        self.model_name = model_name

    def start(self):
        """Start the GPU monitoring process and saves metrics to CSV"""
        os.makedirs(f"{self.server_id}_monitor",exist_ok=True)

        try:
            self.output_file = f"{self.server_id}_monitor/{self.i}_{self.model_name}.csv"
            self._ensure_unique_filename(self.output_file)
            
            # nvidia-smi for power metric logging
            self.process = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=index,timestamp,power.draw,power.min_limit,power.max_limit',
                 '--format=csv,nounits', '-l', str(self.interval)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            time.sleep(0.2)
            if self.process.poll() is not None:
                # Process terminated quickly, likely an error
                _, stderr_data = self.process.communicate()
                self._stderr_output = stderr_data
                raise RuntimeError(f"nvidia-smi process terminated unexpectedly. Exit code: {self.process.returncode}. Stderr: {self._stderr_output.strip()}")
            
            self.monitor_thread = threading.Thread(target=self._monitor_output)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print(f"[PID:{self.process.pid}] GPU monitoring started. Stats are being saved to {self.output_file}")
            return True
        except Exception as e:
            print(f"Failed to start GPU monitoring: {e}", file=sys.stderr)
            return False

    def _monitor_output(self):
        """Thread function to monitor and save nvidia-smi output"""
        try:
            with open(self.output_file, 'w') as f:
                first_line = self.process.stdout.readline()
                f.write(first_line)
                f.flush()
                
                while not self.stop_event.is_set():
                    line = self.process.stdout.readline()
                    if not line:
                        break
                    f.write(line)
                    f.flush()
        except Exception as e:
            if not self.stop_event.is_set():
                print(f"Error in monitoring thread: {e}", file=sys.stderr)

    def stop(self):
        """Stops the GPU monitoring process."""
        if not self.process:
            print("GPU monitor is not running...")
            return 
    
        self.stop_event.set()

        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print(f"[PID: {self.process.pid}] Stopped monitoring process.")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
                print(f"[PID: {self.process.pid}] Force killing monitoring process.")
        except Exception:
            print(f"[PID: {self.process.pid}] Error stopping monitoring process")
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

        self.process = None
        return

    def _ensure_unique_filename(self, output_path):
        if os.path.exists(output_path):
            base, ext = os.path.splitext(output_path)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            self.output_file = f"{base}_{counter}{ext}"
            print(f"Output file already exists. Using {self.output_file} instead.")
    
        
        

class TaskRunner:
    "Executes benchmarking task, and starts GPU monitor."
    def __init__(self, i, command, server_id, model_name, gpu_monitor=None):
        self.i = i
        self.command = command
        self.gpu_monitor = gpu_monitor
        self.process = None
        self.server_id = server_id
        self.model_name = model_name

    def run(self, timeout=None):
        """Run the command locally and monitor GPU usage"""
        try:
            # Start GPU monitoring if available
            if self.gpu_monitor and not self.gpu_monitor.start():
                print("Failed to start GPU monitoring, exiting")
                return False
            
            print(f"Running command: {self.command}")

            os.makedirs(f"{self.server_id}_logs",exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{self.server_id}_logs/log_{self.i}_{self.model_name}_{timestamp}.txt"
            print(f"Logging to: {log_file}")
            with open(log_file, 'w') as f:
                self.process = subprocess.Popen(
                    shlex.split(self.command), 
                    stdout=f, 
                    stderr=subprocess.STDOUT
                )
                
            try:
                exit_code = self.process.wait(timeout=timeout)
                print(f"Command completed with exit code: {exit_code}")
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                print(f"Command timed out after {timeout} seconds")
                exit_code = 1


            if self.gpu_monitor:
                self.gpu_monitor.stop()
            
            return exit_code == 0

        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping...")
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            if self.gpu_monitor:
                self.gpu_monitor.stop()
            
            return False
        except Exception as e:
            print(f"Error running command: {e}", file=sys.stderr)
            
            if self.gpu_monitor:
                self.gpu_monitor.stop()
            
            return False
        
def setup_signal_handlers(task_runner):
    """Set up signal handlers for graceful termination"""
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, shutting down gracefully...")
        if task_runner and task_runner.process:
            task_runner.process.terminate()
        if task_runner and task_runner.gpu_monitor:
            task_runner.gpu_monitor.stop()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def get_model_command(model):
    commands = {
        'resnet50': """mlcr run-mlperf,inference,_full,_r5.0-dev \
            --model=resnet50 \
            --implementation=reference \
            --framework=tensorflow \
            --category=datacenter \
            --scenario=Offline \
            --execution_mode=valid \
            --device=cuda \
            --rerun \
            --quiet
            """,
        'bert-large': """mlcr run-mlperf,inference,_full,_r5.0-dev \
            --model=bert-99.9 \
            --implementation=reference \
            --framework=pytorch \
            --category=datacenter \
            --scenario=Offline \
            --execution_mode=valid \
            --device=cuda \
            --rerun \
            --quiet 
            """,
        'sdxl-turbo': "python3 huggingface/sdxl-turbo.py",
        'glc': "python3 huggingface/git-large-coco.py"
    }

    return commands[model]

def main():
    parser = argparse.ArgumentParser(description='Run inference benchmarks with GPU monitoring.')
    parser.add_argument('--server_id', type=int, default=0,
                        help='Server ID (default: 0)')
    parser.add_argument('--model', type=str, default=None,
                        help='Comma-separated list of models to benchmark (resnet50, bert-large, sdxl-turbo, glc)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations to run each model (default: 1)')

    args = parser.parse_args()

    server_id = args.server_id
    if args.server_id == 0:
        print("WARNING: Server ID not specified, using default 0.")
    
    # Sanitize model input.
    if not args.model:
        print("Please specify models to benchmark. Terminating...")
        return
    input_models = [model.strip() for model in args.model.split(',')]
    models = []
    for model in input_models:
        command = get_model_command(model)
        if not command:
            print("WARNING: Invalid model inputted. Skipping...")
            continue
        models.append(model)
    
    if not models:
        print("ERROR: No valid models inputted. Terminating...")
        return

    iterations = args.iterations

    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        for model in models:
            command = get_model_command(model)
            print(f"[{i+1}] Running {model}")
            gpu_monitor = GPUMonitor(i+1, server_id, model, i+1)
            task_runner = TaskRunner(i+1, command, server_id, model, gpu_monitor)
            setup_signal_handlers(task_runner)
            task_runner.run()
            print(f"[{i+1}] Completed {model}")
    
    print("Benchmarks completed!")
        
if __name__ == '__main__':
    main()