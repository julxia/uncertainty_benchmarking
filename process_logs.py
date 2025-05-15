import os
import re
from datetime import datetime
import pandas as pd

def read_log_lines(file_path: str) -> list[str]:
    """Reads all lines from a file and handles potential errors."""
    try:
        with open(file_path, 'r') as f:
            return f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def extract_timestamps_sdxl_glc(lines: list[str], pattern: re.Pattern) -> tuple[str | None, str | None]:
    """Extracts start and end timestamps from log lines for 'sdxl' or 'glc' models."""

    start_timestamp_str = None
    end_timestamp_str = None
    for i in range(len(lines) - 1, 0, -1):
        match = pattern.search(lines[i])
        if match:
            start_timestamp_str = match.group(1)
            end_timestamp_str = match.group(2)
            break
    return start_timestamp_str, end_timestamp_str

def extract_throughput(lines: list[str], pattern: re.Pattern) -> int | None:
    """Extracts throughput from log lines based on the provided regex pattern."""

    throughput = None
    for line in lines:
        match = pattern.search(line)
        if match:
            throughput = int(match.group(1))
            break
    return throughput

def parse_log_sdxl_glc(model: str, file_path: str) -> tuple[str | None, str | None, int | None]:
    """Parses log files for 'sdxl' and 'glc' models."""

    lines = read_log_lines(file_path)
    if not lines:
        return None, None, None

    timestamp_pattern = re.compile(r"^START:\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}),\s+END:\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})")
    start_timestamp_str, end_timestamp_str = extract_timestamps_sdxl_glc(lines, timestamp_pattern)

    throughput_pattern = re.compile(r"Total prompts processed:\s+(\d+)") if model == 'sdxl' else re.compile(r"Total tokens generated:\s+(\d+)")
    throughput = extract_throughput(lines, throughput_pattern)

    return start_timestamp_str, end_timestamp_str, throughput

def extract_throughput_bert_resnet(log_content: str, pattern: re.Pattern) -> float | None:
    """Extracts throughput for 'bert' and 'resnet' models from the entire log content."""
    match = pattern.search(log_content)
    return float(match.group(1)) if match else None

def extract_timestamps_bert_resnet(lines: list[str]) -> tuple[str | None, str | None]:
    """Extracts start and end timestamps from log lines for 'bert' and 'resnet' models."""

    std_timestamp_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
    tf_timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d{6}:")
    
    stars_marker_text = "***************************************************************************"
    start_marker_text = "external/local_xla/xla/stream_executor/cuda/cuda_fft.cc"
    end_marker_text = 'call "postprocess" from /home/cc/MLC/repos/mlcommons@mlperf-automations/script/benchmark-program/customize.py'
    script_marker = "CM script::benchmark-program/run.sh"
    performance_cmd_pattern = re.compile(r"CMD:.*--scenario(?:=|\s+)Offline(?!.*--accuracy)")

    in_performance_block = False
    found_first_stars = False
    found_start_marker_line = False
    start_timestamp_str = None
    end_timestamp_str = None

    for i, line in enumerate(lines):
        if not found_first_stars:
            if stars_marker_text in line.strip():
                found_first_stars = True
            continue

        if not in_performance_block and script_marker in line:
            if any(performance_cmd_pattern.search(lines[j]) for j in range(i + 1, min(i + 5, len(lines)))):
                in_performance_block = True
                found_start_marker_line = False
            if in_performance_block:
                continue

        if in_performance_block:
            if not found_start_marker_line and start_marker_text in line:
                timestamp = None
                match_std = std_timestamp_pattern.search(line)
                match_tf = tf_timestamp_pattern.search(line)
                timestamp = match_std.group(1) if match_std else match_tf.group(1) if match_tf else None

                if timestamp:
                    start_timestamp_str = timestamp
                    found_start_marker_line = True
                else:
                    print(f"DEBUG: Found start marker text but no timestamp at line {i+1}")
            elif found_start_marker_line and end_marker_text in line:
                match_std = std_timestamp_pattern.search(line)
                if match_std:
                    end_timestamp_str = match_std.group(1)
                    break
                else:
                    print(f"DEBUG: Found end marker text but no timestamp at line {i+1}")
            elif script_marker in line and i > 0 and script_marker not in lines[i - 1]:
                print(f"DEBUG: Exiting performance block due to new script marker at line {i+1}")
                in_performance_block = False
    return start_timestamp_str, end_timestamp_str

def parse_log_bert_resnet(model: str, file_path: str) -> tuple[str | None, str | None, float | None]:
    """Parses log files specifically for 'bert' and 'resnet' models."""

    lines = read_log_lines(file_path)
    if not lines:
        return None, None, None

    log_content = "".join(lines)
    model_name = f"{model}-99.9" if model == 'bert' else 'resnet'
    throughput_pattern = re.compile(rf"\|\s*{model_name}[-\w]*\s*\|\s*Offline\s*\|\s*[\d.]+\s*\|\s*([\d.]+)")
    throughput = extract_throughput_bert_resnet(log_content, throughput_pattern)

    start_timestamp_str, end_timestamp_str = extract_timestamps_bert_resnet(lines)

    return start_timestamp_str, end_timestamp_str, throughput

def parse_log(model: str, file_path: str) -> tuple[str | None, str | None, float | None]:
    """
    Parses a log file to extract start and end timestamps, and throughput
    based on the specified model.
    """

    if not os.path.isfile(file_path):
        print(f"Path does not exist: {file_path}")
        return None, None, None
    
    try:
        if model in ('sdxl', 'glc'):
            return parse_log_sdxl_glc(model, file_path)
        elif model in ('bert', 'resnet'):
            return parse_log_bert_resnet(model, file_path)
        else:
            print(f"No parser for model: {model} ({file_path})")
            return None, None, None
    except Exception as e:
        print(f"An error occurred processing {file_path}: {e}")
        return None, None, None

def calculate_sample_count(model: str, throughput: float | None, delta: float) -> float | None:
    """Calculates the sample count based on the model type and throughput."""
    if throughput is None:
        return None
    
    if model in ('bert', 'resnet'):
        return throughput * delta
    elif model in ('sdxl', 'glc'):
        return throughput
    else:
        return None

def process_log_file(results_path: str, server_id: str, run_num: str, model: str, file_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Processes a single log file and appends the results to the DataFrame."""

    log_path = os.path.join(results_path, f"{server_id}_logs", file_name)
    start, end, throughput = parse_log(model, log_path)

    if not start or not end:
        print(f"Warning: Could not extract start or end time from {log_path}, skipping")
        return df

    start = start.replace('/', '-').split(".")[0]
    end = end.replace('/', '-').split(".")[0]
    date_format = "%Y-%m-%d %H:%M:%S"
    start_dt = datetime.strptime(start, date_format)
    end_dt = datetime.strptime(end, date_format)

    delta = (end_dt - start_dt).total_seconds()
    delta_h = delta / 3600.0
    sample_count = calculate_sample_count(model, throughput, delta)

    monitor_path = os.path.join(results_path, f"{server_id}_monitor/{run_num}_{model}.csv")
    monitor_df = pd.read_csv(monitor_path, skipinitialspace=True)
    monitor_df.columns = monitor_df.columns.str.strip()
    monitor_df['timestamp'] = monitor_df['timestamp'].apply(lambda x: datetime.strptime(x.replace('/', '-').split(".")[0], date_format))
    filtered_df = monitor_df[(monitor_df['timestamp'] >= start_dt) & (monitor_df['timestamp'] <= end_dt)]
    average_power = filtered_df['power.draw [W]'].mean() / 1000.0
    energy = average_power * delta_h
    energy_per_sample = (energy) / sample_count if sample_count else None

    new_row = pd.DataFrame([[server_id, model, run_num, start_dt, start_dt.hour, end_dt, delta, delta_h, average_power, energy, energy_per_sample]], columns=df.columns)
    return pd.concat([df, new_row], ignore_index=True)

def process_results():
    """Processes log and monitor files to calculate performance and energy metrics."""

    RESULTS_PATH = 'results'
    df = pd.DataFrame(columns=['server_id', 'model', 'run_#', 'start_time', 'start_hour', 'end_time', 'delta (in seconds)',
                               'delta (in hours)', 'avg_pow_draw (kW)', 'total_energy (kWh)', 'energy_per_sample (kWh)'])

    folders = os.listdir(RESULTS_PATH)
    for folder_name in folders:
        if folder_name.startswith(".") or os.path.isfile(os.path.join(RESULTS_PATH, folder_name)):
            continue
        server_id, folder_type = folder_name.split("_")
        if folder_type == 'logs':
            folder_path = os.path.join(RESULTS_PATH, folder_name)
            files = os.listdir(folder_path)
            for file_name in files:
                if not file_name.startswith("."):
                    _, run_num, model, *_ = file_name.split("_")
                    df = process_log_file(RESULTS_PATH, server_id, run_num, model, file_name, df)

    df.to_csv('results/processed_results.csv')

if __name__ == "__main__":
    process_results()