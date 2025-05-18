import time
import argparse
import multiprocessing # Added for parallel processing
import signal
import sys
import logging

# Global shutdown event for coordinating graceful termination
shutdown_event = multiprocessing.Event()

# Setup basic logging
logger = logging.getLogger("CollatzExplorer")
# Prevent multiprocessing from trying to reconfigure root logger on Windows if script is frozen
if not getattr(sys, 'frozen', False) and not sys.stdout.isatty() :
    # Heuristic: if not frozen and stdout is not a TTY, likely being piped or run by a service.
    # This can sometimes cause issues with multiprocessing's attempts to re-initialize logging.
    # In such cases, rely on the parent process's logging setup.
    pass
else:
    # Default handler for console - level will be set based on args
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO) # Default logger level, can be overridden by args

# --- Global variable to store the file path for numbers not reaching 1 ---
# This allows the process_number_wrapper to access it without passing it through every call if set.
# This is a slight simplification for multiprocessing; a more complex setup might use a dedicated queue for this.
_failed_log_path = None


def signal_handler(sig, frame):
    # Use print for critical signal path, as logger might be in a weird state during shutdown
    print(f'Signal {sig} received, initiating graceful shutdown...', flush=True)
    logger.warning(f"Signal {sig} received, initiating graceful shutdown.")
    shutdown_event.set()

def get_collatz_sequence_info(start_n, max_iterations=10000):
    """
    Calculates the Collatz sequence for a starting number using the (3n+1)/2 shortcut,
    determines its stopping time, peak value, and detects cycles.

    Args:
        start_n (int): The positive integer to start the sequence from.
        max_iterations (int): The maximum number of steps to perform.

    Returns:
        dict: Information about the sequence:
            - "start_number", "sequence", "steps", "peak_value", 
            - "reached_one", "hit_max_iterations", "cycle_detected",
            - "is_trivial_cycle": True if 1 reached or cycle is the (1,2) trivial cycle.
            - "detected_cycle_path": Standard [1,2] if trivial, actual cycle otherwise.
    """
    if not isinstance(start_n, int) or start_n < 1:
        # This should ideally not happen if input validation is done before calling.
        logger.error(f"Invalid start_n received in get_collatz_sequence_info: {start_n}")
        raise ValueError("Starting number must be a positive integer.")

    current_n = start_n
    sequence_list = [current_n]
    steps = 0
    seen_in_this_path = {current_n: 0} # Store number and its index in sequence_list
    max_val_in_sequence = start_n

    # Trivial cycle with the (3n+1)/2 shortcut is (1, 2)
    TRIVIAL_CYCLE_REPRESENTATION = [1, 2] 
    # Permutations for checking, e.g. if cycle is detected as [2,1]
    SORTED_TRIVIAL_CYCLE = sorted(TRIVIAL_CYCLE_REPRESENTATION)

    result = {
        "start_number": start_n,
        "sequence": [],
        "steps": 0,
        "peak_value": start_n,
        "reached_one": False,
        "hit_max_iterations": False,
        "cycle_detected": False,
        "is_trivial_cycle": False,
        "detected_cycle_path": []
    }

    for i in range(max_iterations):
        if current_n == 1:
            result["reached_one"] = True
            result["cycle_detected"] = True 
            result["is_trivial_cycle"] = True
            result["detected_cycle_path"] = TRIVIAL_CYCLE_REPRESENTATION
            break 

        # Apply Optimized Collatz rule
        if current_n % 2 == 0:
            current_n = current_n // 2
        else:
            # (3n+1)/2 shortcut for odd numbers
            current_n = (3 * current_n + 1) // 2
        
        steps += 1
        sequence_list.append(current_n)
        max_val_in_sequence = max(max_val_in_sequence, current_n)

        if current_n in seen_in_this_path:
            result["cycle_detected"] = True
            cycle_start_index = seen_in_this_path[current_n]
            detected_cycle = sequence_list[cycle_start_index:-1]
            result["detected_cycle_path"] = detected_cycle
            
            is_trivial = False
            if 1 in detected_cycle or 2 in detected_cycle: # If 1 or 2 is in the cycle, it must be the trivial one
                if sorted(detected_cycle) == SORTED_TRIVIAL_CYCLE:
                    is_trivial = True
                    result["detected_cycle_path"] = TRIVIAL_CYCLE_REPRESENTATION
            
            result["is_trivial_cycle"] = is_trivial
            if current_n == 1: # Cycle explicitly ended by reaching 1
                result["reached_one"] = True
                result["is_trivial_cycle"] = True # Ensure this is set and path is standard
                result["detected_cycle_path"] = TRIVIAL_CYCLE_REPRESENTATION

            break 
        
        seen_in_this_path[current_n] = len(sequence_list) - 1 
    else: 
        result["hit_max_iterations"] = True

    result["sequence"] = sequence_list
    result["steps"] = steps
    result["peak_value"] = max_val_in_sequence
    return result

# Helper function for multiprocessing to pass multiple arguments to get_collatz_sequence_info
# Since pool.map/imap only accept functions with a single argument easily.
def process_number_wrapper(args_tuple):
    # Check for shutdown before starting potentially long computation
    if shutdown_event.is_set():
        return None # Or some indicator that task was skipped due to shutdown
    
    num, max_iter = args_tuple
    result = get_collatz_sequence_info(num, max_iterations=max_iter)

    # Handle logging for numbers not reaching 1 to a dedicated file, if configured
    # This check is done in the worker process.
    if _failed_log_path and result['hit_max_iterations'] and not result['reached_one']:
        try:
            with open(_failed_log_path, 'a') as f_failed:
                f_failed.write(f"{result['start_number']}\n")
        except IOError as e:
            logger.error(f"Worker failed to write to failed_log {_failed_log_path} for N={result['start_number']}: {e}")
            # Optionally, could return a special marker or re-raise to signal this,
            # but for now, just log it from worker. Main process won't add it to its in-memory list.
    return result

# Task generator for indefinite run
def task_generator_func(initial_n, max_iterations_per_task, event):
    n = initial_n
    while not event.is_set():
        yield (n, max_iterations_per_task)
        n += 1
    logger.info(f"Task generator received shutdown signal. Stopped yielding new tasks. Last n offered: {n-1}")

# Function to initialize the global failed_log_path for worker processes
def initialize_worker(failed_log_file_path):
    global _failed_log_path
    _failed_log_path = failed_log_file_path
    # Also, re-register signal handler for workers if they are expected to catch signals independently
    # For SIGINT, usually only main process gets it. For SIGTERM, it depends.
    # For now, workers rely on shutdown_event which is multiprocessing-safe.

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Collatz Conjecture Sequence Analyzer using (3n+1)/2 shortcut. Runs indefinitely until interrupted (Ctrl+C).",
        formatter_class=argparse.RawTextHelpFormatter # Allows for newlines in help text
    )
    parser.add_argument("--start", type=int, default=1, 
                        help="Starting number for the analysis. To explore different number ranges upon restarting the script, "
                             "provide a different starting number. Default: 1.")
    # Removed --end argument for indefinite run
    parser.add_argument("--maxiter", type=int, default=2000, help="Maximum iterations per number (using (3n+1)/2 shortcut). Default: 2000.")
    parser.add_argument("--outfile", type=str, default="novel_collatz_cycles.txt", help="Output file for detected novel cycles. Default: novel_collatz_cycles.txt.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes (default: number of CPU cores).")
    parser.add_argument("--verbose", action="store_true", help="Print details for each number processed (can be very noisy).")
    parser.add_argument("--progress-interval", type=int, default=10000, help="Print a progress update every N numbers processed. Default: 10000. Set to 0 to disable.")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file for saving all log messages (DEBUG level and above).")
    parser.add_argument("--failed-log", type=str, default=None, help="Path to a file for logging numbers that hit max_iterations without reaching 1. If not set, these are kept in memory (and may be truncated in summary).")

    args = parser.parse_args()

    # Configure logging based on arguments
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='a') # Append mode
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
        # If logging to file, we might want console to be less verbose unless --verbose is also set
        if not args.verbose:
            logger.setLevel(logging.INFO) # Default for logger if file is used, console is INFO
            console_handler.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG) # Overall logger level
            console_handler.setLevel(logging.DEBUG) # Console also DEBUG
    elif args.verbose: # No log file, but verbose is set
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
    else: # No log file, not verbose
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

    start_num_for_run = args.start # Renamed from start_range
    max_iter_per_num = args.maxiter
    output_file_for_novel_cycles = args.outfile
    num_workers = args.workers
    verbose_output = args.verbose
    progress_update_interval = args.progress_interval
    
    # Set the global for worker processes if --failed-log is used
    if args.failed_log:
        _failed_log_path = args.failed_log
        logger.info(f"Numbers hitting max_iterations will be logged to: {_failed_log_path}")
        # Clear the file at the start of the run if it exists? Or append? For now, append.
        # Consider adding a check or an option for this.

    if start_num_for_run < 1:
        logger.error("Error: Start number must be a positive integer.")
        exit(1)
    if num_workers < 1:
        logger.error("Error: Number of workers must be at least 1.")
        exit(1)

    # Removed numbers_to_test list and tasks_with_args pre-population

    logger.info(f"Starting Collatz sequence analysis (using (3n+1)/2 shortcut) from number {start_num_for_run} indefinitely.")
    logger.info(f"Script will run until manually interrupted (Ctrl+C).")
    logger.info(f"Max iterations per number: {max_iter_per_num}")
    logger.info(f"Using {num_workers} worker process(es).")
    if progress_update_interval > 0:
        logger.info(f"Progress updates will be printed every {progress_update_interval} numbers.")
    logger.info(f"Novel cycles will be saved to: {output_file_for_novel_cycles}")
    if args.log_file:
        logger.info(f"Detailed logs will be saved to: {args.log_file}")

    non_trivial_cycles_found_list = []
    # This list is now only used if _failed_log_path is None
    did_not_reach_one_within_limit_memory = [] 
    count_did_not_reach_one = 0 # Always count, regardless of logging to file or memory
    
    longest_stopping_time = 0
    num_with_longest_stopping_time = 0
    highest_peak_value = 0
    num_with_highest_peak_value = 0
    
    start_time = time.time()
    numbers_processed = 0
    last_progress_print_time = start_time

    # Using multiprocessing Pool
    try:
        # Pass the _failed_log_path to the worker initializer
        pool_initializer = initialize_worker if args.failed_log else None
        pool_initargs = (args.failed_log,) if args.failed_log else ()

        with multiprocessing.Pool(processes=num_workers, initializer=pool_initializer, initargs=pool_initargs) as pool:
            task_iterable = task_generator_func(start_num_for_run, max_iter_per_num, shutdown_event)
            
            current_highest_processed_n = start_num_for_run -1

            for info in pool.imap_unordered(process_number_wrapper, task_iterable):
                if shutdown_event.is_set() and info is None: 
                    # This handles case where task_generator stops and pool might still process a few Nones
                    # if process_number_wrapper returns None due to shutdown_event before actual processing
                    logger.debug("Received None from worker, likely due to shutdown, skipping.")
                    continue
                if info is None: # Should not happen if shutdown_event check above is robust
                    logger.warning("Received unexpected None from worker. Skipping.")
                    continue

                numbers_processed += 1
                current_highest_processed_n = max(current_highest_processed_n, info['start_number'])

                logger.debug(f"Num: {info['start_number']:<8} Steps: {info['steps']:<5} Peak: {info['peak_value']:<10} Reached_1: {info['reached_one']} Hit_Max: {info['hit_max_iterations']}")

                if info["reached_one"]: 
                    if info["steps"] > longest_stopping_time:
                        longest_stopping_time = info["steps"]
                        num_with_longest_stopping_time = info["start_number"]
            
                if info["peak_value"] > highest_peak_value:
                    highest_peak_value = info["peak_value"]
                    num_with_highest_peak_value = info["start_number"]

                if info['hit_max_iterations'] and not info['reached_one']:
                    count_did_not_reach_one += 1
                    if not _failed_log_path: # Only append to memory list if not logging to file
                        did_not_reach_one_within_limit_memory.append(info['start_number'])
            
                if info['cycle_detected'] and not info['is_trivial_cycle']:
                    novel_cycle_data = {
                        "start_number": info['start_number'], 
                        "cycle_path": info['detected_cycle_path'], 
                        "steps_to_cycle_entry": info['steps'], 
                        "peak_before_cycle": info['peak_value'] 
                    }
                    non_trivial_cycles_found_list.append(novel_cycle_data)
                
                    # Log as warning and also print to stdout for emphasis
                    cycle_msg = (f"NON-TRIVIAL CYCLE DETECTED! Start: {novel_cycle_data['start_number']}, "
                                 f"Path: {novel_cycle_data['cycle_path']}, Steps: {novel_cycle_data['steps_to_cycle_entry']}")
                    logger.warning(cycle_msg)
                    print(f"\n  *** {cycle_msg} ***") # Direct print for immediate visibility
                    print(f"    Saving to {output_file_for_novel_cycles}...", flush=True)
                    logger.info(f"Saving non-trivial cycle for start N={novel_cycle_data['start_number']} to {output_file_for_novel_cycles}")
                
                    try:
                        with open(output_file_for_novel_cycles, 'a') as f_out:
                            f_out.write(f"NON-TRIVIAL CYCLE DETECTED ((3n+1)/2 shortcut version):\n")
                            f_out.write(f"  Start Number: {novel_cycle_data['start_number']}\n")
                            f_out.write(f"  Cycle Path: {novel_cycle_data['cycle_path']}\n")
                            f_out.write(f"  Steps to Cycle Entry: {novel_cycle_data['steps_to_cycle_entry']}\n")
                            f_out.write(f"  Peak Value in Sequence: {novel_cycle_data['peak_before_cycle']}\n")
                            f_out.write(f"  Full sequence leading to cycle: {info['sequence']}\n") 
                            f_out.write(f"---\n")
                    except IOError as e:
                        logger.error(f"Error writing novel cycle to file {output_file_for_novel_cycles}: {e}")
                        print(f"Error writing novel cycle to file: {e}", flush=True)
                
                # Progress Reporting
                if progress_update_interval > 0 and numbers_processed % progress_update_interval == 0:
                    current_run_time = time.time() - start_time
                    nums_per_sec = numbers_processed / current_run_time if current_run_time > 0 else 0
                    logger.info("--- Progress ---")
                    logger.info(f"  Numbers Processed: {numbers_processed}")
                    logger.info(f"  Current Highest N: {current_highest_processed_n}")
                    logger.info(f"  Elapsed Time: {current_run_time:.2f}s")
                    logger.info(f"  Rate: {nums_per_sec:.2f} num/s")
                    logger.info(f"  Longest Stop: {num_with_longest_stopping_time} (Steps: {longest_stopping_time})")
                    logger.info(f"  Highest Peak: {num_with_highest_peak_value} (Peak: {highest_peak_value})")
                    logger.info(f"  Novel Cycles Found: {len(non_trivial_cycles_found_list)}")
                    logger.info(f"  Hit Max Iterations: {count_did_not_reach_one}")

                if shutdown_event.is_set():
                    logger.info("Shutdown initiated by event, processing remaining tasks from pool...")
                    # Loop will break naturally when imap_unordered is exhausted after generator stops.

            if shutdown_event.is_set(): # After loop completes, if shutdown was triggered
                logger.info("All tasks processed after shutdown signal.")

    except KeyboardInterrupt: 
        # This is a fallback; signal handler should catch it first.
        # print is okay here as it's an emergency exit.
        print("\nKeyboardInterrupt caught directly in main. Forcing shutdown...", flush=True)
        logger.critical("KeyboardInterrupt caught directly in main. Forcing shutdown...")
        shutdown_event.set()
    
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
        # print is okay here for visibility if logger isn't working.
        print(f"\nAn unexpected error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
        shutdown_event.set() 

    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        summary_type = "Interrupted Summary" if shutdown_event.is_set() else "Final Summary"
        
        logger.info(f"\n--- {summary_type} ({numbers_processed} numbers processed) ---")
        logger.info(f"Total processing time: {total_time:.2f} seconds.")
        if numbers_processed > 0 and total_time > 0:
            avg_time_num = total_time / numbers_processed
            avg_rate_num = numbers_processed / total_time
            logger.info(f"Average time per number: {avg_time_num:.6f} seconds.")
            logger.info(f"Average rate: {avg_rate_num:.2f} numbers/second.")
        
        last_n_val = current_highest_processed_n if 'current_highest_processed_n' in locals() else 'N/A'
        logger.info(f"Last number processed or attempted: {last_n_val}")
        logger.info(f"Number with longest stopping time (reached 1): {num_with_longest_stopping_time} (Steps: {longest_stopping_time}) (Using (3n+1)/2 shortcut)")
        logger.info(f"Number with highest peak value: {num_with_highest_peak_value} (Peak: {highest_peak_value})")

        logger.info(f"Total numbers that hit max_iterations ({max_iter_per_num}) without reaching 1: {count_did_not_reach_one}")
        if _failed_log_path:
            logger.info(f"  These numbers were logged to: {_failed_log_path}")
        elif count_did_not_reach_one > 0 : # Only print from memory if not using failed_log and list is not empty
            if count_did_not_reach_one <= 20:
                 logger.info(f"  List: {did_not_reach_one_within_limit_memory}")
            else:
                 logger.info(f"  (In-memory list truncated. First 20: {did_not_reach_one_within_limit_memory[:20]})")
        elif count_did_not_reach_one == 0 :
             logger.info("All processed numbers reached 1 within the iteration limit or were part of a detected cycle.")

        if non_trivial_cycles_found_list:
            logger.warning(f"Non-trivial cycles detected: {len(non_trivial_cycles_found_list)}")
            for cycle_info in non_trivial_cycles_found_list:
                logger.warning(f"  - Start: {cycle_info['start_number']}, Cycle: {cycle_info['cycle_path']}, Steps to entry: {cycle_info['steps_to_cycle_entry']}")
        else:
            logger.info("No non-trivial cycles detected during this run.")
        
        logger.info(f"Novel cycles (if any) saved to: {output_file_for_novel_cycles}")
        logger.info("Exiting.")
        
        # Ensure all handlers are flushed, especially file handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler) # Optional: clean up handlers
        
        # Final print to physical stdout for very basic confirmation if logging is broken
        print("Script finished or interrupted. Check logs for details.", flush=True) 