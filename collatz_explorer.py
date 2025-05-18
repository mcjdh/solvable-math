import time
import argparse
import multiprocessing # Added for parallel processing

def get_collatz_sequence_info(start_n, max_iterations=10000):
    """
    Calculates the Collatz sequence for a starting number,
    determines its stopping time, peak value, and detects cycles.

    Args:
        start_n (int): The positive integer to start the sequence from.
        max_iterations (int): The maximum number of steps to perform
                              to prevent potential infinite loops.

    Returns:
        dict: A dictionary containing information about the sequence:
            - "start_number": The initial number.
            - "sequence": The list of numbers in the sequence.
            - "steps": The number of steps taken.
            - "peak_value": The highest number encountered in the sequence.
            - "reached_one": Boolean, true if 1 was reached.
            - "hit_max_iterations": Boolean, true if max_iterations was reached.
            - "cycle_detected": Boolean, true if any cycle was detected.
            - "is_trivial_cycle": Boolean, true if 1 was reached or the cycle is (4, 2, 1).
            - "detected_cycle_path": The list of numbers forming the detected cycle.
                                     (Standard [4,2,1] if 1 reached, actual cycle otherwise)
    """
    if not isinstance(start_n, int) or start_n < 1:
        raise ValueError("Starting number must be a positive integer.")

    current_n = start_n
    sequence_list = [current_n]
    steps = 0
    seen_in_this_path = {current_n: 0} # Store number and its index in sequence_list
    max_val_in_sequence = start_n

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
            result["cycle_detected"] = True # Reaching 1 means entering the 4-2-1 cycle
            result["is_trivial_cycle"] = True
            result["detected_cycle_path"] = [4, 2, 1] # Standard representation
            break 

        # Apply Collatz rule
        if current_n % 2 == 0:
            current_n = current_n // 2
        else:
            current_n = 3 * current_n + 1
        
        steps += 1
        sequence_list.append(current_n)
        max_val_in_sequence = max(max_val_in_sequence, current_n)

        if current_n in seen_in_this_path:
            result["cycle_detected"] = True
            cycle_start_index = seen_in_this_path[current_n]
            # Capture the cycle elements, not including the final repeated element that closes it
            detected_cycle = sequence_list[cycle_start_index:-1]
            result["detected_cycle_path"] = detected_cycle
            
            is_trivial = False
            # Check if the cycle contains 1 or is a permutation of [1,2,4]
            if 1 in detected_cycle:
                is_trivial = True
                result["detected_cycle_path"] = [4, 2, 1] # Standardize
            else:
                # Check for permutations like [2,4,1], [4,1,2] etc. for the [1,2,4] set
                # A simple way is to check if the sorted version of the detected cycle is [1,2,4]
                # This handles short cycles. For very long cycles, this direct comparison might be too simple
                # if we were looking for sub-cycles not containing 1, but the primary Collatz disproof is a non-1 cycle.
                if sorted(detected_cycle) == [1,2,4]:
                     is_trivial = True
                     result["detected_cycle_path"] = [4, 2, 1] # Standardize
            
            result["is_trivial_cycle"] = is_trivial
            if current_n == 1: # Cycle explicitly ended by reaching 1
                result["reached_one"] = True
                result["is_trivial_cycle"] = True # Ensure this is set

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
    num, max_iter = args_tuple
    return get_collatz_sequence_info(num, max_iterations=max_iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collatz Conjecture Sequence Analyzer.")
    parser.add_argument("--start", type=int, default=1, help="Starting number of the range to test (inclusive).")
    parser.add_argument("--end", type=int, default=2000, help="Ending number of the range to test (inclusive).")
    parser.add_argument("--maxiter", type=int, default=1000, help="Maximum iterations per number before stopping.")
    parser.add_argument("--outfile", type=str, default="novel_collatz_cycles.txt", help="Output file for detected novel cycles.")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes for parallel execution (default: number of CPU cores).")

    args = parser.parse_args()

    start_range = args.start
    end_range = args.end
    max_iter_per_num = args.maxiter
    output_file_for_novel_cycles = args.outfile
    num_workers = args.workers

    if start_range > end_range:
        print("Error: Start range cannot be greater than end range.")
        exit(1)
    if start_range < 1:
        print("Error: Start range must be a positive integer.")
        exit(1)
    if num_workers < 1:
        print("Error: Number of workers must be at least 1.")
        exit(1)

    numbers_to_test = list(range(start_range, end_range + 1))
    # Prepare arguments for the wrapper function
    tasks_with_args = [(num, max_iter_per_num) for num in numbers_to_test]

    print(f"Starting Collatz sequence analysis for numbers from {start_range} to {end_range}")
    print(f"Max iterations per number: {max_iter_per_num}")
    print(f"Using {num_workers} worker process(es).")
    print(f"Novel cycles will be saved to: {output_file_for_novel_cycles}\n")

    non_trivial_cycles_found_list = [] # Renamed to avoid conflict
    did_not_reach_one_within_limit = []
    
    longest_stopping_time = 0
    num_with_longest_stopping_time = 0
    highest_peak_value = 0
    num_with_highest_peak_value = 0
    
    start_time = time.time()
    numbers_processed = 0

    # Using multiprocessing Pool
    # The chunksize can be tuned. Small chunksize can provide better load balancing for varied task times.
    # Larger chunksize reduces overhead of task distribution. 
    # Default chunksize for imap_unordered is 1, which is fine here.
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Using imap_unordered to get results as they complete
        # Pass the wrapper function and the list of argument tuples
        for info in pool.imap_unordered(process_number_wrapper, tasks_with_args):
            numbers_processed += 1

            if info["reached_one"]: # Only consider stopping time if 1 was reached
                if info["steps"] > longest_stopping_time:
                    longest_stopping_time = info["steps"]
                    num_with_longest_stopping_time = info["start_number"]
            
            if info["peak_value"] > highest_peak_value:
                highest_peak_value = info["peak_value"]
                num_with_highest_peak_value = info["start_number"]

            # Reporting for each number (optional, can be verbose)
            # print(f"--- Number: {info['start_number']} ---")
            # print(f"  Steps: {info['steps']}, Peak: {info['peak_value']}")
            # print(f"  Reached 1: {info['reached_one']}")
            
            if info['hit_max_iterations'] and not info['reached_one']:
                # print(f"  Hit max iterations ({max_iter_per_num}) before reaching 1 or a confirmed cycle.")
                did_not_reach_one_within_limit.append(info['start_number'])
            
            if info['cycle_detected'] and not info['is_trivial_cycle']:
                novel_cycle_data = {"start_number": info['start_number'], "cycle_path": info['detected_cycle_path'], "steps_to_cycle_entry": info['steps'], "peak_before_cycle": info['peak_value'] }
                non_trivial_cycles_found_list.append(novel_cycle_data)
                
                print(f"  *** NON-TRIVIAL CYCLE DETECTED! (Processed by main thread) ***")
                print(f"    Start Number: {novel_cycle_data['start_number']}")
                print(f"    Cycle Path: {novel_cycle_data['cycle_path']}")
                print(f"    Steps to Cycle: {novel_cycle_data['steps_to_cycle_entry']}")
                print(f"    Saving to {output_file_for_novel_cycles}...")
                
                try:
                    with open(output_file_for_novel_cycles, 'a') as f_out:
                        f_out.write(f"NON-TRIVIAL CYCLE DETECTED:\n")
                        f_out.write(f"  Start Number: {novel_cycle_data['start_number']}\n")
                        f_out.write(f"  Cycle Path: {novel_cycle_data['cycle_path']}\n")
                        f_out.write(f"  Steps to Cycle Entry: {novel_cycle_data['steps_to_cycle_entry']}\n")
                        f_out.write(f"  Peak Value in Sequence: {novel_cycle_data['peak_before_cycle']}\n")
                        f_out.write(f"  Full sequence leading to cycle: {info['sequence']}\n") # Potentially very long
                        f_out.write(f"---\n")
                    print(f"    Successfully saved to {output_file_for_novel_cycles}.")
                except Exception as e:
                    print(f"    ERROR: Could not write to file {output_file_for_novel_cycles}: {e}")
            # elif not info['reached_one'] and not info['hit_max_iterations']:
            #      print(f"  Ended without reaching 1, detecting a cycle, or hitting max_iterations (Unusual).")
            # print("-" * 20)

    end_time = time.time()
    duration = end_time - start_time
    processing_rate = numbers_processed / duration if duration > 0 else float('inf')
    
    print("\n--- Overall Summary ---")
    print(f"Processed {numbers_processed} numbers (from {start_range} to {end_range}) in {duration:.2f} seconds.")
    print(f"Processing rate: {processing_rate:.2f} numbers/sec.")
    
    if num_with_longest_stopping_time:
        print(f"Number with longest stopping time (to reach 1): {num_with_longest_stopping_time} ({longest_stopping_time} steps).")
    else:
        print("No numbers reached 1 within the iteration limit (or all hit max iterations).")
        
    if num_with_highest_peak_value:
        print(f"Number that reached highest peak value: {num_with_highest_peak_value} (peak: {highest_peak_value}).")

    if non_trivial_cycles_found_list:
        print(f"\nFound {len(non_trivial_cycles_found_list)} non-trivial cycle(s) during this run:")
        for item in non_trivial_cycles_found_list:
            print(f"  Start Number: {item['start_number']}, Cycle Path: {item['cycle_path']}")
    else:
        print("\nNo non-trivial cycles found in the tested range during this run.")

    if did_not_reach_one_within_limit:
        # Sort this list for more consistent output if needed, especially when using multiple workers
        did_not_reach_one_within_limit.sort()
        print(f"Numbers that did not reach 1 (or a confirmed cycle) within {max_iter_per_num} iterations: {len(did_not_reach_one_within_limit)}")
        if len(did_not_reach_one_within_limit) < 20: # Print specifics if the list is not too long
             print(f"  List: {did_not_reach_one_within_limit}")
    else:
        print(f"All tested numbers reached 1 or a trivial cycle within {max_iter_per_num} iterations.")

    # Example of testing a number known for long sequence (e.g., 27, or a higher one like 97 for max_iter=1000)
    # For max_iter_per_num = 1000, 703 takes 261 steps. 97 takes 118 steps. 27 takes 111.
    test_specific_num = 27
    if test_specific_num <= end_range: # only if it was part of the scan or relevant range
        print(f"\n--- Re-checking a specific long sequence ({test_specific_num}) with {max_iter_per_num} iterations ---")
        info_specific = get_collatz_sequence_info(test_specific_num, max_iterations=max_iter_per_num)
        print(f"Number: {info_specific['start_number']}")
        print(f"Steps: {info_specific['steps']}, Peak: {info_specific['peak_value']}")
        print(f"Reached 1: {info_specific['reached_one']}")
        # print(f"Sequence (first 10 and last 10): {info_specific['sequence'][:10]} ... {info_specific['sequence'][-10:]}")
        print(f"Hit max_iterations: {info_specific['hit_max_iterations']}")
        if info_specific['cycle_detected']:
            print(f"Cycle Detected: {info_specific['detected_cycle_path']}")
            print(f"Is Trivial Cycle: {info_specific['is_trivial_cycle']}") 