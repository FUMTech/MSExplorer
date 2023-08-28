import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_script(_=None):
    result = subprocess.run(["python", "test.py"], capture_output=True, text=True)
    # Return the captured output with separators
    return result.stdout + "\n" + "="*30 + "\n"

def main():
    num_runs = 15
    outputs = []

    # Use ThreadPoolExecutor to run the script in parallel
    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        outputs = list(executor.map(run_script, range(num_runs)))

    # Open the text file for appending
    with open("output.txt", "a") as output_file:
        for output in outputs:
            output_file.write(output)

    print("Finished running and appending to output.txt")

if __name__ == "__main__":
    main()
