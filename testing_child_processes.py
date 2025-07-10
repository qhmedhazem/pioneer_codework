import subprocess
import time


if __name__ == "__main__":
    proc = subprocess.Popen(
        "roscore",
        shell=False,
    )

    time.sleep(10)

    proc.wait(100)
    print("Process Destroyed")