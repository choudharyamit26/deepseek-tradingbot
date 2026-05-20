import pyautogui
import time

# Configure pyautogui
pyautogui.FAILSAFE = True  # Move mouse to upper-left corner to abort
pyautogui.PAUSE = 0.5  # Add a small pause between actions


def keep_system_awake(interval=120):
    """
    Simulates a key press every 'interval' seconds to keep the system awake.
    Default interval is 120 seconds (2 minutes).
    """
    print("Keeping system awake. Move mouse to upper-left corner to stop.")
    try:
        while True:
            # Press and release the F15 key (a safe key that doesn't affect most apps)
            pyautogui.press("f15")
            print(f"Sent F15 key press at {time.strftime('%H:%M:%S')}")
            # Wait for the specified interval
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Script stopped by user.")
    except pyautogui.FailSafeException:
        print("Script stopped due to mouse in upper-left corner.")


if __name__ == "__main__":
    # Give user a moment to prepare before starting
    print("Starting in 5 seconds...")
    time.sleep(5)
    keep_system_awake()
