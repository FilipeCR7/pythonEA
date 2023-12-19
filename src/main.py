import argparse
import importlib
import sys

def main():
    parser = argparse.ArgumentParser(description="Run a script's main function")
    parser.add_argument('--script', help='Name of the script to run (without .py)', required=True)
    args = parser.parse_args()

    script_name = args.script

    try:
        # Dynamically import the module
        module = importlib.import_module(script_name)

        # Check if the module has a main() function and call it
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"The script {script_name} does not have a main() function.")

    except ModuleNotFoundError:
        print(f"Script {script_name} not found.")

if __name__ == "__main__":
    sys.path.insert(0, './src')  # Add src directory to system path
    main()