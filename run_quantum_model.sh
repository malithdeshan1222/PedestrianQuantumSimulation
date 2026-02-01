#!/bin/bash
# Script to run the Quantum Urban Model with specified parameters

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Check if the IBMQ API token environment variable is set
if [ -z "$IBMQ_API_TOKEN" ]; then
    echo "Warning: IBMQ_API_TOKEN environment variable not set."
    echo "To use a real quantum device, set this variable or provide the token with the --api-token option."
fi

# Function to display help
function show_help {
    echo "Usage: ./run_quantum_model.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message and exit"
    echo "  -c, --classical     Use classical computation instead of quantum"
    echo "  -r, --real-device   Use real quantum device instead of simulator"
    echo "  -t, --times TIMES   Times of day to simulate (morning, afternoon, evening, night)"
    echo "                     Multiple times can be specified separated by spaces"
    echo "  -d, --data PATH     Path to input data files"
    echo "  -o, --output PATH   Path to save results"
    echo "  -a, --api-token TKN IBM Quantum API token (will use IBMQ_API_TOKEN env var if not provided)"
    echo ""
    echo "Examples:"
    echo "  ./run_quantum_model.sh --classical"
    echo "  ./run_quantum_model.sh --real-device --times morning evening"
    echo "  ./run_quantum_model.sh --data /path/to/data --output /path/to/results"
}

# Parse command line arguments
ARGS=""
TIMES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--classical)
            ARGS="$ARGS --no-quantum"
            shift
            ;;
        -r|--real-device)
            ARGS="$ARGS --real-device"
            shift
            ;;
        -t|--times)
            shift
            while [[ $# -gt 0 ]] && ! [[ $1 == -* ]]; do
                TIMES="$TIMES $1"
                shift
            done
            ;;
        -d|--data)
            ARGS="$ARGS --data-path $2"
            shift 2
            ;;
        -o|--output)
            ARGS="$ARGS --result-path $2"
            shift 2
            ;;
        -a|--api-token)
            ARGS="$ARGS --api-token $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# If no API token was provided but the environment variable is set, use that
if [[ $ARGS == *"--real-device"* ]] && [[ ! $ARGS == *"--api-token"* ]] && [[ ! -z "$IBMQ_API_TOKEN" ]]; then
    ARGS="$ARGS --api-token $IBMQ_API_TOKEN"
fi

# If times were specified, add them to the command
if [[ ! -z "$TIMES" ]]; then
    ARGS="$ARGS --times$TIMES"
fi

# Run the model
echo "Running Quantum Urban Model..."
python3 main.py $ARGS