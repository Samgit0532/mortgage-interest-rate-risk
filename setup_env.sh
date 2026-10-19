#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="fixed_variable_loans_project"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_YML="$SCRIPT_DIR/environment.yml"

echo "Creating or updating Conda environment..."
conda env create -f "$ENV_YML" \
    || conda env update -n "$ENV_NAME" -f "$ENV_YML" --prune

echo "Installing Jupyter kernel..."
conda run -n "$ENV_NAME" python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python (Fixed-Variable Loans Project)"

echo "Creating environment variables file..."
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'EOF'
PREPAY_INTERCEPT=-5
PREPAY_BETA_CHANGE=4
PREPAY_BETA_GAIN=3e-05
MINIMUM_RATE=0.003
ALMIN= 0.004
EOF
    echo "Created $ENV_FILE"
else
    echo "$ENV_FILE already exists, leaving it unchanged."
fi

eval "$(conda shell.bash hook)"
conda activate fixed_variable_loans_project

echo "Environnement Conda activé !"
echo "Installation completed successfully!"
echo "Select the kernel: Python (Fixed-Variable Loans Project)"
