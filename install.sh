#!/bin/bash
# Installation script for Mem0 CLI

echo "Installing Mem0 CLI..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the mem0 script executable
chmod +x "$SCRIPT_DIR/mem0"

# Create a symlink in /usr/local/bin (requires sudo)
if command -v sudo &> /dev/null; then
    echo "Creating symlink in /usr/local/bin..."
    
    # Create a wrapper script that sets the project context
    cat > /tmp/mem0-wrapper << EOF
#!/bin/bash
# Wrapper script for Mem0 CLI
export MEM0_PROJECT_ROOT="$SCRIPT_DIR"
exec "$SCRIPT_DIR/mem0" "\$@"
EOF
    
    chmod +x /tmp/mem0-wrapper
    sudo mv /tmp/mem0-wrapper /usr/local/bin/mem0
    
    echo "✅ Mem0 CLI installed successfully!"
    echo "You can now run 'mem0' from anywhere."
else
    echo "⚠️  Could not create system-wide symlink (sudo not available)"
    echo "You can still run the CLI with: $SCRIPT_DIR/mem0"
    echo "Or add $SCRIPT_DIR to your PATH"
fi

echo ""
echo "Usage examples:"
echo "  mem0 memory add 'I love basketball' --user alice"
echo "  mem0 train --max-trials 3 --num-epochs 2"
echo "  mem0 benchmark inference --model llama-3.1-8b-instruct-bf16"
echo "  mem0 help"
