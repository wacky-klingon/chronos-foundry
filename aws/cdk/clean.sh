#!/bin/bash
cd "$(dirname "$0")"
echo "Cleaning CDK workspace..."
rm -rf cdk.out/ node_modules/ build/
echo "Clean complete. Run 'npm install && npm run build' to rebuild."

