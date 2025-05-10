#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
SAVE_DIR="$PROJECT_DIR/saved_runs"

# Find pattern 20250426_[143854|144438|145013]_*4x32*
dirnames=($SAVE_DIR/20250426_15*_*4x32*)
# dirnames=$(ls -d $SAVE_DIR/20250426_*4x128*)

# Askar for confirmation before renaming
echo "The following directories will be renamed:"

newnames=()

for dirname in "${dirnames[@]}"; do
    newname=$(echo "$dirname" | sed 's/4x32/2x64/')
    newnames+=("$newname")
done

for ((i=0; i<${#dirnames[@]}; i++)); do
    newname=${newnames[$i]}
    dirname=${dirnames[$i]}
    echo "$i Renaming $dirname to $newname "
    printf "\n"
done

read -p "Are you sure you want to rename these directories? (y/n) " -n 1 -r
# exit if the user does not confirm
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\nAborting."
    exit 1
fi

for ((i=0; i<${#dirnames[@]}; i++)); do
    newname=${newnames[$i]}
    dirname=${dirnames[$i]}
    echo "Renaming $dirname to $newname"
    mv "$dirname" "$newname"
done

