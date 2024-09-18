#!/bin/bash

# Script to copy files from source directory to destination directory
# without overwriting existing files. If a file with the same name exists,
# the script appends an underscore and a number to the filename.

# Usage: ./copy_unique.sh source_directory destination_directory

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_directory destination_directory"
    exit 1
fi

# Assign arguments to variables for clarity
src_dir="$1"
dst_dir="$2"

# Check if source directory exists
if [ ! -d "$src_dir" ]; then
    echo "Error: Source directory '$src_dir' does not exist."
    exit 1
fi

# Check if destination directory exists; if not, create it
if [ ! -d "$dst_dir" ]; then
    echo "Destination directory '$dst_dir' does not exist. Creating it."
    mkdir -p "$dst_dir"
fi

# Function to copy files uniquely
copy_files_uniquely() {
    local src="$1"
    local dst="$2"

    # Iterate over all items in the source directory
    for src_path in "$src"/*; do
        # Skip if no files are found
        if [ ! -e "$src_path" ]; then
            continue
        fi

        # Get the base name of the file or directory
        item_name=$(basename "$src_path")
        dst_path="$dst/$item_name"

        if [ -d "$src_path" ]; then
            # If it's a directory, create the directory in destination and recurse
            mkdir -p "$dst_path"
            copy_files_uniquely "$src_path" "$dst_path"
        elif [ -f "$src_path" ]; then
            if [ ! -e "$dst_path" ]; then
                # If the file doesn't exist in destination, copy it
                cp "$src_path" "$dst_path"
                echo "Copied '$src_path' to '$dst_path'"
            else
                # File exists; find a unique name
                base="${item_name%.*}"
                ext="${item_name##*.}"
                if [ "$base" = "$ext" ]; then
                    # No extension
                    ext=""
                else
                    ext=".$ext"
                fi
                i=1
                while [ -e "$dst/$base_$i$ext" ]; do
                    i=$((i + 1))
                done
                unique_name="$base_$i$ext"
                cp "$src_path" "$dst/$unique_name"
                echo "Copied '$src_path' to '$dst/$unique_name'"
            fi
        fi
    done
}

# Start copying files
copy_files_uniquely "$src_dir" "$dst_dir"
