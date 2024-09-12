#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source-directory>"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="."

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory $SOURCE_DIR does not exist."
    exit 1
fi

# Find and copy all .png files
find "$SOURCE_DIR" -type f -name '*.png' | while IFS= read -r file; do
    # Get the base name of the file
    base_name=$(basename "$file")
    dest_file="$DEST_DIR/$base_name"

    # If the file already exists in the destination, rename it
    if [ -e "$dest_file" ]; then
        # Find a new name by appending a number
        base_name_no_ext="${base_name%.*}"
        ext="${base_name##*.}"
        counter=1

        while [ -e "$DEST_DIR/${base_name_no_ext}_$counter.$ext" ]; do
            ((counter++))
        done

        dest_file="$DEST_DIR/${base_name_no_ext}_$counter.$ext"
    fi

    # Copy the file to the destination
    cp "$file" "$dest_file"
    echo "Copied $file to $dest_file"
done
