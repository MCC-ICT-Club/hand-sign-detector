#!/bin/bash

# Function to generate a unique filename
get_unique_filename() {
    local dir="$1"
    local filename="$2"
    local base="${filename%.*}"
    local ext="${filename##*.}"
    local counter=1
    local new_filename="$filename"

    while [ -e "$dir/$new_filename" ]; do
        new_filename="${base}_${counter}.${ext}"
        counter=$((counter + 1))
    done

    echo "$new_filename"
}

# Loop over each G folder
for g in G{1..11}; do
    # Create or clear the target G folder
    mkdir -p "$g"

    # Copy PNGs from each S folder to the target G folder with unique names
    for s in S{1..4}; do
        if [ -d "$s/$g" ]; then
            echo "Merging PNGs from $s/$g into $g"
            find "$s/$g" -name '*.png' | while read -r file; do
                filename=$(basename "$file")
                unique_filename=$(get_unique_filename "$g" "$filename")
                cp "$file" "$g/$unique_filename"
            done
        fi
    done
done
