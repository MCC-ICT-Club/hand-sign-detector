#!/bin/bash

# Name of the script file to copy
script_file="RenameScript.sh"

# Ensure the script file exists
if [[ ! -f  $script_file ]]; then
	echo "Error: $script_file does not exist"
	exit 1

fi

# Loop through all directories in the current directory
for dir in */; do
	if [[ -d $dir ]]; then
		# Copy the script file into the directory
		cp "$script_file" "$dir"
		echo "Copied $script_file to $dir"
	fi
done

echo "Done!"
