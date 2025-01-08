count=0
for file in *.jpg; do
	new_name=$(printf "G10_%04d.jpg" "$count")
	mv "$file" "$new_name"
	count=$((count + 1))
	done


