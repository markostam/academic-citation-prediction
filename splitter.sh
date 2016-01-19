#!/bin/bash
if [ $1!="" ];
	then cd $1
fi
mkdir train
mkdir tune
for filename in *; do
	if [ -f $filename ]; 
		then if [ $(($RANDOM % 2)) -eq 0 ];
			then mv "$filename" "./train"
			else mv "$filename" "./tune"
		fi
		else echo $filename" is not a file"
	fi
done
echo "Files randomly separated."
