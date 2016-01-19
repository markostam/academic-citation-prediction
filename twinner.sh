#!/bin/bash
if [ $1!="" ];
	then cd $1
fi
mkdir shuffle
for filename in ./train; do
	twins=ls | grep ${filename%.*}
	if [ $twins!="" ];
		then for twin in twins; do
			mv $twin shuffle/$twin
		done
	fi
done
for file in ./shuffle; do
	mv shuffle/$file train/$file
done
echo "train twinned"
for filename in ./tune; do
	twins=ls | grep ${filename%.*}
	if [ $twins!="" ];
		then for twin in twins; do
			mv $twin shuffle/$twin
		done
	fi
done
for file in ./shuffle; do
	mv shuffle/$file tune/$file
done
echo "tune twinned"
rmdir shuffle