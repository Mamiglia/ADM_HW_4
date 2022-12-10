#!/bin/bash

# To do this commandline exercise we didn't use the original dataset, 
# but the one we used for the first question because we cannot have all those
# zeros for the Male and Female part.
# we will be using the bank_clean.csv dataset, which is the same as before,
# only with dropped Na's and the columns not needed for the first question.

# 1. Which location has the maximum number of purchases been made?
echo
echo "the maximum number of purchases and the location are:" `awk '{print $4}' bank_clean.csv | sort |uniq -c | sort -r |head -1`

# here uniq -c will count the frequencies of elements in the $4 column, but to do so
# it has to be sorted before, so first we sort and then we use uniq -c,
# now we sort again but by occurences so no alphabetical order,
# and finally we choose the first one which is the max with head -1

# 2. In the dataset provided, did females spend more than males, or vice versa?

# We find the spending habits of the two groups F/M
echo
female=$(awk -F'\t' '$3 == "F" {sum += $8} END {print sum}' bank_clean.csv)
male=$(awk -F'\t' '$3 == "M" {sum += $8} END {print sum}' bank_clean.csv)

# We return the spending habits of both to compare them
echo "females spent" $female "in total"
echo "males spent" $male "in total"

# this if statemens will return which group spent more
echo
if [ "$female" -gt "$male" ]
then
    echo "females spent more then males";
else
    echo "males spent more then females";
fi
echo

# 3. Report the customer with the highest average transaction amount in the dataset
echo "The highest average transaction and the customerID associated with it are:"
awk -F '\t' '{arr[$1]+=$8; count[$1]++} END{for (x in arr)print  arr[x]/count[x],x}' bank_clean.csv | sort -nr | head -1

#using the first double condition it searches through the array $1 and adds everytime the amount of $8 when there is an 
# occurence in $1, and it also sums the corresping frequencies in $1,
# then we loop through the arr and it prints the average for each customerID of $1 then we sort it and choose the first one