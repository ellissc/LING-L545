total=`cat wiki.txt  | sed 's/[^a-zA-Z]\+/ /g' | tr ' ' '\n'  | grep -v '^$' | wc -l`
unknown=`cat wiki.txt  | sed 's/[^a-zA-Z]\+/ /g' | tr ' ' '\n'  | grep -v '^$' | hfst-lookup -qp grn.mor.hfst  | grep 'inf' | wc -l`
echo "(($total-$unknown)/$total)*100" | bc -l
