mkdir unix_test_analysis
cd unix_test_analysis
wget http://www.gutenberg.org/cache/epub/17192/pg17192.txt -O theraven.txt
head theraven.txt
tail theraven.txt
cp theraven.txt > theraven_backup.txt
sed '1,695d' theraven.txt > theraven_noheader.txt
sed '282,655d' theraven_noheader.txt > theraven_textonly.txt
wc theraven_textonly.txt
grep -n "raven" theraven_textonly.txt
grep -E -n "(R|r)aven" theraven.txt
tr -d [:punct:] < theraven_textonly.txt > theraven_nopunct.txt
tr [:upper:] [:lower:] < theraven_nopunct.txt > theraven_lower.txt
tr -d '\r' < theraven_lower.txt > theraven_lf.txt
tr ' ' '\n' < theraven_lf.txt > theraven_oneword.txt
sed -i '/^$/d' theraven_oneword.txt
sort theraven_oneword.txt > theraven_sorted.txt
uniq -c theraven_sorted.txt > theraven_wordfreq.txt
