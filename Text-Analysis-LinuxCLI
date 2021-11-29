mkdir text_analysis
cd text_analysis
wget https://archive.org/stream/fishingwithflysk00orvi/fishingwithflysk00orvi_djvu.txt -O fishingorvis.txt
cp fishingorvis.txt fishingorvis_backup.txt
less -N fishingorvis.txt
sed '1,1900d' fishingorvis.txt > fishing_noheader.txt
less -N fishing_noheader.txt
tail fishing_noheader.txt
tail -N fishing_noheader.txt
less -N fishing_noheader.txt
sed '10856,10952d' fishingorvis.txt > fishing_textonly.txt
sed '10856,10952d' fishing_noheader.txt > fishing_textonly.txt
tr -d [:punct:] < fishing_textonly.txt > fishing_nopunct.txt
less -N fishing_nopunct.txt
tr -d [:digit:] < fishing_nopunct.txt > fishing_clean.txt
tr [:upper:] [:lower:] < fishing_clean.txt  > fishing_clean2.txt
less fishing_clean2.txt
tr -d '\r' < fishing_clean2.txt  > fishing_clean3.txt
tr ' ' '\n' < fishing_clean3.txt > fishing_clean4.txt
sed -i '/^$/d' fishing_clean4.txt
sort fishing_clean4.txt > fishing_clean5.txt
uniq -c fishing_clean5.txt > fishing_wordfreq.txt
less fishing_wordfreq.txt
sort -rn fishing_wordfreq.txt > fishing_sorted.txt
