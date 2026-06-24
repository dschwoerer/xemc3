files=$(git grep oauth2: | cut -d: -f1 |sort -u)
echo $files
sed -e "s/oauth2:.*@/oauth2:$1@/" -i $files
git add -p
git commit -m 'Update'
