#! /bin/bash

set -e

DOWNLOAD_PATH="$LOSS_NLI_DATA"
if ["$LOSS_NLI_DATA" == ""]; then
    echo "LOSS_NLI_DATA not set; downloading to default path ('data')."
fi

# Down the data zip file
gdown 1EMenaT4KezsGBkbmmKwyaoBBfY4BftcB
echo "Download zip file done!"

# unzip
unzip -o data_nli.zip
echo "Unzip file done!"

# change name folder
mv ./data_nli ./data

# remove zip file
rm -f data_nli.zip

echo "Download dataset nli done!"