# https://github.com/hendrycks/robustness?tab=readme-ov-file

wget https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar

tar -xvf CIFAR-10-C.tar
tar -xvf CIFAR-100-C.tar

rm CIFAR-10-C.tar
rm CIFAR-100-C.tar

default_rootdir_logpath="cache/rootdir"

# Define warning message
warn_msg="Warning: $default_rootdir_logpath does not exist. Setting to './cache'"

# Check if the directory exists
if [ -d "$default_rootdir_logpath" ]; then
    Rootdir=$(cat "$default_rootdir_logpath")  # or use appropriate logic to read the content if needed
else
    echo "$warn_msg"
    Rootdir="./cache"
fi

mkdir -p "$Rootdir"

echo "Moving CIFAR-10-C and CIFAR-100-C dataset to $Rootdir"
mv CIFAR-10-C "$Rootdir/"
mv CIFAR-100-C "$Rootdir/"
echo "CIFAR-10-C and CIFAR-100-C dataset has been moved to $Rootdir"