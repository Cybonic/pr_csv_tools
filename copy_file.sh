#!/bin/bash

# Copy an array of PDF files to a different folder

# Source files array
# source_files=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
source_files=(
    "kitti-GEORGIA-FR-husky-orchards-10nov23-00_range5m.pdf"
    "kitti-greenhouse-e3_range5m.pdf"
    "kitti-uk-orchards-aut22_range5m.pdf"
    "kitti-uk-orchards-june23_range5m.pdf"
    "kitti-uk-orchards-sum22_range5m.pdf"
    "kitti-uk-strawberry-june23_range5m.pdf"
)

# Root folder path
root_folder="/home/deep/Dropbox/SHARE/orchards-uk/code/result_tools/saved_graphs_paper_iros24v2/top25/w_label"

# Destination folder path
destination_folder="/home/deep/Dropbox/SHARE/orchards-uk/SPCov/figures"

paper_destination_folder="/home/deep/workspace/SPCoV/paper_figs"
 
# Loop through the source files array and copy each file to the destination folder
for file in "${source_files[@]}"; do
    cp "$root_folder/$file" "$destination_folder"
done


for file in "${source_files[@]}"; do
    cp "$root_folder/$file" "$paper_destination_folder"
done