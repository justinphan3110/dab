#!/bin/bash
export PROJECT_ID=vietai-research
gcloud config set project ${PROJECT_ID}
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID

for i in {0..999}; do
    file_numb=$i
    # translate_tail=_vietnews.txt.fixed.vi2en.beam4
    # decode_from=gs://best_vi_translation/raw/vietnew_split_by_5k/
    # decode_to_file=$decode_from$file_numb$translate_tail
    
    translate_tail=_stories.txt.en2vi.beam4
    decode_from=gs://best_vi_translation/raw/split_by_5k/
    decode_to_file=$decode_from$file_numb$translate_tail
    
    if [ -f "$decode_to_file" ]; then
        echo ""
    else 
        echo "$file_numb not exists."
    fi
done