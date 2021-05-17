#!/bin/bash
translate() {
    tpu_num=$1
    tpu_work_num=$2
    j=$3
    # file_numb=$((tpu_num * tpu_work_num + j))
    file_addr=$((tpu_num * tpu_work_num + j))
    file_numb="${left[file_addr]}"
    tpu_name=$4
    echo 'file_numb' $file_numb
    echo 'tpu_name' $tpu_name

    file_tail=_vietnews.txt.fixed
    translate_tail=_vietnews.txt.fixed.vi2en.beam4
    decode_from=gs://best_vi_translation/raw/vietnew_split_by_5k/
    decode_to_file=$decode_from$file_numb$translate_tail
    decode_from_file=$decode_from$file_numb$file_tail
    
    echo 'decode from ' $decode_from_file
    echo 'decode to ' $decode_to_file
    
    python3 t2t_decoder.py \
    --alsologtostderr \
    --output_dir=$ckpt_dir \
    --checkpoint_path=$ckpt_path \
    --use_tpu \
    --cloud_tpu_name=$tpu_name \
    --data_dir=$train_data_dir --problem=$problem \
    --hparams_set=$hparams_set \
    --model=transformer \
    --decode_hparams="beam_size=4alpha=0.6log_results=Falsereturn_beams=True" \
    --decode_from_file=$decode_from_file \
    --decode_to_file=$decode_to_file 
    
}

tpu_translate(){
    vm_num=$1
    count=$2
    i=$3
    name=$4

    tpu_num=$((vm_num * count + i))
    tpu_name=$name$tpu_num
    # tpu_work_num=10
    tpu_work_num=4
    
    for j in {0..3}; do
        translate $tpu_num $tpu_work_num $j $tpu_name
    done    
    echo 'done on tpu' $tpu_num
}


export name='translate-'

vm_num=$1  ##{0..4}
count=20


export train_data_dir=gs://best_vi_translation/data/translate_class11_pure_vien_iwslt32k
export problem=translate_class11_pure_vien_iwslt32k
export hparams_set=transformer_tall9

export ckpt_dir=gs://best_vi_translation/checkpoints/translate_class11_pure_vien_tall9_2m/SAVE
export ckpt_path=gs://best_vi_translation/checkpoints/translate_class11_pure_vien_tall9_2m/SAVE/model.ckpt-500000

export PROJECT_ID=vietai-research
gcloud config set project ${PROJECT_ID}
gcloud beta services identity create --service tpu.googleapis.com --project $PROJECT_ID
gcloud auth application-default login

left=(180 181 182 183 184 185 186 187 188 189 200 201 202 
      203 204 205 206 207 208 209 210 211 212 213 214 215 
      216 217 218 219 220 221 222 223 224 225 226 227 228 
      229 230 231 232 233 234 235 236 237 238 239 240 241 
      242 243 244 245 246 247 248 249 250 251 252 253 254 
      255 256 257 258 259 260 261 262 263 264 265 266 267 
      268 269 270 271 272 273 274 275 276 277 278 279 280 
      281 282 283 284 285 286 287 288 289 290 291 292 293 
      294 295 296 297 298 299 300 301 302 303 304 305 306 
      307 308 309 310 311 312 313 314 315 316 317 318 319 
      320 321 322 323 324 325 326 327 328 329 330 331 332 
      333 334 335 336 337 338 339 340 341 342 343 344 345 
      346 347 348 349 350 351 352 353 354 355 356 357 358 
      359 360 361 362 363 364 365 366 367 368 369 370 371 
      372 373 374 375 376 377 378 379 380 381 382 383 384 
      385 386 387 388 389 390 391 392 393 394 395 396 397 
      398 399 600 601 602 603 604 605 606 607 608 609 610 
      611 612 613 614 615 616 617 618 619 620 621 622 623 
      624 625 626 627 628 629 630 631 632 633 634 635 636 
      637 638 639 640 641 642 643 644 645 646 647 648 649 
      650 651 652 653 654 655 656 657 658 659 660 661 662 
      663 664 665 666 667 668 669 670 671 672 673 674 675 
      676 677 678 679 680 681 682 683 684 685 686 687 688 
      689 690 691 692 693 694 695 696 697 698 699 700 701 
      702 703 704 705 706 707 708 709 710 711 712 713 714 
      715 716 717 718 719 720 721 722 723 724 725 726 727 
      728 729 730 731 732 733 734 735 736 737 738 739 740 
      741 742 743 744 745 746 747 748 749 750 751 752 753 
      754 755 756 757 758 759 760 761 762 763 764 765 766 
      767 768 769 770 771 772 773 774 775 776 777 778 779 
      780 781 782 783 784 785 786 787 788 789 790 791 792 
      793 794 795 796 797 798 799);


# for ((i=0;i<$count;i++)); do  # i in {$count_0..$count_1}; do  
for i in {0..19}; do
    tpu_translate $vm_num $count $i $name & 
done