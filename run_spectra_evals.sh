#!/bin/bash


# 1 Box
sbatch sbatch_eval_barcode.sh outputs/dec30_random/25_12_31_08:14:58_pet_pika_nearwall_boxfilter_4x_sr        #NW, aug                                                                                  
sbatch sbatch_eval_barcode.sh outputs/dec30_random/25_12_30_21:27:13_solid_hermit_nearwall_boxfilter_4x_sr    #NW, noaug                                                                                      
sbatch sbatch_eval_barcode.sh outputs/dec30_random/25_12_30_13:17:35_super_spider_middle_boxfilter_4x_sr     #MIDDLE, aug                                                                                            
sbatch sbatch_eval_barcode.sh outputs/dec30_random/25_12_30_09:21:54_polite_hog_middle_boxfilter_4x_sr  # Middle no aug

# 3 Box
sbatch sbatch_eval_barcode.sh outputs/jan1_3box/26_01_02_17:44:02_hot_piglet_nearwall_boxfilter_4x_sr        #NW, aug                                                                                  
sbatch sbatch_eval_barcode.sh outputs/jan1_3box/26_01_02_06:05:51_bold_macaw_nearwall_boxfilter_4x_sr    #NW, noaug                                                                                      
sbatch sbatch_eval_barcode.sh outputs/jan1_3box/26_01_01_20:48:50_normal_sloth_middle_boxfilter_4x_sr     #MIDDLE, aug                                                                                            
sbatch sbatch_eval_barcode.sh outputs/jan1_3box/26_01_01_14:26:00_social_vervet_middle_boxfilter_4x_sr  #middle no aug
