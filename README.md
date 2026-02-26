# Assignment ⅤI: AVP Localization
  For this assignment, you are required to complete the localization algorithm for the AVP system. The assignment provides a dataset of segmented ipm images from simulation. You need to complete the process of the localization algorithm and display the final result of the AVP localization.  

  - Complete imageRegistration() in src/LocalizationEngine.cpp  

## Build & Run

Terminal (Open in l6_ws): 

```bash 1
catkin_make 
source ./devel/setup.bash
roslaunch DMA_EKF dmaekf_avp_localization.launch
```
 - roslaunch will load data/avp_map.bin and play bag automatically.  
 - you can try  avp_map.bin you built in ***Assignment Ⅴ***


