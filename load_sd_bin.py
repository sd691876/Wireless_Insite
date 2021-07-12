# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:03:08 2021

@author: User
"""

import os, glob, subprocess
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from scipy.io import savemat, loadmat
from distutils import dir_util, file_util
import pdb

HEADER_LEN = 7
RSSI_LEN = 192
CSI_LEN = 56
total_len = 311
num_int16_per_trans = total_len*4

OUT_MAT_FULL = True        # new mat with full data

file = r"Result\\"

def parse_side_info(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN):
    CSI_LEN_HALF = round(CSI_LEN/2)
    #num_dma_symbol_per_trans = HEADER_LEN + CSI_LEN + num_eq*EQUALIZER_LEN
    num_dma_symbol_per_trans = HEADER_LEN + RSSI_LEN + CSI_LEN+CSI_LEN
    num_int16_rssi_agc_per_trans = (HEADER_LEN + RSSI_LEN ) * 4
    num_int16_per_trans = num_dma_symbol_per_trans*4 # 64bit per dma symbol
    num_trans = round(len(side_info)/num_int16_per_trans)
    side_info = side_info.reshape([num_trans, num_int16_per_trans])
    side_info_uint16 = np.array(side_info[:,:28], dtype='uint16')
    
    #side_info_fifo_wr_count = side_info_uint16[:,3]
    ht_flag_capture = side_info_uint16[:,2]&1
    fcs_valid = side_info_uint16[:,2]&2
    #match_cfg = side_info_uint16[:,2]&4
    ofdm_rx_count_state = side_info_uint16[:,2]&18
    band = side_info_uint16[:,1]
    channel = side_info_uint16[:,0]
    timestamp = side_info_uint16[:,4] + pow(2,16)*side_info_uint16[:,5] + pow(2,32)*side_info_uint16[:,6] + pow(2,48)*side_info_uint16[:,7]
    
    freq_offset = (20e6*side_info[:,8]/512)/(2*3.14159265358979323846)
    fc = side_info_uint16[:,12] #+ pow(2,16)*side_info_uint16[:,13] # + pow(2,32)*side_info_uint16[:,14] + pow(2,48)*side_info_uint16[:,15]
    dest_mac =  side_info_uint16[:,16] + pow(2,16)*side_info_uint16[:,17] + pow(2,32)*side_info_uint16[:,18] + pow(2,48)*side_info_uint16[:,19]
    src_mac = side_info_uint16[:,20] + pow(2,16)*side_info_uint16[:,21] + pow(2,32)*side_info_uint16[:,22] + pow(2,48)*side_info_uint16[:,23]
    gateway_mac = side_info_uint16[:,24] + pow(2,16)*side_info_uint16[:,25] + pow(2,32)*side_info_uint16[:,26] + pow(2,48)*side_info_uint16[:,27]

    csi = np.zeros((num_trans, CSI_LEN), dtype='int16')
    csi = csi + csi*1j
    ht_csi = np.zeros((num_trans, CSI_LEN), dtype='int16')
    ht_csi = ht_csi + ht_csi*1j
    ddc_i = np.zeros((num_trans, RSSI_LEN), dtype='int16')
    ddc_q = np.zeros((num_trans, RSSI_LEN), dtype='int16')
    rssi = np.zeros((num_trans, RSSI_LEN), dtype='int16')
    agc = np.zeros((num_trans, RSSI_LEN), dtype='int16')

    for i in range(num_trans):
        ddc_i[i,:] = side_info[i,29:(num_int16_rssi_agc_per_trans):4]
        ddc_q[i,:] = side_info[i,28:(num_int16_rssi_agc_per_trans):4] 

        if channel[i] < 32 : 
            rssi_correction = 153
        elif channel[i] <= 48 : 
            rssi_correction = 145
        else : 
            rssi_correction = 148
        rssi[i,:] = (side_info[i,31:(num_int16_rssi_agc_per_trans):4]>>1)-rssi_correction
        
        agc[i,:] = side_info[i,30:(num_int16_rssi_agc_per_trans):4] # + pow(2,16)*side_info[:,29] + pow(2,32)*side_info[:,30] + pow(2,48)*side_info[:,31]    
        
        tmp_vec_i = side_info[i,num_int16_rssi_agc_per_trans:(num_int16_per_trans-1):4]
        tmp_vec_q = side_info[i,num_int16_rssi_agc_per_trans+1:(num_int16_per_trans-1):4]
        tmp_vec = tmp_vec_i + tmp_vec_q*1j
        # csi[i,:] = tmp_vec[0:CSI_LEN]
        csi[i,:CSI_LEN_HALF] = tmp_vec[CSI_LEN_HALF:CSI_LEN]
        csi[i,CSI_LEN_HALF:] = tmp_vec[0:CSI_LEN_HALF]
        ht_csi[i,:CSI_LEN_HALF] = tmp_vec[CSI_LEN_HALF+CSI_LEN:CSI_LEN+CSI_LEN]
        ht_csi[i,CSI_LEN_HALF:] = tmp_vec[0+CSI_LEN:CSI_LEN_HALF+CSI_LEN]
    return band,channel,timestamp,freq_offset,fc,dest_mac,src_mac,gateway_mac,ddc_i,ddc_q,rssi,agc,csi,ofdm_rx_count_state,ht_csi,fcs_valid,ht_flag_capture

file_name = "Result\save_raw_data_to_sd_card.bin"
arr_time = np.array([])
arr_ch   = np.array([])
arr_freq = np.array([])
arr_fc   = np.array([])
arr_dest = np.array([])
arr_src  = np.array([])
arr_gate = np.array([])
arr_hw_i = np.empty([0, RSSI_LEN])
arr_hw_q = np.empty([0, RSSI_LEN])
arr_hw_rssi = np.empty([0, RSSI_LEN])
arr_hw_agc = np.empty([0, RSSI_LEN])
arr_hw_csi = np.empty([0, CSI_LEN])
with open(file_name, 'rb') as f:
    content = f.read()
    int_values = np.array([x for x in content],dtype='int16')
    side_info = int_values[0:len(int_values):2]+int_values[1:len(int_values):2]*(2**8)
    #side_info = np.array(list(map(int,f.readlines())),dtype='int16')
    band,channel,timestamp,freq_offset,fc,dest_mac,src_mac,gateway_mac,ddc_i,ddc_q,rssi,agc,csi,ofdm_rx_count_state,ht_csi,fcs_valid,ht_flag_capture= parse_side_info(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN)   
    num_trans = int(len(side_info)/num_int16_per_trans)
    #for i in range(num_trans):
    #    arr_time = np.append(arr_time, timestamp[i])
    #    arr_ch = np.append(arr_ch, channel[i])
    #    arr_freq = np.append(arr_freq, freq_offset[i])
    #    arr_fc = np.append(arr_fc, fc[i])
    #    arr_dest = np.append(arr_dest, dest_mac[i])
    #    arr_src  = np.append(arr_src, src_mac[i])
    #    arr_gate = np.append(arr_gate, gateway_mac[i])
    #    arr_hw_i = np.append(arr_hw_i, ddc_i[i,:][np.newaxis,:], axis=0)
    #    arr_hw_q = np.append(arr_hw_q, ddc_q[i,:][np.newaxis,:], axis=0)
    #    arr_hw_rssi = np.append(arr_hw_rssi, rssi[i,:][np.newaxis,:], axis=0)
    #    arr_hw_agc = np.append(arr_hw_agc, agc[i,:][np.newaxis,:], axis=0)
    #    arr_hw_csi = np.append(arr_hw_csi, csi[i,:][np.newaxis,:], axis=0)
    
print('load {} complit'.format(file_name))

if OUT_MAT_FULL:
    data_buf = {"TSF_bin": timestamp, \
                "chan_bin": channel  , \
                "freqOff_bin": freq_offset, \
                "FC_bin": fc  , \
                "dest_bin": dest_mac, \
                "src_bin": src_mac , \
                "gate_bin": gateway_mac, \
                "hw_i_bin": ddc_i, \
                "hw_q_bin": ddc_q, \
                "hw_rssi_bin": rssi, \
                "hw_csi_bin": csi}
    output_name = file_name[:-4] + "_bin.mat"
    savemat(output_name, data_buf)
    
    print('save {} complit'.format(output_name))
