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
SC_TRANS = [52,56]

CALL_WEN_CSI = True
OUT_TXT_RAW = False        # iq raw txt
OUT_MAT_FULL = True        # new mat with full data
OUT_MAT_IQ = False         # new mat with iq only

folder = r"Result\\"

def parse_side_info(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN):
    CSI_LEN_HALF = round(CSI_LEN/2)
    #num_dma_symbol_per_trans = HEADER_LEN + CSI_LEN + num_eq*EQUALIZER_LEN
    num_dma_symbol_per_trans = HEADER_LEN + RSSI_LEN + CSI_LEN+CSI_LEN
    num_int16_rssi_agc_per_trans = (HEADER_LEN + RSSI_LEN ) * 4
    num_int16_per_trans = num_dma_symbol_per_trans*4 # 64bit per dma symbol
    num_trans = round(len(side_info)/num_int16_per_trans)
    side_info = side_info.reshape([num_trans, num_int16_per_trans])
    side_info_uint16 = np.array(side_info[:,:28], dtype='uint16')
    
    side_info_fifo_wr_count = side_info_uint16[:,3]
    ht_flag_capture = side_info_uint16[:,2]&1
    fcs_valid = side_info_uint16[:,2]&2
    match_cfg = side_info_uint16[:,2]&4
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

    #equalizer = np.zeros((0,0), dtype='int16')
    #if num_eq>0:
    #    equalizer = np.zeros((num_trans, num_eq*EQUALIZER_LEN), dtype='int16')
    #    equalizer = equalizer + equalizer*1j

    #form sdr.c ad9361_rf_set_channel() & openwifi_rc_interrupt()
    #if (actual_rx_lo<2412) {
    #    priv->rssi_correction = 153;
    #} else if (actual_rx_lo<=2484) {
    #    priv->rssi_correction = 153;
    #} else if (actual_rx_lo<5160) {
    #    priv->rssi_correction = 153;
    #} else if (actual_rx_lo<=5240) {
    #    priv->rssi_correction = 145;
    #} else if (actual_rx_lo<=5320) {
    #    priv->rssi_correction = 148;
    #} else {
    #    priv->rssi_correction = 148;
    #}
    #rssi_val = (rssi_val>>1);
    #if ( (rssi_val+128)<priv->rssi_correction )
    #    signal = -128;
    #else
    #    signal = rssi_val - priv->rssi_correction;

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
        #if num_eq>0:
        #    equalizer[i,:] = tmp_vec[CSI_LEN:(CSI_LEN+num_eq*EQUALIZER_LEN)]
        #print(i, len(tmp_vec), len(tmp_vec[0:CSI_LEN]), len(tmp_vec[CSI_LEN:(CSI_LEN+num_eq*EQUALIZER_LEN)]))
    #print("num_trans :",i,"CSI_LEN :",len(tmp_vec),"ddc_i :",len(ddc_i[i,:]),"ddc_q :",len(ddc_q[i,:]),"ht_flag_capture:",ht_flag_capture,"match_cfg:",match_cfg,"ofdm_rx_count_state:",ofdm_rx_count_state, "fcs_valid: ",fcs_valid)

    return band,channel,timestamp,freq_offset,fc,dest_mac,src_mac,gateway_mac,ddc_i,ddc_q,rssi,agc,csi,ofdm_rx_count_state,ht_csi,fcs_valid,ht_flag_capture
   
files = glob.glob(folder + "*.npy")
for file_name in files:
    
    list_iq = []
    arr_time = np.array([])
    arr_ch   = np.array([])
    arr_freq = np.array([])
    arr_fc   = np.array([])
    arr_dest = np.array([])
    arr_src  = np.array([])
    arr_gate = np.array([])
    arr_hw_i = np.empty([0, 192])
    arr_hw_q = np.empty([0, 192])
    arr_hw_rssi = np.empty([0, 192])
    arr_hw_agc = np.empty([0, 192])
    arr_hw_csi = np.empty([0, 56])
    with open(file_name, 'rb') as f:
        while True:
            try:
                side_info = np.load(f)
                list_iq.append(side_info)
                band,channel,timestamp,freq_offset,fc,dest_mac,src_mac,gateway_mac,ddc_i,ddc_q,rssi,agc,csi,ofdm_rx_count_state,ht_csi,fcs_valid,ht_flag_capture= parse_side_info(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN)
                num_trans = round(len(side_info)/num_int16_per_trans)
                for i in range(num_trans):
                    arr_time = np.append(arr_time, timestamp[i])
                    arr_ch = np.append(arr_ch, channel[i])
                    arr_freq = np.append(arr_freq, freq_offset[i])
                    arr_fc = np.append(arr_fc, fc[i])
                    arr_dest = np.append(arr_dest, dest_mac[i])
                    arr_src  = np.append(arr_src, src_mac[i])
                    arr_gate = np.append(arr_gate, gateway_mac[i])
                    #pdb.set_trace()
                    arr_hw_i = np.append(arr_hw_i, ddc_i[i,:][np.newaxis,:], axis=0)
                    arr_hw_q = np.append(arr_hw_q, ddc_q[i,:][np.newaxis,:], axis=0)
                    arr_hw_rssi = np.append(arr_hw_rssi, rssi[i,:][np.newaxis,:], axis=0)
                    arr_hw_agc = np.append(arr_hw_agc, agc[i,:][np.newaxis,:], axis=0)
                    arr_hw_csi = np.append(arr_hw_csi, csi[i,:][np.newaxis,:], axis=0)
                    #pdb.set_trace()
            except:
                break
    
    print('load {} complit'.format(file_name))
    #pdb.set_trace()
    
    
    if OUT_TXT_RAW:
        iq_out = np.stack(list_iq).flatten()
    
        output_name = file_name[:-4] + r"_out.txt"
        if os.path.exists(output_name):
            os.remove(output_name)
        np.savetxt(output_name, iq_out)
        
        print('save {} complit'.format(output_name))
        
    if OUT_MAT_FULL:
        data_buf = {"TSF": arr_time, \
                    "chan": arr_ch  , \
                    "freqOff": arr_freq, \
                    "FC": arr_fc  , \
                    "dest": arr_dest, \
                    "src": arr_src , \
                    "gate": arr_gate, \
                    "hw_i": arr_hw_i, \
                    "hw_q": arr_hw_q, \
                    "hw_rssi": arr_hw_rssi, \
                    "hw_csi": arr_hw_csi}
        output_name = file_name[:-4] + "_full.mat"
        savemat(output_name, data_buf)
        
        print('save {} complit'.format(output_name))
        
    
