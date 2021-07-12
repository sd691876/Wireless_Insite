#
# openwifi side info receive and display program
# Xianjun jiao. putaoshu@msn.com; xianjun.jiao@imec.be
#
import os
import sys
import socket
import numpy as np
import matplotlib.pyplot as plt

def display_side_info(freq_offset, csi, ht_csi, equalizer, CSI_LEN, EQUALIZER_LEN):
    if not hasattr(display_side_info, 'freq_offset_store'):
        display_side_info.freq_offset_store = np.zeros((256,))

    len_freq_offset = len(freq_offset)
    display_side_info.freq_offset_store[:(256-len_freq_offset)] = display_side_info.freq_offset_store[len_freq_offset:]
    display_side_info.freq_offset_store[(256-len_freq_offset):] = freq_offset
    
    #fig_freq_offset = plt.figure(0)
    #fig_freq_offset.clf()
    #plt.xlabel("packet idx")
    #plt.ylabel("Hz")
    #plt.title("freq offset")
    #plt.plot(display_side_info.freq_offset_store)
    #fig_freq_offset.show()
    #plt.pause(0.0001)
    csi_for_plot = csi.T
    ht_csi_for_plot = ht_csi.T

    if (1):
        fig_csi = plt.figure(1)
        fig_csi.clf()
        #L-csi
        ax_abs_csi = fig_csi.add_subplot(411)
        #ax_abs_csi.set_xlabel("subcarrier idx")
        ax_abs_csi.set_ylabel("abs")
        ax_abs_csi.set_title("L-CSI")
        plt.plot(np.abs(csi_for_plot))
        ax_phase_csi = fig_csi.add_subplot(412)
        #ax_phase_csi.set_xlabel("subcarrier idx")
        ax_phase_csi.set_ylabel("phase")
        plt.plot(np.angle(csi_for_plot))
        #ht-csi
        ax_abs_ht_csi = fig_csi.add_subplot(413)
        #ax_abs_ht_csi.set_xlabel("subcarrier idx")
        ax_abs_ht_csi.set_ylabel("abs")
        ax_abs_ht_csi.set_title("HT-CSI")
        plt.plot(np.abs(ht_csi_for_plot))
        ax_phase_ht_csi = fig_csi.add_subplot(414)
        ax_phase_ht_csi.set_xlabel("subcarrier idx")
        ax_phase_ht_csi.set_ylabel("phase")
        plt.plot(np.angle(ht_csi_for_plot))

        fig_csi.show()
        plt.pause(0.0001)

def display_iq(iq_capture):
    fig_iq_capture = plt.figure(3)
    fig_iq_capture.clf()
    plt.xlabel("sample")
    plt.ylabel("I/Q")
    plt.title("I (blue) and Q (red) capture")
    plt.plot(iq_capture.real, 'b')
    plt.plot(iq_capture.imag, 'r')
    plt.ylim(-32767, 32767)
    fig_iq_capture.show()
    plt.pause(0.0001)


def parse_side_info(side_info, num_eq, CSI_LEN, EQUALIZER_LEN, HEADER_LEN, RSSI_LEN):
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

def parse_header(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN):
    num_dma_symbol_per_trans = HEADER_LEN + RSSI_LEN + CSI_LEN+CSI_LEN
    num_int16_per_trans = num_dma_symbol_per_trans*4 # 64bit per dma symbol
    num_trans = round(len(side_info)/num_int16_per_trans)
    side_info = side_info.reshape([num_trans, num_int16_per_trans])
    side_info_uint16 = np.array(side_info[:,:28], dtype='uint16')
    
    channel = side_info_uint16[:,0]
    timestamp = side_info_uint16[:,4] + pow(2,16)*side_info_uint16[:,5] + pow(2,32)*side_info_uint16[:,6] + pow(2,48)*side_info_uint16[:,7]
    fc = side_info_uint16[:,12] #+ pow(2,16)*side_info_uint16[:,13] # + pow(2,32)*side_info_uint16[:,14] + pow(2,48)*side_info_uint16[:,15]
    dest_mac =  side_info_uint16[:,16] + pow(2,16)*side_info_uint16[:,17] + pow(2,32)*side_info_uint16[:,18] + pow(2,48)*side_info_uint16[:,19]
    src_mac = side_info_uint16[:,20] + pow(2,16)*side_info_uint16[:,21] + pow(2,32)*side_info_uint16[:,22] + pow(2,48)*side_info_uint16[:,23]
    gateway_mac = side_info_uint16[:,24] + pow(2,16)*side_info_uint16[:,25] + pow(2,32)*side_info_uint16[:,26] + pow(2,48)*side_info_uint16[:,27]
    
    return timestamp, channel, fc, dest_mac, src_mac, gateway_mac

UDP_IP = "192.168.10.1" #Local IP to listen
UDP_PORT = 4000         #Local port to listen

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

# align with side_ch_control.v and all related user space, remote files
MAX_NUM_DMA_SYMBOL = 8192
CSI_LEN = 56 # length of single CSI
EQUALIZER_LEN = (56-4) # for non HT, four {32767,32767} will be padded to achieve 52 (non HT should have 48)
HEADER_LEN = 7 #2 # timestamp and frequency offset
RSSI_LEN = 192
IQ_LEN = 192

cnt = 0

#if len(sys.argv)<2:
#    print("Assume num_eq = 8!")
#    num_eq = 8
#else:
#    num_eq = int(sys.arg

#num_dma_symbol_per_trans = HEADER_LEN + CSI_LEN + num_eq*EQUALIZER_LEN
num_dma_symbol_per_trans = HEADER_LEN + RSSI_LEN + CSI_LEN+CSI_LEN 
num_byte_per_trans = 8*num_dma_symbol_per_trans
num_int16_per_trans = num_dma_symbol_per_trans*4 # 64bit per dma symbol

filename = "Result/0708_all_continuous"
file_txt = filename + "_scaninfo.txt"
file_npy = filename + ".npy"

if os.path.exists(file_txt):
    os.remove(file_txt)
simple_fd=open(file_txt,'a')
if os.path.exists(file_npy):
    os.remove(file_npy);
all_fd=open(file_npy,'ab');

while True:
    try:
        data, addr = sock.recvfrom(MAX_NUM_DMA_SYMBOL*8) # buffer size
        
        #if len(data)>0 : 
        #    print(addr)
        #    print("data len :",len(data), num_byte_per_trans)
        test_residual = len(data)%num_byte_per_trans
        if (test_residual != 0):
            print("Abnormal length")

        side_info = np.frombuffer(data, dtype='int16')
        np.save(all_fd, side_info)
        cnt = cnt+1
        
        num_trans = round(len(side_info)/num_int16_per_trans)

        timestamp, channel, fc, dest_mac, src_mac, gateway_mac = parse_header(side_info, CSI_LEN, HEADER_LEN, RSSI_LEN)
        
        #band,channel,timestamp,freq_offset,fc,dest_mac,src_mac,gateway_mac,ddc_i,ddc_q,rssi,agc,csi,ofdm_rx_count_state,ht_csi,fcs_valid,ht_flag_capture= parse_side_info(side_info, num_eq, CSI_LEN, EQUALIZER_LEN, HEADER_LEN, RSSI_LEN)
        #iq_capture = ddc_i[:,:] + ddc_q[:,:]*1j
        #iq_capture = iq_capture.reshape([num_trans*RSSI_LEN])
        #equalizer = np.zeros((0,0), dtype='int16')
        
        for i in range(num_trans):
            if channel[i] !=1 :
                string = '# {}: Time: {} Channel: {} FC: 0x{:02x} DEST: 0x{:012x} SRC: 0x{:012x} GATE: 0x{:012x}\n'.format(cnt, timestamp[i],channel[i],fc[i],dest_mac[i],src_mac[i],gateway_mac[i]) 
                print(string)
                simple_fd.write(string)
                
    except KeyboardInterrupt:
        print('User quit')
        break

print('close()')
simple_fd.close()
all_fd.close()
sock.close()
