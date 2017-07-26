#ifndef _INC_PPACKET_
#define _INC_PPACKET_
#include <iostream>
#include <map>
#include <pcap.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <vector>

typedef struct Packet {
    uint32_t type;
    uint32_t id;
    uint32_t nid;
    //Network Information
    uint8_t ip_src[4];
    uint8_t ip_dst[4];
    uint32_t port_src;
    uint32_t port_dst;
    uint32_t flag;
    uint32_t seq;
    uint32_t ack;
    uint64_t insns;
    uint32_t payload;
    uint32_t cputime_us;
    uint32_t nettime_us;
    uint32_t nid_src;
    uint32_t nid_dst;
    uint32_t lt;
    double timestamp;
} Packet;

class PcapFile {
public:
    std::string path;
    std::string color;
    FILE* pFile;
    std::vector<Packet> packets;
    bool parse_from_file();
    int parse();
    double max_wct;
    double min_wct;
};

bool PcapFile::parse_from_file()
{
    pFile = fopen(path.c_str(), "r");
    if (pFile == NULL) {
        printf("[Warning] No pcap file\n");
        return false; 
    } else {
        fclose(pFile);
        parse();
    }
    return true;
}

int PcapFile::parse()
{
    Packet packet;
    // Note: errbuf in pcap_open functions is assumed to be able to hold at least PCAP_ERRBUF_SIZE chars
    //       PCAP_ERRBUF_SIZE is defined as 256.
    // http://www.winpcap.org/docs/docs_40_2/html/group__wpcap__def.html
    char errbuff[PCAP_ERRBUF_SIZE];

    // Use pcap_open_offline
    // http://www.winpcap.org/docs/docs_41b5/html/group__wpcapfunc.html#g91078168a13de8848df2b7b83d1f5b69
    pcap_t* pcap = pcap_open_offline(path.c_str(), errbuff);

    /*
    * Step 5 - Create a header and a data object
    */

    // Create a header object:
    // http://www.winpcap.org/docs/docs_40_2/html/structpcap__pkthdr.html
    struct pcap_pkthdr* header;

    // Create a character array using a u_char
    // u_char is defined here:
    // C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\Include\WinSock2.h
    // typedef unsigned char   u_char;
    const u_char* data;

    /*
    * Step 6 - Loop through packets and print them to screen
    */
    uint64_t packetCount = 0;
    while (pcap_next_ex(pcap, &header, &data) >= 0) {
        // Print using printf. See printf reference:
        // http://www.cplusplus.com/reference/clibrary/cstdio/printf/

        // Show the packet number

        printf("Packet # %i\n", ++packetCount);

        // Show the size in bytes of the packet
        //printf("Packet size: %d bytes\n", header->len);

        // Show a warning if the length captured is different
        if (header->len != header->caplen)
            printf("Warning! Capture size different than packet size: %ld bytes\n", header->len);

        // Show Epoch Time
        //printf("Epoch Time: %d.%d seconds\n", header->ts.tv_sec, header->ts.tv_usec);

        // loop through the packet and print it as hexidecimal representations of octets
        // We also have a function that does this similarly below: PrintData()
        //for (u_int i = 0; (i < header->caplen); i++) {
        //    // Start printing on the next after every 16 octets
        //    if ((i % 16) == 0)
        //        printf("\n");

        //    // Print each octet as hex (x), make sure there is always two characters (.2).
        //    printf("%.2x ", data[i]);
        //}
        //getchar();

        // Add two lines between packets
        //printf("\n\n");

        packet.id = packetCount;
        packet.nid = 0;
        packet.type = 4;
        packet.nid_src = data[29];
        packet.nid_dst = data[33];
        packet.seq = (data[38] << 24) + (data[39] << 16) + (data[40] << 8) + data[41];
        packet.ack = (data[42] << 24) + (data[43] << 16) + (data[44] << 8) + data[45];
        packet.flag = data[20];
        packet.port_src = (data[34] << 8) + data[35];
        packet.port_dst = (data[36] << 8) + data[37];
        packet.payload = header->len;
        packet.timestamp = header->ts.tv_sec + header->ts.tv_usec / 1000000.0;
        packet.insns = 0;
        /*
        printf("id:%u, nid:%u, nidsrc:%u, niddst:%u, seq:%u, ack:%u, flag:%u, portsrc:%u, portdst:%u, insns:%lu, payload:%u, ts:%lf \n\n",
            packet.id,
            packet.nid,
            packet.nid_src,
            packet.nid_dst,
            packet.seq,
            packet.ack,
            packet.flag,
            packet.port_src,
            packet.port_dst,
            packet.insns,
            packet.payload,
            packet.timestamp);
            */
        packets.push_back(packet);
    }

    if (packets.size() > 0) {
        this->min_wct = packets.at(0).timestamp;
        this->max_wct = packets.at(packets.size() - 1).timestamp;
    } else {
        printf("Empty packets!\n");
        return -1;
    }
}
#endif
