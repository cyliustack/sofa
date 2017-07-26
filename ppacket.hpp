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
    char errbuff[PCAP_ERRBUF_SIZE];
    pcap_t* pcap = pcap_open_offline(path.c_str(), errbuff);
    struct pcap_pkthdr* header;
    const u_char* data;
    uint64_t packetCount = 0;
    while (pcap_next_ex(pcap, &header, &data) >= 0) {
        packetCount++;
        // Show a warning if the length captured is different
        if (header->len != header->caplen)
            printf("Warning! Capture size different than packet size: %ld bytes\n", header->len);

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
